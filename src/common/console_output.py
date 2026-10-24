"""Keep journald/pipe backpressure away from application event loops.

JSONL logs remain the complete record. Only the redundant console output is
bounded; an unavailable console must not stop QQ reception or Runner health.
"""

from functools import lru_cache
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import TextIO

import atexit
import time


class QueuedConsoleStream:
    def __init__(self, stream: TextIO, *, max_pending: int = 256) -> None:
        self._stream = stream
        self._queue: Queue[str] = Queue(maxsize=max_pending)
        self._lock = Lock()
        self._dropped = 0
        self._last_error: str | None = None
        self._closed = False
        self._worker = Thread(target=self._write_loop, name="MaiBot.console-writer", daemon=True)
        self._worker.start()
        atexit.register(self._shutdown)

    @property
    def encoding(self):
        return self._stream.encoding

    @property
    def closed(self) -> bool:
        return self._closed

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._stream.fileno()

    def write(self, text: str) -> int:
        if self._closed:
            raise ValueError("write to closed console stream")
        if text:
            try:
                self._queue.put_nowait(text)
            except Full:
                with self._lock:
                    self._dropped += 1
        return len(text)

    def flush(self) -> None:
        # logging.StreamHandler and Rich flush on every output; neither may wait
        # for journald. The worker flushes the underlying stream after writing.
        pass

    def wait_idle(self, timeout: float) -> bool:
        """Bounded drain for explicit shutdown/tests, never for each log call."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    def close(self) -> None:
        self._closed = True

    def _shutdown(self) -> None:
        self.wait_idle(0.2)
        self.close()

    def _write_loop(self) -> None:
        while not self._closed or not self._queue.empty():
            try:
                text = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                with self._lock:
                    dropped, self._dropped = self._dropped, 0
                if dropped:
                    self._stream.write(
                        f"[console] 控制台拥塞，省略 {dropped} 段输出；完整应用日志保存在 logs/*.jsonl。\n"
                    )
                self._stream.write(text)
                self._stream.flush()
            except (OSError, ValueError) as exc:
                self._last_error = str(exc)
            finally:
                self._queue.task_done()


@lru_cache(maxsize=4)
def get_console_stream(stream: TextIO) -> TextIO:
    """Interactive terminals retain immediate output; service pipes use a queue."""
    if stream.isatty():
        return stream
    return QueuedConsoleStream(stream)
