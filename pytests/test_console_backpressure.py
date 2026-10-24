from io import StringIO
from threading import Event

import asyncio

import pytest

from src.common.console_output import QueuedConsoleStream, get_console_stream


@pytest.mark.asyncio
async def test_blocked_console_does_not_block_heartbeat_or_grow_queue():
    entered, release = Event(), Event()

    class BlockedSink(StringIO):
        def write(self, text):
            entered.set()
            release.wait(timeout=3)
            return super().write(text)

    sink = BlockedSink()
    stream = QueuedConsoleStream(sink, max_pending=2)
    try:
        stream.write("first\n")
        assert await asyncio.to_thread(entered.wait, 1)
        heartbeats = 0
        for i in range(20):
            stream.write(f"line {i}\n")
            stream.flush()
            await asyncio.sleep(0)
            heartbeats += 1
        assert heartbeats == 20
        assert stream._queue.qsize() <= 2
        assert not stream.wait_idle(0.01)
    finally:
        release.set()
        assert await asyncio.to_thread(stream.wait_idle, 1)
        stream.close()
    assert "first" in sink.getvalue()
    assert "完整应用日志" in sink.getvalue()


def test_interactive_console_remains_immediate():
    class Terminal(StringIO):
        def isatty(self):
            return True

    sink = Terminal()
    assert get_console_stream(sink) is sink


def test_service_console_preserves_order():
    sink = StringIO()
    stream = QueuedConsoleStream(sink)
    try:
        stream.write("first\n")
        stream.write("second\n")
        assert stream.wait_idle(1)
        assert sink.getvalue() == "first\nsecond\n"
    finally:
        stream.close()
