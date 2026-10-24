"""Request deadlines and owner-scoped cancellation for both sides of IPC."""
from typing import Awaitable, Callable
import asyncio
import contextlib

from src.plugin_runtime.protocol.envelope import Envelope, MessageType


class IncomingRequests:
    def __init__(self) -> None:
        self._tasks = {}

    def start(self, envelope: Envelope, connection, handler: Callable[[], Awaitable]) -> asyncio.Task:
        key = (id(connection), envelope.request_id)
        task = asyncio.create_task(handler())
        self._tasks[key] = (envelope.plugin_id, task)

        def finished(done):
            if self._tasks.get(key) == (envelope.plugin_id, done):
                self._tasks.pop(key, None)

        task.add_done_callback(finished)
        return task

    def cancel(self, envelope: Envelope, connection) -> None:
        request_id = envelope.payload.get("request_id")
        if type(request_id) is not int:
            return
        entry = self._tasks.get((id(connection), request_id))
        if entry is not None and entry[0] == envelope.plugin_id:
            entry[1].cancel()

    @staticmethod
    async def invoke(envelope: Envelope, handler):
        # The caller's budget bounds the complete remote operation, including
        # model selection, fallback, and HTTP work, rather than just its Future.
        seconds = envelope.timeout_ms / 1000 if envelope.timeout_ms > 0 else None
        async with asyncio.timeout(seconds):
            return await handler(envelope)


async def notify_cancel(connection, codec, envelope: Envelope) -> None:
    if connection is None or connection.is_closed:
        return
    event = Envelope(
        request_id=envelope.request_id,
        message_type=MessageType.BROADCAST,
        method="rpc.cancel",
        plugin_id=envelope.plugin_id,
        payload={"request_id": envelope.request_id},
    )
    # Cancellation notification must never replace the caller's timeout/error.
    with contextlib.suppress(Exception):
        await asyncio.wait_for(connection.send_frame(codec.encode_envelope(event)), timeout=.5)
