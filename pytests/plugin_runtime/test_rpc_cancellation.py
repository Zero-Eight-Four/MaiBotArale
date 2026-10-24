"""Exercise cancellation across the actual Host/Runner request boundaries."""
import asyncio
from unittest.mock import Mock

import pytest

from src.plugin_runtime.host.rpc_server import RPCServer
from src.plugin_runtime.runner.rpc_client import RPCClient
from src.plugin_runtime.protocol.errors import RPCError
from src.plugin_runtime.protocol.envelope import Envelope, MessageType
from src.plugin_runtime.request_lifecycle import IncomingRequests


class Wire:
    is_closed = False

    def __init__(self):
        self.queue = asyncio.Queue()

    async def send_frame(self, data):
        await self.peer.queue.put(data)

    async def recv_frame(self):
        return await self.queue.get()


@pytest.mark.asyncio
@pytest.mark.parametrize('direction', ['runner_to_host', 'host_to_runner'])
@pytest.mark.parametrize('timeout', [False, True])
async def test_remote_work_stops_when_caller_cancels_or_times_out(direction, timeout):
    server = RPCServer(Mock())
    client = RPCClient('unused', 'test')
    a, b = Wire(), Wire()
    a.peer, b.peer = b, a
    server._connection, client._connection = a, b
    server._running = client._running = True
    caller, receiver = (client, server) if direction == 'runner_to_host' else (server, client)
    started, stopped = asyncio.Event(), asyncio.Event()

    async def slow(envelope):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    receiver.register_method('slow', slow)
    loops = [asyncio.create_task(server._recv_loop(a)), asyncio.create_task(client._recv_loop())]
    request = asyncio.create_task(caller.send_request('slow', plugin_id='owner', timeout_ms=50 if timeout else 5000))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if timeout:
            with pytest.raises(RPCError):
                await request
        else:
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
        await asyncio.wait_for(stopped.wait(), .5)
        assert not caller._pending_requests
    finally:
        server._running = client._running = False
        for task in [request, *loops, *server._tasks, *client._background_tasks]:
            task.cancel()
        await asyncio.gather(request, *loops, *server._tasks, *client._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_cannot_target_another_plugin_or_connection():
    scope, connection = IncomingRequests(), object()
    request = Envelope(request_id=7, message_type=MessageType.REQUEST, method='slow', plugin_id='owner')
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.Event().wait()

    task = scope.start(request, connection, slow)
    await started.wait()
    cancel = Envelope(request_id=8, message_type=MessageType.BROADCAST, method='rpc.cancel',
                      plugin_id='other', payload={'request_id': 7})
    scope.cancel(cancel, connection)
    cancel.plugin_id = 'owner'
    scope.cancel(cancel, object())
    await asyncio.sleep(0)
    assert not task.done()
    scope.cancel(cancel, connection)
    with pytest.raises(asyncio.CancelledError):
        await task
