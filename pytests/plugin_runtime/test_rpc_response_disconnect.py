from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import src.plugin_runtime.runner.rpc_client as module
from src.plugin_runtime.protocol.envelope import Envelope, MessageType
from src.plugin_runtime.runner.rpc_client import RPCClient


@pytest.mark.asyncio
async def test_disconnect_after_handler_does_not_repeat_response_or_log_handler_error(monkeypatch):
    client = RPCClient("unused", "test")
    connection = SimpleNamespace(is_closed=False, send_frame=AsyncMock(side_effect=ConnectionResetError("lost")))
    client._connection = connection
    handler = AsyncMock(side_effect=lambda envelope: envelope.make_response({"success": True}))
    client.register_method("plugin.unregister", handler)
    log = Mock()
    monkeypatch.setattr(module, "logger", log)
    envelope = Envelope(request_id=1, message_type=MessageType.REQUEST, method="plugin.unregister", payload={})
    await client._handle_request(envelope)
    handler.assert_awaited_once()
    connection.send_frame.assert_awaited_once()
    log.error.assert_not_called()


@pytest.mark.asyncio
async def test_real_handler_bug_still_logs_error_and_returns_error_response(monkeypatch):
    client = RPCClient("unused", "test")
    connection = SimpleNamespace(is_closed=False, send_frame=AsyncMock())
    client._connection = connection
    client.register_method("test", AsyncMock(side_effect=ValueError("handler bug")))
    log = Mock()
    monkeypatch.setattr(module, "logger", log)
    envelope = Envelope(request_id=1, message_type=MessageType.REQUEST, method="test", payload={})
    await client._handle_request(envelope)
    log.error.assert_called_once()
    result = client._codec.decode_envelope(connection.send_frame.call_args.args[0])
    assert result.error["code"] == "E_UNKNOWN"
