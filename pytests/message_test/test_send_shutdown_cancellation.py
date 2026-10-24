from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio

import pytest

import src.services.send_service as module


@pytest.mark.asyncio
@pytest.mark.parametrize("during_prepare", [False, True])
async def test_shutdown_never_sends_to_platform(monkeypatch, during_prepare):
    state = {"shutdown": not during_prepare}
    message = SimpleNamespace(message_id="test")
    hook = AsyncMock(return_value=(SimpleNamespace(aborted=False, kwargs={}), message))
    manager = SimpleNamespace(ensure_send_pipeline_ready=AsyncMock(),
        build_route_key_from_message=Mock(return_value="route"), send_message=AsyncMock())

    async def prepare(*args, **kwargs):
        state["shutdown"] = True

    monkeypatch.setattr(module, "is_shutdown_requested", lambda: state["shutdown"])
    monkeypatch.setattr(module, "_invoke_send_hook", hook)
    monkeypatch.setattr(module, "get_platform_io_manager", lambda: manager)
    monkeypatch.setattr(module, "_prepare_message_for_platform_io", prepare)
    with pytest.raises(asyncio.CancelledError):
        await module._send_via_platform_io(message, typing=True, set_reply=False,
            reply_message_id=None, storage_message=False, show_log=False)
    manager.send_message.assert_not_awaited()
    if not during_prepare:
        hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_delivery_is_saved_if_shutdown_starts_after_send(monkeypatch):
    state = {"shutdown": False}
    message = SimpleNamespace(message_id="real-platform-id")
    hook = AsyncMock(return_value=(SimpleNamespace(aborted=False, kwargs={}), message))
    store = AsyncMock()
    batch = SimpleNamespace(has_success=True, sent_receipts=[])

    async def deliver(*args, **kwargs):
        state["shutdown"] = True
        return batch

    manager = SimpleNamespace(ensure_send_pipeline_ready=AsyncMock(),
        build_route_key_from_message=Mock(return_value="route"), send_message=AsyncMock(side_effect=deliver))
    monkeypatch.setattr(module, "is_shutdown_requested", lambda: state["shutdown"])
    monkeypatch.setattr(module, "_invoke_send_hook", hook)
    monkeypatch.setattr(module, "get_platform_io_manager", lambda: manager)
    monkeypatch.setattr(module, "_prepare_message_for_platform_io", AsyncMock())
    monkeypatch.setattr(module, "_apply_successful_delivery_receipt", AsyncMock())
    monkeypatch.setattr(module, "_store_sent_message", store)
    monkeypatch.setattr(module, "_record_sent_emoji_usage", Mock())
    callbacks, memory = AsyncMock(), AsyncMock()
    monkeypatch.setattr(module, "_dispatch_adapter_callbacks", callbacks)
    monkeypatch.setattr(module, "_notify_memory_automation_on_message_sent", memory)
    result = await module._send_via_platform_io(message, typing=False, set_reply=False,
        reply_message_id=None, storage_message=True, show_log=False)
    assert result is message
    store.assert_awaited_once_with(message)
    assert hook.await_count == 1
    callbacks.assert_not_awaited()
    memory.assert_not_awaited()
