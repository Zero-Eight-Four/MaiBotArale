from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio

import pytest

import src.plugin_runtime.host.hook_dispatcher as module
from src.plugin_runtime.host.hook_dispatcher import HookDispatcher
from src.plugin_runtime.host.hook_spec_registry import HookSpec
from src.plugin_runtime.protocol.errors import ErrorCode, RPCError


@pytest.mark.asyncio
async def test_shutdown_cancels_hook_before_any_plugin_or_policy_is_run(monkeypatch):
    dispatcher = HookDispatcher()
    monkeypatch.setattr(module, "is_shutdown_requested", lambda: True)
    with pytest.raises(asyncio.CancelledError):
        await dispatcher.invoke_hook("send_service.before_send", message={})


@pytest.mark.asyncio
@pytest.mark.parametrize("shutting_down", [False, True])
@pytest.mark.parametrize("error_code", [ErrorCode.E_SHUTTING_DOWN, ErrorCode.E_PLUGIN_CRASHED])
async def test_inflight_rpc_cancels_shutdown_but_preserves_real_crashes(monkeypatch, shutting_down, error_code):
    dispatcher = HookDispatcher()
    entry = SimpleNamespace(full_name="test.shutdown.hook", plugin_id="test.shutdown", name="hook")
    state = {"shutdown": False}

    async def stopping_rpc(*args, **kwargs):
        state["shutdown"] = shutting_down
        raise RPCError(error_code, "stopping")

    target = SimpleNamespace(entry=entry, supervisor=SimpleNamespace(invoke_plugin=AsyncMock(side_effect=stopping_rpc)))
    monkeypatch.setattr(dispatcher, "_resolve_timeout_ms", lambda spec, target: 1000)
    monkeypatch.setattr(module, "is_shutdown_requested", lambda: state["shutdown"])
    if shutting_down or error_code == ErrorCode.E_SHUTTING_DOWN:
        with pytest.raises(asyncio.CancelledError):
            await dispatcher._invoke_handler(hook_name="test", hook_spec=HookSpec("test"), target=target, kwargs={})
    else:
        result = await dispatcher._invoke_handler(hook_name="test", hook_spec=HookSpec("test"), target=target, kwargs={})
        assert not result.success
        assert "E_PLUGIN_CRASHED" in result.error_message
