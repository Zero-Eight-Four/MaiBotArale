"""Reproduce reply stalls and routing mistakes without model calls or QQ sends."""

from pathlib import Path
from types import SimpleNamespace
import asyncio
import importlib
import sys
import types

import pytest

from src.maisaka.reasoning_engine import MaisakaReasoningEngine
from src.maisaka.runtime import MaisakaHeartFlowChatting
from src.maisaka.turn_trigger.scheduler import MessageTurnScheduler


@pytest.mark.asyncio
async def test_cancelled_preparation_releases_schedule_and_new_message_can_queue(monkeypatch):
    r = SimpleNamespace(
        _running=True,
        session_id="s",
        log_prefix="[test]",
        chat_stream=SimpleNamespace(is_group_session=True),
        _message_turn_scheduled=True,
        _internal_turn_queue=asyncio.Queue(),
        _agent_state="stop",
        _STATE_WAIT="wait",
    )
    r._mark_message_turn_unscheduled = lambda: setattr(r, "_message_turn_scheduled", False)
    r._log_internal_loop_cancelled = lambda: None
    r._get_pending_message_count = lambda: 1
    r._get_message_quiet_period_remaining_seconds = lambda: 0.0
    r._get_effective_reply_frequency = lambda: 0
    r._is_reply_frequency_silent = lambda: True
    r._cancel_deferred_message_turn_task = lambda: None
    r._enqueue_message_turn = lambda: MaisakaHeartFlowChatting._enqueue_message_turn(r)
    started = asyncio.Event()

    async def prepare(trigger):
        started.set()
        await asyncio.Event().wait()

    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = r
    engine._prepare_turn_start_context = prepare
    r._internal_turn_queue.put_nowait("message")
    task = asyncio.create_task(engine.run_loop())
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not r._message_turn_scheduled
    await MessageTurnScheduler(r).schedule_message_turn()
    assert r._internal_turn_queue.get_nowait() == "message"


def poke_plugin():
    name = "regression_smart_poke"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path("plugins/arale-poke").resolve())]
        sys.modules[name] = package
    return importlib.import_module(name + ".plugin").SmartPokePlugin()


def test_poke_uses_host_account_when_notice_additional_has_no_route_fields():
    plugin = poke_plugin()
    ctx = plugin._extract_poke_context(
        {
            "is_notify": True,
            "session_id": "actual",
            "account_id": "bot-account",
            "scope": "connection",
            "message_info": {
                "additional_config": {
                    "napcat_notice_type": "notify",
                    "napcat_notice_sub_type": "poke",
                    "napcat_notice_payload": {
                        "self_id": "bot-account",
                        "user_id": "owner",
                        "target_id": "bot-account",
                    },
                }
            },
        }
    )
    assert ctx.account_id == "bot-account" and ctx.scope == "connection"
