"""Snapshots remain detached while asynchronous policies and later messages overlap."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio

import pytest

from pytests.maisaka.reply_fixtures import _message
from src.common.data_models.message_component_data_model import MessageSequence, ReplyComponent
from src.maisaka.runtime import MaisakaHeartFlowChatting
from src.plugin_runtime import context_views, policy_bridge


@pytest.mark.asyncio
async def test_overlapping_flow_calls_use_their_own_snapshot(monkeypatch):
    original = _message("original")
    original.extension_traits = {"direct_request": True}
    state = SimpleNamespace(
        session_id="chat",
        _context_revision=7,
        _input_version=1,
        _extension_sequence=0,
        _last_message_received_at=123.0,
        _last_processed_index=0,
        chat_stream=SimpleNamespace(is_group_session=True),
        message_cache=[original],
    )
    started, release = asyncio.Event(), asyncio.Event()
    snapshots = []
    parameters = {"plugin_options": {"extension": {"enabled": True}}}

    async def resolve(feature, action, payload):
        snapshots.append(payload)
        if len(snapshots) == 1:
            started.set()
            await release.wait()
        return True, {"interrupt": False}

    monkeypatch.setattr(policy_bridge, "resolve_policy", resolve)
    first = asyncio.create_task(
        MaisakaHeartFlowChatting._conversation_policy(state, "received", message=original, parameters=parameters)
    )
    await started.wait()
    original.extension_traits["direct_request"] = False
    parameters["plugin_options"]["extension"]["enabled"] = False
    state._input_version = 2
    state.message_cache.append(_message("next"))
    await MaisakaHeartFlowChatting._conversation_policy(state, "received", message=state.message_cache[1])
    release.set()
    await first
    assert [p["event_sequence"] for p in snapshots] == [1, 2]
    assert [p["input_version"] for p in snapshots] == [1, 2]
    assert [len(p["pending"]) for p in snapshots] == [1, 2]
    assert snapshots[0]["message"]["traits"]["direct_request"] is True
    assert snapshots[0]["parameters"]["plugin_options"]["extension"]["enabled"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handled,result,raises", [(False, None, False), (True, {}, False), (True, None, True), (True, [], True)]
)
async def test_policy_result_contract_does_not_confuse_absence_with_invalid_decisions(
    monkeypatch, handled, result, raises
):
    monkeypatch.setattr(policy_bridge, "resolve_policy", AsyncMock(return_value=(handled, result)))
    request = policy_bridge.resolve_mapping_policy("flow", "prepare_reply", {})
    if raises:
        with pytest.raises(ValueError, match="flow/prepare_reply"):
            await request
    else:
        assert await request == (result if handled else None)


def test_reply_window_preserves_order_target_and_its_quote():
    messages = [SimpleNamespace(message_id=f"m{i}") for i in range(50)]
    target = _message("m1")
    target.raw_message = MessageSequence([ReplyComponent("m2")])
    selected = context_views.window_reply_history(messages, target=target, limit=24)
    assert [m.message_id for m in selected] == ["m1", "m2", *[f"m{i}" for i in range(26, 50)]]
    assert context_views.window_reply_history(messages, target=None, limit=24) == messages[-24:]
    assert context_views.window_reply_history(messages, target=target, limit=100) is messages


def test_blank_targets_do_not_pin_old_internal_records():
    messages = [SimpleNamespace(message_id="") for _ in range(50)]
    assert context_views.window_reply_history(messages, target=_message(""), limit=24) == messages[-24:]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_group_profile_only_renders_provenance_checked_evidence(monkeypatch, enabled):
    from src.services import conversation_privacy

    monkeypatch.setattr(conversation_privacy, "group_session", lambda sid: SimpleNamespace(group_id="g"))
    monkeypatch.setattr(conversation_privacy, "_store", lambda: None)
    payload = {
        "person_id": "p",
        "profile_text": "PRIVATE GLOBAL SUMMARY",
        "summary": "PRIVATE GLOBAL SUMMARY",
        "evidence": [
            {"content": "visible public fact", "source": "web_import:source"},
            {"content": "PRIVATE FACT", "metadata": {"private_only": True}},
            {"content": "another public fact", "source": "web_import:source"},
        ],
    }
    filtered = conversation_privacy.filter_profile(payload, "chat")

    async def policy(feature, action, payload):
        assert "PRIVATE FACT" not in str(payload)
        return (True, {"indices": [1], "max_chars_per_record": 8}) if enabled else (False, None)

    monkeypatch.setattr(policy_bridge, "resolve_policy", policy)
    result = await context_views.select_memory(filtered, "chat", "profile")
    assert "PRIVATE" not in str(result)
    assert result["profile_text"] == ("another " if enabled else "visible public fact\nanother public fact")
    assert result["profile_text"] == result["summary"]
    assert payload["profile_text"] == "PRIVATE GLOBAL SUMMARY"
