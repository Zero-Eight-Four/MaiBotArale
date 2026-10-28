"""Real reply execution retains confirmed sends across partial failure and cancellation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio

import pytest

from pytests.maisaka.reply_fixtures import _message
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.common.data_models.reply_generation_data_models import ReplyGenerationResult
from src.maisaka.builtin_tool import reply
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext, PostProcessedReplyMessage
from src.plugin_runtime import context_views


@pytest.fixture
def execution(monkeypatch):
    events = []
    target = _message("request")
    runtime = SimpleNamespace(
        session_id="chat",
        _context_revision=7,
        _input_version=1,
        _chat_history=[],
        log_prefix="[test]",
        _update_stage_status=Mock(),
        chat_stream=SimpleNamespace(
            platform="qq", is_group_session=False, user_id="user", group_id="", account_id="bot"
        ),
    )

    async def policy(action, **kwargs):
        events.append((action, kwargs.get("target")))
        return {"receipt_hint": "confirmed"} if action == "sent" else {"decision": "allow"}

    runtime._conversation_policy = AsyncMock(side_effect=policy)
    runtime.record_planner_reply = AsyncMock()
    generated = ReplyGenerationResult()
    generated.completion.response_text = "first second third"
    generator = SimpleNamespace(generate_reply_with_context=AsyncMock(return_value=(True, generated)))
    monkeypatch.setattr(reply.replyer_manager, "get_replyer", lambda **kwargs: generator)
    monkeypatch.setattr(reply, "is_bot_self", lambda *args: False)
    monkeypatch.setattr(reply, "_resolve_reply_quote", AsyncMock(return_value=(False, None)))
    monkeypatch.setattr(reply, "_invoke_before_post_process_hook", AsyncMock(return_value=("first second third", {})))
    monkeypatch.setattr(context_views, "prepare_history", AsyncMock(return_value=SimpleNamespace(messages=[])))
    extension = SimpleNamespace(prepare=AsyncMock(), targets=[], reply_id="reply-id")
    monkeypatch.setattr(reply, "ReplyExtensionExecution", lambda **kwargs: extension)
    items = [PostProcessedReplyMessage(MessageSequence([TextComponent(text)])) for text in ["first", "second", "third"]]
    context = SimpleNamespace(
        runtime=runtime,
        engine=SimpleNamespace(_active_logical_turn_id="turn"),
        build_failure_result=BuiltinToolRuntimeContext.build_failure_result,
        build_success_result=BuiltinToolRuntimeContext.build_success_result,
        post_process_reply_message_items_async=AsyncMock(return_value=items),
    )
    return SimpleNamespace(runtime=runtime, events=events, target=target, context=context, generator=generator)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "first_failed", "second_failed", "cancel_second"])
async def test_delivery_acknowledged_once_at_first_confirmation(execution, monkeypatch, outcome):
    e = execution
    calls = []

    async def send(**kwargs):
        index = len(calls)
        calls.append(kwargs)
        if index == 1:
            assert sum(action == "sent" for action, _ in e.events) == 1
        if outcome == "first_failed" or (outcome == "second_failed" and index == 1):
            return None
        if outcome == "cancel_second" and index == 1:
            raise asyncio.CancelledError()
        return SimpleNamespace(message_id=f"sent-{index}")

    monkeypatch.setattr(reply.send_service, "_send_to_target_with_message", send)
    invocation = reply.execute_reply(
        e.context, {"msg_id": "request", "reply_reference": "brief context"}, call_id="call", target_message=e.target
    )
    if outcome == "cancel_second":
        with pytest.raises(asyncio.CancelledError):
            await invocation
    else:
        result = await invocation
        assert result.success is (outcome == "success")
        if result.success:
            assert "confirmed" in result.content
        if outcome == "second_failed":
            assert result.metadata["sent_message_ids"] == ["sent-0"]
    sent = [(a, target) for a, target in e.events if a == "sent"]
    assert len(sent) == (0 if outcome == "first_failed" else 1)
    assert all(target is e.target for _, target in sent)
    assert sum(a == "prepare_reply" for a, _ in e.events) == 2


@pytest.mark.asyncio
async def test_parallel_quote_does_not_replace_response_recipient(execution, monkeypatch):
    e = execution
    quoted = _message("quoted-other")
    supplement = SimpleNamespace(
        quote_message=quoted, plan=SimpleNamespace(model_dump=lambda: {}), metadata={}, compose=lambda items: items
    )
    monkeypatch.setattr(
        reply.send_service,
        "_send_to_target_with_message",
        AsyncMock(side_effect=[SimpleNamespace(message_id=f"s{i}") for i in range(3)]),
    )
    result = await reply.execute_reply(
        e.context,
        {"msg_id": "request", "reply_reference": "brief context"},
        call_id="call",
        target_message=e.target,
        supplement_runner=AsyncMock(return_value=supplement),
    )
    assert result.success
    assert e.generator.generate_reply_with_context.await_args.kwargs["reply_message"] is None
    e.runtime.record_planner_reply.assert_awaited_once_with(e.target, event_id="call")
    assert all(target is e.target for _, target in e.events)


@pytest.mark.asyncio
async def test_receipt_hook_failure_keeps_already_confirmed_message_id(execution, monkeypatch):
    e = execution
    original = e.runtime._conversation_policy.side_effect

    async def policy(action, **kwargs):
        if action == "sent":
            raise RuntimeError("fixture hook unavailable")
        return await original(action, **kwargs)

    e.runtime._conversation_policy.side_effect = policy
    monkeypatch.setattr(
        reply.send_service,
        "_send_to_target_with_message",
        AsyncMock(return_value=SimpleNamespace(message_id="already-delivered")),
    )
    result = await reply.execute_reply(
        e.context, {"msg_id": "request", "reply_reference": "brief"}, call_id="call", target_message=e.target
    )
    assert not result.success
    assert result.metadata["sent_message_ids"] == ["already-delivered"]
    assert result.metadata["send_results"][0]["success"] is True


@pytest.mark.asyncio
async def test_input_changed_during_generation_is_checked_again_before_send(execution, monkeypatch):
    e = execution
    generated = e.generator.generate_reply_with_context.return_value

    async def generate(**kwargs):
        e.runtime._context_revision += 1
        return generated

    async def policy(action, **kwargs):
        e.events.append((action, kwargs.get("target")))
        stale = kwargs.get("parameters", {}).get("_host_draft_revision") != e.runtime._context_revision
        return {"decision": "replan" if stale else "allow", "reason": "fixture new input"}

    e.generator.generate_reply_with_context.side_effect = generate
    e.runtime._conversation_policy.side_effect = policy
    sender = AsyncMock()
    monkeypatch.setattr(reply.send_service, "_send_to_target_with_message", sender)
    result = await reply.execute_reply(
        e.context, {"msg_id": "request", "reply_reference": "brief"}, call_id="call", target_message=e.target
    )
    assert not result.success and result.metadata["stale_reply"]
    sender.assert_not_awaited()
    assert [action for action, _ in e.events] == ["prepare_reply", "prepare_reply"]


@pytest.mark.asyncio
async def test_cli_acknowledges_after_local_render_without_platform_delivery(execution, monkeypatch):
    e = execution
    e.runtime.chat_stream.platform = reply.CLI_PLATFORM_NAME
    e.context.append_guided_reply_to_chat_history = Mock()
    rendered = []
    monkeypatch.setattr(reply, "render_cli_message", rendered.append)
    sender = AsyncMock()
    monkeypatch.setattr(reply.send_service, "_send_to_target_with_message", sender)
    result = await reply.execute_reply(
        e.context,
        {
            "msg_id": "request",
            "reply_reference": "brief",
            "_host_context": {"account_id": "spoof", "reply_target_message_id": "wrong"},
        },
        call_id="call",
        target_message=e.target,
    )
    assert result.success and rendered == ["first", "second", "third"]
    sender.assert_not_awaited()
    assert sum(action == "sent" for action, _ in e.events) == 1
    host = e.generator.generate_reply_with_context.await_args.kwargs["reply_tool_args"]["_host_context"]
    assert host["account_id"] == "bot" and host["reply_target_message_id"] == "request"
