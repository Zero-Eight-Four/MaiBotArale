"""One policy call per reply, while direct generation and session boundaries stay safe."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from pytests.maisaka.reply_fixtures import _history, _message, engine
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import reply as reply_tool
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.context.messages import ToolResultMessage
from src.plugin_runtime import policy_bridge
from src.plugin_runtime.context_views import prepare_history, select_history


def build_history_message(text):
    message = _message(text)
    message.session_id = "session-1"
    return _history(message)


def build_session_backed_text_message(*, speaker_name, text, timestamp, source_kind, message_id):
    message = build_history_message(text)
    message.message_id = message_id
    message.source_kind = source_kind
    return message


@pytest.fixture
def reply_context(engine, monkeypatch):
    runtime = engine._runtime
    runtime.session_id = "session-1"
    runtime._input_version = 0
    runtime._context_revision = 0
    runtime.chat_stream.account_id = "bot"
    runtime._conversation_policy = AsyncMock(return_value={})
    runtime.find_source_message_by_id = lambda mid: _message(mid)
    context = BuiltinToolRuntimeContext(engine, runtime)
    # Stop generation before any model/send, while running the real projection path.
    extension = SimpleNamespace(prepare=AsyncMock(), targets=[], reply_id="test")
    monkeypatch.setattr(reply_tool, "ReplyExtensionExecution", lambda **kwargs: extension)
    monkeypatch.setattr(reply_tool, "is_bot_self", lambda *args: False)
    return context


@pytest.fixture
def history_policy(monkeypatch):
    async def _resolve(feature, action, payload):
        if (feature, action) == ("context", "select_history"):
            items = payload.get("messages") or []
            recalled_ids = {
                str(item.get("content") or "").replace("已撤回消息 ", "").rstrip("。")
                for item in items
                if item.get("tool_name") == "recall_message" and item.get("success")
            }
            indices = [
                idx
                for idx, item in enumerate(items)
                if str(item.get("message_id") or "") not in recalled_ids
            ]
            return True, {"indices": indices}
        return False, None

    resolve = AsyncMock(side_effect=_resolve)
    monkeypatch.setattr(policy_bridge, "resolve_policy", resolve)
    return resolve


def recalled_history():
    bad = build_session_backed_text_message(
        speaker_name="bot",
        text="withdrawn reply",
        timestamp=datetime.now(),
        source_kind="guided_reply",
        message_id="sent-bad",
    )
    user = build_history_message("current request")
    receipt = ToolResultMessage(
        content="已撤回消息 sent-bad。",
        timestamp=datetime.now(),
        tool_call_id="recall-1",
        logical_turn_id="turn-1",
        tool_name="recall_message",
        success=True,
    )
    return [bad, user, receipt]


def stopped_generator(observed):
    generator = object.__new__(BaseMaisakaReplyGenerator)
    generator._resolve_session_id = lambda stream_id: stream_id
    generator._persist_reply_preview = Mock()
    generator.express_model = SimpleNamespace()

    async def capture(**kwargs):
        observed.extend(kwargs["chat_history"])
        raise RuntimeError("test stops before any model or send call")

    generator._build_reply_context = capture
    return generator


@pytest.mark.asyncio
async def test_reply_tool_and_real_generator_share_one_selection(reply_context, monkeypatch, history_policy):
    history = recalled_history()
    reply_context.runtime._chat_history = history
    observed = []
    generator = stopped_generator(observed)
    monkeypatch.setattr(reply_tool.replyer_manager, "get_replyer", lambda **kwargs: generator)
    result = await reply_tool.handle_tool(
        reply_context, ToolInvocation("reply", arguments={"msg_id": "msg-1", "reply_reference": "回应当前请求"})
    )
    assert not result.success
    assert history[0] not in observed and history[1] in observed
    calls = [c for c in history_policy.call_args_list if c.args[:2] == ("context", "select_history")]
    assert len(calls) == 1
    assert len(history) == 3  # Filtering does not delete audit records.


@pytest.mark.asyncio
async def test_direct_generator_still_filters_recalled_history(history_policy):
    history = recalled_history()
    observed = []
    success, _ = await stopped_generator(observed).generate_reply_with_context(
        chat_history=history, stream_id="session-1"
    )
    assert not success
    assert history[0] not in observed and history[1] in observed
    calls = [c for c in history_policy.call_args_list if c.args[:2] == ("context", "select_history")]
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["session", "append", "replace", "plain_flag"])
async def test_projection_cannot_be_reused_for_other_records_or_sessions(history_policy, change):
    history = recalled_history()
    projection = await prepare_history(history, "session-1")
    selected = list(projection.messages)
    session = "session-1"
    if change == "session":
        session = "session-2"
    elif change == "append":
        selected.append(history[0])
    elif change == "replace":
        selected[0] = build_history_message("changed request")
    else:
        projection = True
    with pytest.raises(ValueError, match="does not match"):
        await select_history(selected, session, projection=projection)
