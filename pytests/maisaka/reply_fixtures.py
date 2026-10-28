"""Shared native reply test fixtures, independent of obsolete Planner text suites."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.config.config import global_config
from src.core.tooling import ToolRegistry
from src.maisaka.builtin_tool.provider import MaisakaBuiltinToolProvider
from src.maisaka.context.messages import SessionBackedMessage
from src.maisaka.reasoning_engine import MaisakaReasoningEngine

def _message(message_id: str) -> SessionMessage:
    message = SessionMessage(message_id, datetime.now(), "test", received_at=datetime.now())
    message.session_id = "chat"
    message.message_info = MessageInfo(UserInfo("user", "User"))
    message.raw_message = MessageSequence([TextComponent(message_id)])
    message.processed_plain_text = message_id
    return message


def _history(message: SessionMessage) -> SessionBackedMessage:
    return SessionBackedMessage.from_session_message(
        message, raw_message=message.raw_message, visible_text=message.processed_plain_text
    )


@pytest.fixture
def engine(monkeypatch) -> MaisakaReasoningEngine:
    monkeypatch.setattr(global_config.experimental, "passive_replyer", True)
    runtime = SimpleNamespace(
        session_id="chat",
        log_prefix="[test]",
        _chat_history=[_history(_message("original"))],
        _input_version=0,
        _context_revision=0,
        _consecutive_wait_count=0,
        _conversation_policy=AsyncMock(return_value={}),
        record_planner_reply=AsyncMock(),
        chat_stream=SimpleNamespace(platform="test", is_group_session=True, group_id="group", user_id="", account_id="bot"),
        _end_planner_continuation=Mock(),
        _reset_consecutive_wait_count=Mock(),
        _enter_stop_state=Mock(),
        _update_stage_status=Mock(),
        expand_message_id_aliases=lambda args: args,
        is_action_tool_currently_available=lambda name: name != "reply",
        _tool_registry=ToolRegistry(),
        update_deferred_tool_specs=Mock(),
        set_current_action_tool_names=Mock(),
        sync_discovered_deferred_tools_with_context=Mock(),
        get_discovered_deferred_tool_specs=lambda: [],
        build_deferred_tools_reminder=lambda: "",
        _max_context_size=100,
        _cycle_counter=1,
        _max_internal_rounds=6,
        _chat_loop_service=SimpleNamespace(select_llm_context_messages=lambda *a, **kw: ([], 0)),
    )
    result = MaisakaReasoningEngine(runtime)
    runtime._reasoning_engine = result
    runtime._tool_registry.register_provider(MaisakaBuiltinToolProvider(result.build_builtin_tool_handlers()))
    result._emit_planner_progress = AsyncMock()
    result._record_tool_execution_effects = AsyncMock()
    result._append_tool_execution_result = Mock()
    return result

