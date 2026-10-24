"""工具声明与当前开关一致，参考资料不会覆盖回复意图。"""

import pytest

from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.maisaka.builtin_tool.reply import get_tool_spec as reply_spec
from src.maisaka.builtin_tool.wait import get_tool_spec as wait_spec


def test_native_reply_reference_and_quote_contract():
    native = reply_spec()
    assert native.parameters_schema["required"] == ["msg_id", "reply_reference"]
    assert "set_quote" not in native.parameters_schema["properties"]


@pytest.mark.parametrize(
    "reason,reference",
    [("意图", "证据"), ("意图", ""), ("", "证据"), ("同一内容", "同一内容"), ("", "")],
)
def test_reference_keeps_intent_without_duplicate_text(reason, reference):
    text = BaseMaisakaReplyGenerator._build_reply_reference_message(reason, reference)
    if reason:
        assert reason in text
    if reference:
        assert reference in text
    if reason == reference:
        assert text.count(reason) == 1 if reason else text == ""
    if reason or reference:
        assert "host_internal_reply_context" in text


def test_wait_declares_schema():
    spec = wait_spec()
    assert "等待一段时间" in spec.description
    assert spec.parameters_schema["properties"]["seconds"]["type"] == "integer"
