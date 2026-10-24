"""身份传递不依赖昵称，保留引用和自身消息语义。"""

from datetime import datetime
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from src.chat.replyer import maisaka_generator_base as generator_module
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.common.data_models.message_component_data_model import MessageSequence, ReplyComponent, TextComponent
from src.maisaka.context.planner_messages import build_planner_prefix, build_planner_user_prefix_from_session_message


def _message(user_id="2730843725", nickname="匿名用户", platform="qq"):
    return SimpleNamespace(
        platform=platform, timestamp=datetime(2026, 10, 9, 17, 47, 12),
        message_id="1531667542", session_id="session-test", is_notify=False,
        message_info=SimpleNamespace(user_info=SimpleNamespace(
            user_id=user_id, user_nickname=nickname, user_cardname="依然范特西")),
        raw_message=MessageSequence([
            ReplyComponent(target_message_id="12345678"), TextComponent("请按要求调整")]),
    )


@pytest.mark.parametrize("user_id,nickname,platform", [
    ("2730843725", "匿名用户", "qq"),
    ("other-user", "2730843725", "qq"),
    ("2730843725", "匿名用户", "other-platform"),
])
def test_planner_preserves_actual_identity_and_existing_attributes(user_id, nickname, platform):
    message = _message(user_id, nickname, platform)
    attrs = ElementTree.fromstring(build_planner_user_prefix_from_session_message(
        message, include_chat_id=True) + "</message>").attrib
    assert attrs["platform"] == platform and attrs["user_id"] == user_id
    assert attrs["user"] == nickname and attrs["group_card"] == "依然范特西"
    assert attrs["msg_id"] == message.message_id and attrs["session_id"] == message.session_id
    assert attrs["quote"] == "12345678" and attrs["time"] == "17:47:12"
    assert "is_self_message" not in attrs


def test_planner_escapes_identity_and_keeps_optional_compatibility():
    prefix = build_planner_prefix(timestamp=datetime.now(), user_name="user",
                                  platform='p&"<', user_id='id&"<', is_self_message=True)
    attrs = ElementTree.fromstring(prefix + "</message>").attrib
    assert attrs["platform"] == 'p&"<' and attrs["user_id"] == 'id&"<'
    assert attrs["is_self_message"] == "true"
    attrs = ElementTree.fromstring(build_planner_prefix(
        timestamp=datetime.now(), user_name="user") + "</message>").attrib
    assert "platform" not in attrs and "user_id" not in attrs


def test_reply_target_uses_actual_sender_and_keeps_quote(monkeypatch):
    monkeypatch.setattr(generator_module, "is_bot_self", lambda platform, user_id: False)
    generator = object.__new__(BaseMaisakaReplyGenerator)
    generator._build_target_message_content = lambda message: "请按要求调整"
    block = generator._build_target_message_block(_message("other-user", "2730843725"))
    assert "platform=qq，user_id=other-user" in block
    assert "1531667542" in block and "12345678" in block and "请按要求调整" in block
    assert generator._build_target_message_block(None) == ""


def test_reply_target_keeps_bot_self_semantics(monkeypatch):
    monkeypatch.setattr(generator_module, "is_bot_self", lambda platform, user_id: True)
    generator = object.__new__(BaseMaisakaReplyGenerator)
    generator._build_target_message_content = lambda message: "自己的发言"
    block = generator._build_target_message_block(_message())
    assert "补充说明你自己" in block and "不要把你自己的发言当成别人的发言" in block
    assert "发送者身份" not in block
