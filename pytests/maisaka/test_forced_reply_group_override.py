from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.config.config import global_config
from src.maisaka import runtime as runtime_module


@pytest.mark.parametrize("detected", [(True, False, 1.0), (True, True, 1.0), (True, False, 0.5)])
def test_disabled_group_keeps_markers_without_arming_forced_reply(monkeypatch, detected):
    monkeypatch.setattr(global_config.chat.reply_timing, "mentioned_bot_reply", True)
    monkeypatch.setattr(global_config.chat.reply_timing, "inevitable_at_reply", True)
    monkeypatch.setattr(runtime_module, "is_mentioned_bot_in_message", lambda message: detected)
    arm = Mock()
    runtime = SimpleNamespace(
        chat_stream=SimpleNamespace(is_group_session=True, platform="qq", group_id="664646220"),
        _arm_forced_turn=arm,
    )
    message = SimpleNamespace(is_at=False, is_mentioned=False,extension_chat_policy={"force_disabled":True})

    runtime_module.MaisakaHeartFlowChatting._update_message_trigger_state(runtime, message)

    arm.assert_not_called()
    assert message.is_mentioned == detected[0]
    assert message.is_at == detected[1]


@pytest.mark.parametrize(
    "platform,group_id,is_group",
    [("qq", "959142150", True), ("other", "664646220", True), ("qq", "", False)],
)
@pytest.mark.parametrize("detected", [(True, False, 0.5), (True, True, 1.0)])
def test_other_groups_platforms_and_private_chats_keep_forced_reply(
    monkeypatch, platform, group_id, is_group, detected
):
    monkeypatch.setattr(global_config.chat.reply_timing, "mentioned_bot_reply", True)
    monkeypatch.setattr(global_config.chat.reply_timing, "inevitable_at_reply", True)
    monkeypatch.setattr(runtime_module, "is_mentioned_bot_in_message", lambda message: detected)
    arm = Mock()
    runtime = SimpleNamespace(
        chat_stream=SimpleNamespace(is_group_session=is_group, platform=platform, group_id=group_id),
        _arm_forced_turn=arm,
    )
    message = SimpleNamespace(is_at=False, is_mentioned=False)

    runtime_module.MaisakaHeartFlowChatting._update_message_trigger_state(runtime, message)

    arm.assert_called_once_with(message, is_at=detected[1], is_mentioned=detected[0])


def test_empty_override_preserves_existing_detector_boost(monkeypatch):
    monkeypatch.setattr(global_config.chat.reply_timing, "mentioned_bot_reply", False)
    monkeypatch.setattr(global_config.chat.reply_timing, "inevitable_at_reply", False)
    monkeypatch.setattr(runtime_module, "is_mentioned_bot_in_message", lambda message: (True, True, 1.0))
    arm = Mock()
    runtime = SimpleNamespace(
        chat_stream=SimpleNamespace(is_group_session=True, platform="qq", group_id="664646220"),
        _arm_forced_turn=arm,
    )
    message = SimpleNamespace(is_at=False, is_mentioned=False)

    runtime_module.MaisakaHeartFlowChatting._update_message_trigger_state(runtime, message)

    arm.assert_called_once()


@pytest.mark.parametrize('detected', [(True, False, 0.0), (True, True, 0.0)])
def test_group_override_arms_forced_turn_without_global_switches(monkeypatch, detected):
    monkeypatch.setattr(global_config.chat.reply_timing, 'mentioned_bot_reply', False)
    monkeypatch.setattr(global_config.chat.reply_timing, 'inevitable_at_reply', False)
    monkeypatch.setattr(runtime_module, 'is_mentioned_bot_in_message', lambda message: detected)
    arm = Mock()
    runtime = SimpleNamespace(_arm_forced_turn=arm)
    message = SimpleNamespace(is_at=False, is_mentioned=False, extension_chat_policy={
        'force_disabled': False, 'mentioned_bot_reply': True, 'inevitable_at_reply': True})
    runtime_module.MaisakaHeartFlowChatting._update_message_trigger_state(runtime, message)
    arm.assert_called_once()
