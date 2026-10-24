"""Verify native tool behavior with plugin-owned mode and attachment guidance."""
from pathlib import Path
from types import ModuleType
import importlib
import sys

import pytest

from src.config.config import global_config
from src.maisaka.builtin_tool import get_all_builtin_tool_specs
from src.maisaka.builtin_tool.reply import get_tool_spec
from src.maisaka.tool_result_condenser import should_condense_tool_result
from src.plugin_runtime.hook_payloads import serialize_tool_definitions

ROOT = Path(__file__).resolve().parents[2]


def module(section):
    name = "native_cleanup_" + section
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / "plugins" / "Arale-policy" / section)]
        sys.modules[name] = package
    return importlib.import_module(name + ".plugin")


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["zh-CN", "en-US", "ja-JP"])
@pytest.mark.parametrize("new_mode", [False, True])
async def test_mode_guidance_follows_actual_native_tools(monkeypatch, locale, new_mode):
    monkeypatch.setattr(global_config.emoji, "use_new_send_logic", new_mode)
    plugin = module("emoji").create_plugin()
    plugin.set_plugin_config(plugin.get_default_config())
    definitions = serialize_tool_definitions([spec.to_llm_definition() for spec in get_all_builtin_tool_specs()])
    payload = {"items": [{"item_type": "SystemMessageItem", "parts": []}],
               "locale": locale, "tool_definitions": definitions}
    result = await plugin.hints(**payload)
    text = str(result["modified_kwargs"]["items"])
    if new_mode:
        assert "show_emoji_list" in text and "reply.attach_emoji" in text
    else:
        assert "show_emoji_list" not in text and "reply.attach_emoji" not in text
    repeated = await plugin.hints(**result["modified_kwargs"])
    assert repeated["modified_kwargs"]["items"] == result["modified_kwargs"]["items"]


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["zh-CN", "en-US", "ja-JP"])
async def test_attachment_guidance_only_when_at_is_attached(locale):
    p = module("context").create_plugin()
    p.set_plugin_config(p.get_default_config())
    assert await p.attachment_hint(reply_tool_args={}, locale=locale) == {"action": "continue"}
    result = await p.attachment_hint(reply_tool_args={"attach_at": ["message-1"]},
                                     extra_prompt="prior", locale=locale)
    kwargs = result["modified_kwargs"]
    assert kwargs["extra_prompt"].startswith("prior")
    assert kwargs["reply_tool_args"]["attach_at"] == ["message-1"]
    repeated = await p.attachment_hint(**kwargs)
    assert repeated["modified_kwargs"]["extra_prompt"] == kwargs["extra_prompt"]


@pytest.mark.parametrize("tool", ["search_chat_history", "read_chat_history", "read_file",
                                 "read_archive_file", "read_video", "inspect_file", "list_chat_files", "fetch_url"])
def test_native_long_result_path_keeps_budget_and_preview_rules(monkeypatch, tool):
    monkeypatch.setattr(global_config.capabilities, "tool_result_condense_threshold", 800)
    assert should_condense_tool_result(tool, "x" * 800, success=True)
    assert not should_condense_tool_result(tool, "x" * 799, success=True)
    assert not should_condense_tool_result(tool, "x" * 800, success=True, already_previewed=True)
    assert not should_condense_tool_result(tool, "x" * 800, success=False)
