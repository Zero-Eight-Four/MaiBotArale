from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import importlib.util
import sys

import pytest


def load_media_entrypoints():
    path = Path(__file__).resolve().parents[2] / "plugins/Arale-web/media_entrypoints.py"
    spec = importlib.util.spec_from_file_location("arale_web_media_entrypoints_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MediaEntrypoints


@pytest.mark.asyncio
async def test_media_parse_is_explicit_tool_and_defers_parallel_reply():
    entrypoints_cls = load_media_entrypoints()
    plugin = object.__new__(entrypoints_cls)
    plugin.config = SimpleNamespace(
        plugin=SimpleNamespace(enabled=True),
        media=SimpleNamespace(plugin=SimpleNamespace(enabled=True), parse=SimpleNamespace(enable_douyin=True)),
    )
    plugin._media = SimpleNamespace(
        tool_parse_bilibili_video=AsyncMock(return_value={"success": True, "content": "summary"}),
    )
    plugin.ctx = SimpleNamespace(logger=Mock())

    result = await plugin.tool_parse_bilibili_video(link="https://www.bilibili.com/video/BV1234567890")
    assert result["success"] is True

    output_items = [
        {"item_type": "FunctionCallItem", "tool_call": {"func_name": "parse_bilibili_video"}},
        {"item_type": "FunctionCallItem", "tool_call": {"func_name": "reply"}},
    ]
    deferred = await plugin.defer_media_reply(output_items=output_items)
    assert len(deferred["modified_kwargs"]["output_items"]) == 1
    assert deferred["modified_kwargs"]["output_items"][0]["tool_call"]["func_name"] == "parse_bilibili_video"
