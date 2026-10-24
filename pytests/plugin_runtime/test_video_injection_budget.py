from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import ast
import asyncio

import pytest


def load_injection_hook(*, fast_budget=False):
    path = Path(__file__).resolve().parents[2] / "plugins/YukiSakiko_content_understanding_plugin/plugin.py"
    method = next(n for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "inject_bilibili_summary")
    method.decorator_list = []
    budget = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "timeout")
    assert budget.args[0].value == 7.0  # Less than the registered 10-second Host deadline.
    if fast_budget:
        budget.args[0].value = 0.01
    namespace = {"asyncio": asyncio, "Any": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["inject_bilibili_summary"]


@pytest.mark.asyncio
async def test_slow_video_injection_retains_link_and_does_not_timeout_host_hook():
    message = {"processed_plain_text": "https://www.bilibili.com/video/BV1234567890"}
    original = dict(message)
    cancelled = asyncio.Event()

    async def slow(**kwargs):
        try:
            await asyncio.sleep(0.1)
            kwargs["message"]["processed_plain_text"] = "too late"
        finally:
            cancelled.set()

    plugin = SimpleNamespace(ctx=SimpleNamespace(logger=Mock()), _inject_video_summary=slow)
    result = await load_injection_hook(fast_budget=True)(plugin, message=message)
    assert result == {"action": "continue"}
    assert message == original
    assert cancelled.is_set()
    plugin.ctx.logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_video_injection_preserves_summary_and_propagates_shutdown_cancellation():
    expected = {"action": "continue", "modified_kwargs": {"message": {"processed_plain_text": "summary"}}}
    runner = AsyncMock(return_value=expected)
    plugin = SimpleNamespace(ctx=SimpleNamespace(logger=Mock()), _inject_video_summary=runner)
    assert await load_injection_hook()(plugin, message={}) == expected
    runner.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await load_injection_hook()(plugin, message={})
