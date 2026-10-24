from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pytests.plugin_runtime.test_arale_migration import plugin
from src.maisaka.builtin_tool.send_emoji import get_tool_spec
from src.plugin_runtime.tool_specs import extend_tool_spec


@pytest.mark.asyncio
async def test_emoji_hint_extends_actual_builtin_schema(tmp_path, monkeypatch):
    p, _ = plugin('Arale-emoji-policy', tmp_path)
    async def resolve(feature, action, payload):
        result = await p.tool_schema(feature, action, payload)
        return True, result['modified_kwargs']['result']
    monkeypatch.setattr('src.plugin_runtime.policy_bridge.resolve_policy', resolve)
    spec = await extend_tool_spec(get_tool_spec())
    assert spec.parameters_schema['properties']['emotion']['type'] == 'string'
    assert 'emotion' not in spec.parameters_schema.get('required', [])


@pytest.mark.asyncio
async def test_emoji_hint_reaches_only_same_session_selection_request_and_is_cleared(tmp_path):
    p, _ = plugin('Arale-emoji-policy', tmp_path)
    await p.remember_selection_hint(stream_id='s', requested_emotion='猫猫反击')
    items = [{'item_type':'SystemMessageItem', 'parts':[{'type':'text', 'text':'选择候选；返回emoji_index'}]}]
    result = await p.selection_hint(items=items, session_id='s', item_schema_version=1)
    assert result['modified_kwargs']['item_schema_version'] == 1
    assert '猫猫反击' in str(result['modified_kwargs']['items'])
    assert await p.selection_hint(items=items, session_id='other') == {'action':'continue'}
    assert await p.selection_hint(items=[{'item_type':'SystemMessageItem', 'parts':[{'type':'text', 'text':'聊天'}]}], session_id='s') == {'action':'continue'}
    await p.clear_selection_hint(stream_id='s')
    assert await p.selection_hint(items=items, session_id='s') == {'action':'continue'}
    await p.remember_selection_hint(stream_id='s', requested_emotion='旧偏好')
    await p.remember_selection_hint(stream_id='s', requested_emotion='')
    assert not p._selection_hints


@pytest.mark.asyncio
async def test_compatibility_disables_only_duplicate_qq_lookup(tmp_path):
    p, _ = plugin('Arale-snowluma-compat', tmp_path)
    p.ctx.component = SimpleNamespace(disable_component=AsyncMock(return_value={'success':True}))
    await p._disable_duplicate_lookup()
    p.ctx.component.disable_component.assert_awaited_once_with(
        'maibot-team.snowluma-adapter.find_user_qq_id', 'TOOL')
