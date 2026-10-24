from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pytests.plugin_runtime.test_arale_migration import plugin
from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType, get_item_text
from src.plugin_runtime.hook_payloads import deserialize_prompt_items, serialize_prompt_items


@pytest.mark.asyncio
@pytest.mark.parametrize('locale', ['zh-CN', 'en-US', 'ja-JP'])
async def test_video_rules_round_trip_preserves_host_fields_and_injects_once(tmp_path, locale):
    p, _ = plugin('Arale-web-policy', tmp_path)
    original = [ContextItemBuilder().set_role(RoleType.System).add_text_content('base').build(),
                ContextItemBuilder().set_role(RoleType.User).add_text_content('[B站视频] 测试摘要').build()]
    payload = {'items': serialize_prompt_items(original), 'item_schema_version': 1,
               'locale': locale, 'session_id': 'opaque', 'reply_tool_args': {'reply': '原回复'}}
    for _ in range(2):
        result = await p.video_reply_style(**payload)
        payload = result['modified_kwargs']
    assert payload['reply_tool_args'] == {'reply': '原回复'}
    assert payload['session_id'] == 'opaque'
    items = deserialize_prompt_items(payload['items'], item_schema_version=payload['item_schema_version'],
                                     original_items=original)
    rules = (Path('plugins/Arale-web-policy/prompts') / locale / 'video_rules.prompt').read_text()
    assert get_item_text(items[0]).count(rules) == 1
    assert get_item_text(items[1]) == '[B站视频] 测试摘要'
    assert any(c.get('metadata', {}).get('hook') == 'maisaka.replyer.before_model_request'
               for c in p.get_components())


@pytest.mark.asyncio
async def test_video_rule_leaves_unrelated_request_untouched(tmp_path):
    p, _ = plugin('Arale-web-policy', tmp_path)
    payload = {'items': [{'role': 'user', 'parts': [{'type': 'text', 'text': '今天天气怎么样'}]}]}
    assert await p.video_reply_style(**payload) == {'action': 'continue'}


@pytest.mark.asyncio
async def test_reaction_tool_visibility_duplicate_and_cooldown(tmp_path):
    p, _ = plugin('github_cateye_set-msg-emoji-like', tmp_path)
    tools = {c['name']: c['metadata'] for c in p.get_components()
             if c['name'] in {'emoji_like', 'emoji_like_list'}}
    assert set(tools) == {'emoji_like', 'emoji_like_list'}
    assert all(t['visibility'] == 'visible' for t in tools.values())
    assert p.config.emoji_reaction.tool_cooldown_seconds == 15
    p._apply_emoji_like = AsyncMock(return_value=(True, ''))
    p._record_and_context = AsyncMock()
    p._resolve_emoji_id = lambda _: (76, False)
    first = await p.tool_emoji_like(stream_id='s', target_message_id='m', emoji='76')
    duplicate = await p.tool_emoji_like(stream_id='s', target_message_id='m', emoji='76')
    next_message = await p.tool_emoji_like(stream_id='s', target_message_id='m2', emoji='76')
    assert first['success'] and duplicate['success'] and duplicate['already_reacted']
    assert not next_message['success'] and '频繁' in next_message['error']
    p._apply_emoji_like.assert_awaited_once_with('s', 'm', 76)
    p._record_and_context.assert_awaited_once()
