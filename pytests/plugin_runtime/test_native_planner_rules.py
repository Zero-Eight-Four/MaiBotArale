"""Static rules use the official request hook, with real SDK Item round trips."""
from pathlib import Path

import pytest

from pytests.plugin_runtime.test_arale_migration import plugin, load
from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType, get_item_text
from src.plugin_runtime.hook_payloads import deserialize_prompt_items, serialize_prompt_items

SUPPLIERS = [
    ('Arale-emoji-policy', 'hints', ['emoji_rules']),
    ('Arale-web-policy', 'planner_part', ['tool_rules', 'reminder_rules', 'network_rules', 'video_rules']),
    ('Arale-conversation-flow', 'planner_part', ['flow_reply_rules', 'flow_wait_rule']),
    ('Arale-asr', 'planner_part', ['asr_rules']),
    ('Arale-group-context', 'planner_part', ['group_context_rules']),
    ('Arale-vision', 'inject_planner_rules', ['vision']),
    ('Arale-reverse-image', 'inject_planner_rules', ['vision_rules']),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('locale', ['zh-CN', 'en-US', 'ja-JP'])
@pytest.mark.parametrize('locale_source', ['payload', 'environment'])
async def test_native_rule_chain_preserves_contract_locale_and_is_idempotent(tmp_path, locale, locale_source, monkeypatch):
    original = [ContextItemBuilder().set_role(RoleType.System).add_text_content('base').build(),
                ContextItemBuilder().set_role(RoleType.User).add_text_content('user message').build()]
    payload = {'items': serialize_prompt_items(original), 'item_schema_version': 1,
               'locale': locale, 'session_id': 'opaque', 'selection_reason': 'latest',
               'selected_history_count': 2, 'tool_definitions': [{'name': 'reply'}]}
    if locale_source == 'environment':
        payload.pop('locale')
        monkeypatch.setenv('MAIBOT_LOCALE', locale)
    instances = []
    expected = []
    for ident, method, names in SUPPLIERS:
        p, _ = plugin(ident, tmp_path / ident)
        components = p.get_components()
        assert any(c.get('metadata', {}).get('hook') == 'maisaka.planner.before_request'
                   and c['metadata']['handler_name'] == method for c in components)
        assert not any(route.startswith('prompt_part.') for c in components
                       for route in c.get('metadata', {}).get('policy_routes', []))
        instances.append((p, method))
        expected.extend((load(ident)[1] / 'prompts' / locale / (name + '.prompt')).read_text()
                        for name in names)
        if ident == 'Arale-emoji-policy':
            p.config.intent = 'configured custom sticker intent'
    for _ in range(2):
        for p, method in instances:
            result = await getattr(p, method)(**payload)
            payload = result.get('modified_kwargs', payload)
    if locale_source == 'payload':
        assert payload['locale'] == locale
    else:
        assert 'locale' not in payload
    assert payload['session_id'] == 'opaque'
    assert payload['selected_history_count'] == 2
    assert payload['tool_definitions'] == [{'name': 'reply'}]
    items = deserialize_prompt_items(payload['items'], item_schema_version=1, original_items=original)
    assert len(items) == 2 and get_item_text(items[1]) == 'user message'
    text = get_item_text(items[0])
    assert text.count('configured custom sticker intent') == 1
    for rules in expected:
        assert text.count(rules) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('ident,method,names', SUPPLIERS)
async def test_disabled_supplier_does_not_inject(ident, method, names, tmp_path):
    p, _ = plugin(ident, tmp_path / ident)
    p.config.plugin.enabled = False
    payload = {'items': [{'item_type': 'SystemMessageItem', 'parts': [{'type': 'text', 'text': 'base'}]}],
               'locale': 'zh-CN', 'session_id': 'opaque'}
    result = await getattr(p, method)(**payload)
    assert result == {'action': 'continue'}
    assert payload['items'][0]['parts'] == [{'type': 'text', 'text': 'base'}]
