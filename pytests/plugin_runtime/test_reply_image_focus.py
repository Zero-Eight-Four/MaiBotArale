from copy import deepcopy

import pytest

from pytests.plugin_runtime.test_arale_migration import load, plugin
from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType, get_item_text
from src.plugin_runtime.hook_payloads import deserialize_prompt_items, serialize_prompt_items

mod, _ = load('Arale-vision')
from importlib import import_module
focus = import_module(mod.__name__ + '.reply_images').focus_reply_images


def message(message_id, text='', images=1):
    return {'item_type':'UserMessageItem', 'parts':[
        {'type':'text', 'text':f'<message msg_id="{message_id}">\n{text}'},
        *[{'type':'image', 'image_reference':f'{message_id}-{i}'} for i in range(images)]]}


def count(items):
    return sum(p.get('type') == 'image' for item in items for p in item.get('parts', []))


def test_current_images_preserved_old_data_removed_without_losing_captions():
    items = [message('100', '旧图描述'), message('200'), message('300', images=2)]
    before = deepcopy(items)
    selected, removed = focus(items, '300', '')
    assert count(selected) == 2 and removed == 2
    assert selected[-1] == before[-1] and items == before
    assert '旧图描述' in selected[0]['parts'][0]['text']
    assert len(selected) == len(items)


def test_explicit_old_image_reference_and_comparison():
    items = [message('100'), message('200'), message('300')]
    selected, _ = focus(items, '300', '核对消息100中的图片')
    assert count(selected[0:1]) == 1 and count(selected[1:2]) == 0 and count(selected[2:]) == 1
    selected, _ = focus(items, '300', '比较两张图的构图')
    assert count(selected[0:1]) == 0 and count(selected[1:]) == 2
    selected, _ = focus(items, '300', '对比这三张图')
    assert count(selected) == 3
    items[-1] = message('300', images=3)
    selected, _ = focus(items, '300', '比较这几张图')
    assert count(selected[:-1]) == 0 and count(selected[-1:]) == 3


def test_text_reply_uses_recent_image_only_when_the_task_calls_for_it():
    items = [message('100'), message('200'), message('300', '这张的配色怎么样', images=0)]
    selected, _ = focus(items, '300', '')
    assert count(selected[0:1]) == 0 and count(selected[1:2]) == 1
    items[-1] = message('300', '今天几点开会', images=0)
    selected, _ = focus(items, '300', '')
    assert count(selected) == 0


def test_unknown_target_markup_is_conservative_and_ids_match_exactly():
    items = [message('1100'), message('200')]
    assert focus(items, 'unknown', '') == (items, 0)
    selected, _ = focus(items, '200', '参考100')
    assert count(selected[0:1]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('locale', ['zh-CN', 'en-US', 'ja-JP'])
async def test_native_hook_item_roundtrip_and_host_fields(tmp_path, locale):
    p, _ = plugin('Arale-vision', tmp_path)
    p.config.locale = locale
    image = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII='
    original = [ContextItemBuilder().set_role(RoleType.System).add_text_content('base').build()]
    for message_id in ['100', '200']:
        original.append(ContextItemBuilder().set_role(RoleType.User)
                        .add_text_content(f'<message msg_id="{message_id}">已有描述')
                        .add_image_content('png', image).build())
    payload = {'items':serialize_prompt_items(original), 'item_schema_version':1,
               'reply_message_id':'200', 'session_id':'s', 'attempt':1, 'reply_tool_args':{'reply_reference':''}}
    result = await p.focus_reply_images(**payload)
    fields = result['modified_kwargs']
    assert fields['reply_tool_args'] == payload['reply_tool_args'] and fields['attempt'] == 1
    assert fields['session_id'] == 's' and fields['item_schema_version'] == 1
    restored = deserialize_prompt_items(fields['items'], item_schema_version=fields['item_schema_version'], original_items=original)
    assert len(restored) == len(original) and '已有描述' in get_item_text(restored[1])
    assert count(fields['items']) == 1 and count(payload['items']) == 2


@pytest.mark.asyncio
async def test_disabled_image_focus_leaves_request_unchanged(tmp_path):
    p, _ = plugin('Arale-vision', tmp_path)
    p.config.reply_image_focus = False
    assert await p.focus_reply_images(items=[message('100')], reply_message_id='100') == {'action':'continue'}
