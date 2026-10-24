import pytest

from pytests.plugin_runtime.test_arale_migration import plugin

ERROR_TEXT = (
    "The prompt could not be submitted. The prompt contains sensitive words that violate Google's "
    "[Generative AI Prohibited Use policy](https://policies.google.com/terms/generative-ai/use-policy). "
    "Try rephrasing the prompt."
)


@pytest.mark.asyncio
async def test_upstream_safety_body_is_not_forwarded_as_normal_reply(tmp_path):
    p, _ = plugin('Arale-reply-guard', tmp_path)
    result = await p.intercept_veto(response=ERROR_TEXT, session_id='s',
                                  output_items=[{'item_type':'AssistantMessageItem'}], item_schema_version=1)
    assert result['modified_kwargs']['response'] == ''
    assert result['modified_kwargs'].get('retry') is False
    assert 'output_items' not in result['modified_kwargs']


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    '这张插画的配色挺好看。',
    'The colors look nice in this illustration.',
    '这条错误的意思是：' + ERROR_TEXT,
    '```text\n' + ERROR_TEXT + '\n```',
    '“The prompt could not be submitted.” 是服务端提示，不是图片内容。',
])
async def test_normal_reply_or_error_discussion_is_unchanged(tmp_path, text):
    p, _ = plugin('Arale-reply-guard', tmp_path)
    assert await p.intercept_veto(response=text, session_id='s') == {'action':'continue'}
