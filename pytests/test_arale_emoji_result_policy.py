import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import importlib.util
from pathlib import Path
import pytest
from src.maisaka.builtin_tool import send_emoji
import src.plugin_runtime.policy_bridge as bridge

spec=importlib.util.spec_from_file_location('emoji_selection_bridge',Path(__file__).resolve().parents[1] / 'plugins/Arale-emoji-policy/selection.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

@pytest.mark.asyncio
@pytest.mark.parametrize('text,index', [('```json\n{"emoji_index":2,"reason":"合适"}\n```',2), ('not JSON',None), ('{"emoji_index":99}',None)])
async def test_policy_selection_is_used_without_first_candidate_fallback(monkeypatch,text,index):
    emojis=[SimpleNamespace(name='one'),SimpleNamespace(name='two')]
    monkeypatch.setattr(send_emoji,'emoji_manager',SimpleNamespace(emojis=emojis))
    monkeypatch.setattr(send_emoji,'sample',lambda seq,n:seq[:n])
    monkeypatch.setattr(send_emoji,'_get_emoji_candidate_count',lambda:2)
    monkeypatch.setattr(send_emoji,'_build_emoji_candidate_message',AsyncMock(return_value=SimpleNamespace(to_context_item=lambda:None)))
    monkeypatch.setattr(send_emoji,'_render_emoji_selection_system_prompt',AsyncMock(return_value='select'))
    monkeypatch.setattr(send_emoji,'_resolve_emoji_selector_model_task_name',lambda:'replyer')
    async def resolve(feature,action,payload):
        assert (feature,action)==('emoji','selection_result')
        return True,policy.normalize_selection(payload['content'],payload['candidate_count'])
    monkeypatch.setattr(bridge,'resolve_policy',resolve)
    response=SimpleNamespace(content=text,prompt_tokens=1,completion_tokens=1,total_tokens=2,prompt_html_uri='/test',model_name='test')
    ctx=SimpleNamespace(runtime=SimpleNamespace(run_sub_agent=AsyncMock(return_value=response),log_prefix='[test]'))
    selected,_=await send_emoji._select_emoji_with_sub_agent(ctx,'',[],2,{})
    assert selected is (emojis[index-1] if index is not None else None)
