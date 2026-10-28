"""Exercise the actual terminal-empty recovery branches against the current result class."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator, MaisakaReplyContext
from src.llm_models.exceptions import ModelResponseTerminatedError
from src.maisaka.chat_loop_service import MaisakaChatLoopService
from src.plugin_runtime import policy_bridge


class Hooks:
    def __init__(self):
        self.calls=[]

    async def invoke_hook(self,name,**kwargs):
        self.calls.append((name,kwargs))
        if name=='maisaka.replyer.after_response':
            assert kwargs['reply_tool_args']['arale_generation_failure']=='provider_safety'
            kwargs={**kwargs,'response':'recovered status','retry':False}
        return SimpleNamespace(kwargs=kwargs)


@pytest.mark.asyncio
async def test_planner_terminal_empty_reaches_recovery_hook(monkeypatch):
    hooks=Hooks()
    service=MaisakaChatLoopService(chat_system_prompt='test')
    model=SimpleNamespace(generate_response_with_context=AsyncMock(side_effect=ModelResponseTerminatedError('empty_choices')))
    monkeypatch.setattr(service,'_get_llm_chat_client',lambda kind:model)
    monkeypatch.setattr(MaisakaChatLoopService,'_get_runtime_manager',staticmethod(lambda:hooks))
    response=await service.chat_loop_step([],tool_definitions=[],logical_turn_id='turn')
    after=next(k for n,k in hooks.calls if n=='maisaka.planner.after_response')
    assert after['arale_generation_failure']=='provider_safety'
    assert after['arale_safety_meta']['logical_turn_id']=='turn'
    assert response.output_items == ()
    model.generate_response_with_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_replyer_terminal_empty_can_recover_through_existing_hook(monkeypatch):
    hooks=Hooks()
    model=SimpleNamespace(task_name='replyer',generate_response_with_context=AsyncMock(
        side_effect=ModelResponseTerminatedError('empty_choices')))
    generator=BaseMaisakaReplyGenerator(llm_client_cls=lambda **kwargs:model,load_prompt_func=lambda *a,**k:'test')
    generator._resolve_session_id=lambda sid:sid
    generator._build_reply_context=AsyncMock(return_value=MaisakaReplyContext())
    generator._build_request_messages=Mock(return_value=[])
    generator._uses_retro_prompt=lambda:False
    generator._persist_reply_preview=Mock()
    monkeypatch.setattr(BaseMaisakaReplyGenerator,'_get_runtime_manager',staticmethod(lambda:hooks))
    monkeypatch.setattr(policy_bridge,'resolve_policy',AsyncMock(return_value=(False,None)))
    success,result=await generator.generate_reply_with_context(chat_history=[],stream_id='chat')
    assert success and result.completion.response_text=='recovered status'
    model.generate_response_with_context.assert_awaited_once()
