"""Empty candidates only replay when the plugin explicitly permits the same request."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.exceptions import ModelResponseTerminatedError
from src.llm_models.model_client.base_client import ResponseRequest
from src.llm_models.utils_model import LLMOrchestrator
from src.plugin_runtime import policy_bridge


@pytest.mark.asyncio
@pytest.mark.parametrize('succeed', [True, False])
async def test_empty_choices_retries_once_on_same_request(monkeypatch, succeed):
    calls = []

    async def resolve(feature, action, payload):
        calls.append((action, payload))
        return (True, {'retry': payload['attempt'] == 1, 'retry_after': 0}) if action == 'empty_choices' else (False, {})

    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolve)
    provider = APIProvider(name='test', api_key='test', base_url='https://example.invalid', max_retry=5)
    request = ResponseRequest(model_info=ModelInfo(model_type='llm', name='test-model',
                              model_identifier='test-model', api_provider='test'), context_items=[])
    response = SimpleNamespace()
    client = SimpleNamespace(get_response=AsyncMock(side_effect=[ModelResponseTerminatedError('empty_choices'),
                             response if succeed else ModelResponseTerminatedError('empty_choices')]))
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.request_type, orchestrator.task_name = 'maisaka.planner', 'planner'
    orchestrator._record_success_generation_attempt = Mock()
    orchestrator._schedule_llm_retry_event = Mock()
    monkeypatch.setattr('src.llm_models.utils_model.mark_request_succeeded', lambda *a: None)
    monkeypatch.setattr('src.llm_models.utils_model.update_failed_request_attempt', lambda *a, **k: None)
    monkeypatch.setattr('src.llm_models.utils_model.save_failed_request_snapshot', lambda **k: None)
    if succeed:
        assert await orchestrator._attempt_request_on_model(provider, client, request) is response
    else:
        with pytest.raises(ModelResponseTerminatedError):
            await orchestrator._attempt_request_on_model(provider, client, request)
    assert client.get_response.await_count == 2
    assert all(call.args[0] is request for call in client.get_response.await_args_list)
    assert [p['attempt'] for a, p in calls if a == 'empty_choices'] == ([1] if succeed else [1, 2])


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['empty_choices', 'safety', 'unsupported_location'])
async def test_denied_or_explicit_termination_never_replays(monkeypatch, reason):
    resolver = AsyncMock(return_value=(False, {}))
    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolver)
    monkeypatch.setattr('src.llm_models.utils_model.save_failed_request_snapshot', lambda **k: None)
    provider = APIProvider(name='test', api_key='test', base_url='https://example.invalid', max_retry=5)
    request = ResponseRequest(model_info=ModelInfo(model_type='llm', name='test-model',
                              model_identifier='test-model', api_provider='test'), context_items=[])
    client = SimpleNamespace(get_response=AsyncMock(side_effect=ModelResponseTerminatedError(reason)))
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.request_type, orchestrator.task_name = 'maisaka.planner', 'planner'
    with pytest.raises(ModelResponseTerminatedError):
        await orchestrator._attempt_request_on_model(provider, client, request)
    assert client.get_response.await_count == 1


@pytest.mark.asyncio
async def test_empty_choices_strips_images_on_retry_and_switches_model_when_exhausted(monkeypatch):
    from src.llm_models.exceptions import ModelAttemptFailed
    from src.llm_models.payload_content.context_item import ContextImagePart, ContextItemMeta, ContextTextPart, UserMessageItem

    async def resolve(feature, action, payload):
        if action == 'empty_choices':
            retry = payload['attempt'] == 1
            return True, {'retry': retry, 'retry_after': 0, 'strip_images': retry, 'switch_model': not retry}
        return False, {}

    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolve)
    monkeypatch.setattr('src.llm_models.utils_model.mark_request_succeeded', lambda *a: None)
    monkeypatch.setattr('src.llm_models.utils_model.update_failed_request_attempt', lambda *a, **k: None)
    monkeypatch.setattr('src.llm_models.utils_model.save_failed_request_snapshot', lambda **k: None)
    provider = APIProvider(name='test', api_key='test', base_url='https://example.invalid', max_retry=5)
    item = UserMessageItem(
        meta=ContextItemMeta.create(),
        parts=(ContextTextPart('hello'), ContextImagePart(image_format='gif', image_base64='R0lGODlhAQABAAAAACw=')),
    )
    request = ResponseRequest(
        model_info=ModelInfo(model_type='llm', name='test-model', model_identifier='test-model', api_provider='test'),
        context_items=[item],
    )
    client = SimpleNamespace(get_response=AsyncMock(side_effect=[
        ModelResponseTerminatedError('empty_choices'),
        ModelResponseTerminatedError('empty_choices'),
    ]))
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.request_type, orchestrator.task_name = 'maisaka.planner', 'planner'
    orchestrator._record_success_generation_attempt = Mock()
    orchestrator._schedule_llm_retry_event = Mock()
    with pytest.raises(ModelAttemptFailed):
        await orchestrator._attempt_request_on_model(provider, client, request)
    assert client.get_response.await_count == 2
    first_req = client.get_response.await_args_list[0].args[0]
    second_req = client.get_response.await_args_list[1].args[0]
    assert any(isinstance(p, ContextImagePart) for p in first_req.context_items[0].parts)
    assert not any(isinstance(p, ContextImagePart) for p in second_req.context_items[0].parts)
