"""A region rejection stops the request; identical background tasks do not hammer it."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.exceptions import ModelResponseTerminatedError, RespNotOkException
from src.llm_models.model_client.base_client import ResponseRequest
from src.llm_models.utils_model import LLMOrchestrator
from src.plugin_runtime import policy_bridge


@pytest.mark.asyncio
async def test_region_failure_stops_retries_and_persists_per_task_cooldown(monkeypatch):
    deadlines = {}

    async def resolve(feature, action, payload):
        p = payload or {}
        if action == 'region_failed':
            deadlines[p['key']] = p['now'] + 300
        deadline = deadlines.get(p['key'], 0.0)
        return True, {'allow': deadline <= p['now'], 'retry_at': deadline}

    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolve)
    provider = APIProvider(name='test', api_key='test', base_url='https://example.invalid', max_retry=3)
    request = ResponseRequest(model_info=ModelInfo(model_type='llm', name='test-model',
                              model_identifier='test-model', api_provider='test'), context_items=[])
    client = SimpleNamespace(get_response=AsyncMock(side_effect=RespNotOkException(
        400, 'User location is not supported for the API use.')))
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.request_type = 'expression.selection.profile_probe'
    orchestrator.task_name = 'embedding'
    with pytest.raises(ModelResponseTerminatedError) as error:
        await orchestrator._attempt_request_on_model(provider, client, request)
    assert error.value.status_code == 400 and error.value.reason == 'unsupported_location'
    assert client.get_response.await_count == 1  # configured max_retry=3
    assert '地区不支持' in str(error.value)
    with pytest.raises(ModelResponseTerminatedError) as error:
        await orchestrator._attempt_request_on_model(provider, client, request)
    assert error.value.reason == 'unsupported_location_cooldown'
    assert client.get_response.await_count == 1
    orchestrator.request_type = 'other.task'
    with pytest.raises(ModelResponseTerminatedError):
        await orchestrator._attempt_request_on_model(provider, client, request)
    assert client.get_response.await_count == 2  # other tasks are not globally disabled

@pytest.mark.asyncio
async def test_terminal_result_does_not_select_fallback_and_releases_usage():
    from src.llm_models.utils_model import RequestType
    orchestrator = object.__new__(LLMOrchestrator)
    orchestrator.request_type, orchestrator.task_name = 'test', 'planner'
    orchestrator.model_for_task = SimpleNamespace(model_list=['first', 'fallback'])
    model = SimpleNamespace(name='first')
    orchestrator._select_model = AsyncMock(return_value=(model, object(), object()))
    orchestrator._resolve_effective_session_id = lambda value: value
    orchestrator._build_client_request = lambda **kwargs: object()
    orchestrator._attempt_request_on_model_with_timeout = AsyncMock(side_effect=ModelResponseTerminatedError('unsupported_location', 400))
    orchestrator._adjust_model_usage = Mock()
    with pytest.raises(ModelResponseTerminatedError):
        await orchestrator._execute_request(request_type=RequestType.RESPONSE)
    assert orchestrator._select_model.await_count == 1
    orchestrator._adjust_model_usage.assert_called_once_with('first', usage_penalty_delta=-1)
