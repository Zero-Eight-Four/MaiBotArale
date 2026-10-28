"""Exercise shared empty-output budgets across actual orchestrator branches."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio
import inspect
import sys

import pytest

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.exceptions import EmptyResponseException, ModelResponseTerminatedError, ReqAbortException
from src.llm_models.model_client.base_client import RequestTraceContext, ResponseRequest
from src.llm_models.utils_model import LLMOrchestrator
from src.plugin_runtime import policy_bridge

path = Path(__file__).resolve().parents[1] / "plugins/arale-policy/runtime/plugin.py"
spec = spec_from_file_location("actual_retry_policy", path)
policy_module = module_from_spec(spec)
sys.modules[spec.name] = policy_module
spec.loader.exec_module(policy_module)


@pytest.fixture
def attempt(monkeypatch):
    policy = policy_module.RuntimePolicy()
    policy._plugin_config_instance = policy_module.Config(empty_choices_retry_seconds=0)
    policy._set_context(SimpleNamespace(logger=Mock()))

    async def resolve(feature, action, payload):
        fn = {"empty_choices": policy.retry_empty_choices, "empty_response": policy.retry_empty}.get(action)
        if fn is None:
            return False, None
        result = await fn(feature=feature, action=action, payload=payload)
        kw = result.get("modified_kwargs", {})
        return kw.get("handled", False), kw.get("result")

    monkeypatch.setattr(policy_bridge, "resolve_policy", resolve)
    module = inspect.getmodule(LLMOrchestrator)
    monkeypatch.setattr(module, "save_failed_request_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(module, "update_failed_request_attempt", lambda *a, **k: None)
    request = ResponseRequest(
        model_info=ModelInfo(name="m", model_type="llm", model_identifier="m", api_provider="test"),
        context_items=[],
        trace_context=RequestTraceContext(),
        interrupt_flag=asyncio.Event(),
    )
    provider = APIProvider(
        name="test", api_key="test", base_url="https://example.invalid", max_retry=5, retry_interval=1
    )
    runner = object.__new__(LLMOrchestrator)
    runner.request_type = "maisaka.planner"
    runner.task_name = "planner"
    runner._schedule_llm_retry_event = Mock()
    return SimpleNamespace(runner=runner, policy=policy, request=request, provider=provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("kinds", [["body", "body"], ["body", "choices"], ["choices", "body"], ["choices", "choices"]])
async def test_empty_body_and_choices_share_one_budget(attempt, kinds):
    errors = [
        EmptyResponseException(message="fixture empty body")
        if k == "body"
        else ModelResponseTerminatedError("empty_choices")
        for k in kinds
    ]
    client = SimpleNamespace(get_response=AsyncMock(side_effect=errors))
    with pytest.raises(ModelResponseTerminatedError) as raised:
        await attempt.runner._attempt_request_on_model(attempt.provider, client, attempt.request)
    assert client.get_response.await_count == 2
    assert raised.value.failure_kind == "empty_output"
    assert raised.value.response_metadata["retry_exhausted"] is True
    assert attempt.request.trace_context.empty_output_failures == 2


@pytest.mark.asyncio
async def test_one_attempt_configuration_does_not_force_a_second_call(attempt):
    attempt.policy.config.planner_empty_max_attempts = 1
    client = SimpleNamespace(get_response=AsyncMock(side_effect=ModelResponseTerminatedError("empty_choices")))
    with pytest.raises(ModelResponseTerminatedError):
        await attempt.runner._attempt_request_on_model(attempt.provider, client, attempt.request)
    assert client.get_response.await_count == 1


@pytest.mark.asyncio
async def test_explicit_block_does_not_enter_unknown_empty_retries(attempt):
    client = SimpleNamespace(
        get_response=AsyncMock(
            side_effect=ModelResponseTerminatedError(
                "empty_choices", response_metadata={"prompt_feedback": {"blockReason": "SAFETY"}}
            )
        )
    )
    with pytest.raises(ModelResponseTerminatedError) as raised:
        await attempt.runner._attempt_request_on_model(attempt.provider, client, attempt.request)
    assert raised.value.failure_kind == "provider_safety" and client.get_response.await_count == 1


@pytest.mark.asyncio
async def test_empty_body_retry_wait_respects_new_message_interrupt(attempt):
    attempt.policy.config.empty_choices_retry_seconds = 10
    attempt.runner._schedule_llm_retry_event.side_effect = lambda **kwargs: attempt.request.interrupt_flag.set()
    client = SimpleNamespace(get_response=AsyncMock(side_effect=EmptyResponseException(message="fixture empty")))
    with pytest.raises(ReqAbortException):
        await attempt.runner._attempt_request_on_model(attempt.provider, client, attempt.request)
    assert client.get_response.await_count == 1
