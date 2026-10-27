"""Do not retry/re-route explicit termination or zero-candidate CPA responses."""
from types import SimpleNamespace
import pytest
from src.llm_models.exceptions import ModelResponseTerminatedError
from src.llm_models.model_client.openai_client import _default_normal_response_parser

@pytest.mark.parametrize("reason", [None, "content_filter", "prohibited_content", "safety"])
def test_terminal_response_is_distinct_from_retryable_empty_text(reason):
    response = SimpleNamespace(choices=[] if reason is None else [SimpleNamespace(finish_reason=reason)])
    with pytest.raises(ModelResponseTerminatedError) as error:
        _default_normal_response_parser(response, reasoning_parse_mode=None, tool_argument_parse_mode=None,
            reasoning_key="", logical_turn_id="test", max_tokens=None, trace_context=None)
    assert error.value.reason == (reason or "empty_choices")


def test_empty_choices_keeps_response_identity_and_feedback_for_policy():
    response = SimpleNamespace(choices=[], id='response-1', model='test', model_extra={
        'promptFeedback': {'blockReason': 'PROHIBITED_CONTENT'},
    })
    with pytest.raises(ModelResponseTerminatedError) as error:
        _default_normal_response_parser(response, reasoning_parse_mode=None, tool_argument_parse_mode=None,
            reasoning_key='', logical_turn_id='test', max_tokens=None, trace_context=None)
    assert error.value.response_metadata == {
        'response_id': 'response-1', 'response_model': 'test',
        'prompt_feedback': {'blockReason': 'PROHIBITED_CONTENT'},
    }
