import pytest
from src.llm_models.exceptions import ModelResponseTerminatedError


@pytest.mark.parametrize(
    "reason,meta,kind",
    [
        ("empty_choices", {}, "empty_output"),
        ("empty_output", {}, "empty_output"),
        ("empty_choices", {"prompt_feedback": {"blockReason": "SAFETY"}}, "provider_safety"),
        ("empty_choices", {"prompt_feedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}}, "empty_output"),
        ("content_filter", {}, "provider_safety"),
        ("safety", {}, "provider_safety"),
        ("unsupported_location", {}, "provider_terminated"),
    ],
)
def test_failures_preserve_facts_instead_of_assuming_moderation(reason, meta, kind):
    assert ModelResponseTerminatedError(reason, response_metadata=meta).failure_kind == kind


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prompt_block", [False, True])
def test_native_gemini_blocks_are_not_retried_as_unknown_empty(stream, prompt_block):
    from types import SimpleNamespace
    from io import StringIO
    from src.llm_models.model_client.gemini_client import _default_normal_response_parser, _build_stream_api_response

    response = SimpleNamespace(
        prompt_feedback=SimpleNamespace(block_reason="SAFETY" if prompt_block else None),
        candidates=[] if prompt_block else [SimpleNamespace(finish_reason="SAFETY", content=None)],
    )
    with pytest.raises(ModelResponseTerminatedError) as caught:
        if stream:
            _build_stream_api_response(
                content_buffer=StringIO(),
                reasoning_buffer=StringIO(),
                tool_calls_buffer=[],
                last_response=response,
                logical_turn_id="test",
            )
        else:
            _default_normal_response_parser(response, logical_turn_id="test")
    assert caught.value.failure_kind == "provider_safety"
