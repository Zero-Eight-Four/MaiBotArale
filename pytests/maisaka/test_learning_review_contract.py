from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "review,expected,calls",
    [({"suitable": True, "reason": "合适"}, True, 0), ({"suitable": "yes"}, False, 0), ({}, False, 0), (None, True, 1)],
)
async def test_supplied_review_does_not_trigger_a_second_model(monkeypatch, review, expected, calls):
    import src.learners.expression_learner as module

    check = AsyncMock(return_value=(True, "合适", None))
    monkeypatch.setattr(module, "check_expression_suitability", check)
    monkeypatch.setattr(module, "append_ai_review_log", Mock())
    learner = object.__new__(module.ExpressionLearner)
    learner.session_id = "test"
    assert await learner._check_expression_before_upsert("场景", "表达", review_result=review) is expected
    assert check.await_count == calls


@pytest.mark.asyncio
@pytest.mark.parametrize("review_provided", [False, True])
async def test_missing_jargon_review_has_a_distinct_batch_warning(monkeypatch, review_provided):
    from contextlib import contextmanager
    import src.learners.jargon_miner as module

    @contextmanager
    def database(*a, **kw):
        yield SimpleNamespace(exec=lambda _: SimpleNamespace(all=lambda: []))

    monkeypatch.setattr(module, "get_db_session", database)
    monkeypatch.setattr(module.JargonConfigUtils, "resolve_jargon_group_scope", lambda _: ({"test"}, False))
    log = Mock()
    monkeypatch.setattr(module, "logger", log)
    miner = module.JargonMiner("test", "测试群")
    hook = AsyncMock(
        return_value=SimpleNamespace(aborted=False, kwargs={"initial_meanings": {}} if review_provided else {})
    )
    monkeypatch.setattr(miner, "_get_runtime_manager", lambda: SimpleNamespace(invoke_hook=hook))
    assert await miner.process_extracted_entries(
        [
            {
                "content": "木鱼工程",
                "raw_content": {"颜色修改任务"},
                "evidence_messages": [[{"platform": "qq", "message_id": "one"}]],
            }
        ]
    ) == (0, 0)
    assert log.warning.call_count == (0 if review_provided else 1)
