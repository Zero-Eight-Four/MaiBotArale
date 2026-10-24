"""审核来源、布尔判定与失败隔离，所有模型和入库操作均模拟。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import json

import pytest

from src.config.config import global_config
from src.learners import expression_learner as learner_module
from src.learners import expression_utils as utils
from src.learners.expression_learner import ExpressionLearner
from src.prompt.prompt_manager import prompt_manager


@pytest.fixture(autouse=True)
def prompts_loaded():
    prompt_manager.load_prompts()


def message(text, user_id="user", platform="qq"):
    return SimpleNamespace(platform=platform, processed_plain_text=text,
                           message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user_id)))


def test_source_context_is_bounded_and_excludes_bot(monkeypatch):
    monkeypatch.setattr(learner_module, "is_bot_self", lambda platform, user_id: user_id == "bot")
    messages = [message("BOT_INSTRUCTION", "bot"), message("原话" * 800), message("邻近消息" * 300)]
    context = ExpressionLearner._build_expression_review_context("拒绝", "拉倒吧", [("拒绝", "拉倒吧", "2")], messages)
    assert "BOT_INSTRUCTION" not in context
    data = [json.loads(line) for line in context.splitlines()[1:]]
    assert data[0]["source_id"] == "2" and data[0]["is_candidate_source"]
    assert len(data[0]["content"]) == 800 and len(data[1]["content"]) == 300
    assert not data[1]["is_candidate_source"]


@pytest.mark.parametrize("source", ["0", "100", "bad"])
def test_invalid_source_is_not_invented(source):
    context = ExpressionLearner._build_expression_review_context("拒绝", "拉倒吧", [("拒绝", "拉倒吧", source)], [message("原话")])
    assert "未找到有效来源" in context


def test_source_from_bot_is_not_used(monkeypatch):
    monkeypatch.setattr(learner_module, "is_bot_self", lambda *args: True)
    context = ExpressionLearner._build_expression_review_context("拒绝", "拉倒吧", [("拒绝", "拉倒吧", "1")], [message("BOT_ONLY")])
    assert "BOT_ONLY" not in context and "未找到有效来源" in context


def test_candidates_are_deduplicated_and_bounded(monkeypatch):
    monkeypatch.setattr(learner_module, "is_bot_self", lambda *args: False)
    learner = ExpressionLearner("test")
    items = [("场景", f"表达{i}", "1") for i in range(10)]
    filtered = learner._filter_expressions([items[0]] + items, [message("真实发言")])
    filtered = list(dict.fromkeys(filtered))[:5]
    assert filtered == [("场景", f"表达{i}") for i in range(5)]


def test_failed_candidate_does_not_abort_later_candidate(monkeypatch):
    monkeypatch.setattr(learner_module, "is_bot_self", lambda *args: False)
    monkeypatch.setattr(global_config.expression, "expression_self_reflect", True)
    learner = ExpressionLearner("test")
    learner._build_multi_learning_messages = AsyncMock(return_value=[])
    learner._log_learning_context_preview = lambda *args, **kwargs: None
    learner._get_session_display_name = lambda session: session

    review_results = [
        {"suitable": False, "reason": "离线", "error": "RuntimeError"},
        {"suitable": True, "reason": "有原文依据", "error": None},
    ]

    async def hook(name, **kwargs):
        if name == "expression.learn.after_extract":
            return SimpleNamespace(aborted=False, kwargs={**kwargs, "max_candidates": 5})
        if name == "expression.learn.before_upsert":
            return SimpleNamespace(aborted=False, kwargs={**kwargs, "suitability_review": review_results.pop(0)})
        return SimpleNamespace(aborted=False, kwargs=kwargs)

    learner._get_runtime_manager = lambda: SimpleNamespace(invoke_hook=hook)
    learner._upsert_expression_to_db = AsyncMock(return_value=object())
    learner._sync_expression_vector_index_batch = AsyncMock()
    candidates = [{"situation": "拒绝", "style": "拉倒吧", "id": "1"},
                  {"situation": "同意", "style": "说得对", "id": "2"}]
    monkeypatch.setattr(learner_module.express_learn_model, "generate_response_with_context",
                        AsyncMock(return_value=SimpleNamespace(response=json.dumps(candidates, ensure_ascii=False))))
    logs = []
    monkeypatch.setattr(learner_module, "append_ai_review_log", lambda **kwargs: logs.append(kwargs))
    success = asyncio.run(learner._run_learning_batch([message("拉倒吧"), message("说得对")], learning_session_id="test"))
    assert success
    assert learner._upsert_expression_to_db.await_count == 1
    assert learner._upsert_expression_to_db.call_args.args == ("同意", "说得对")
    assert logs[0]["passed"] is False and logs[0]["error"]
    assert logs[1]["passed"] is True
