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
def prompts_loaded(monkeypatch,tmp_path):
    prompt_manager.load_prompts()
    from pytests.plugin_runtime.test_arale_migration import plugin
    curator,_=plugin('Arale-expression-curator',tmp_path)
    judge=SimpleNamespace(generate_response=AsyncMock())
    monkeypatch.setattr(utils,'judge_llm',judge,raising=False)
    async def generate(prompt,**kwargs):
        response=await judge.generate_response(prompt=prompt,**kwargs)
        return {'success':True,'response':response.response}
    curator.ctx.llm.generate=generate
    async def get_config(key, default=""):
        return {"personality.personality":global_config.personality.personality,
                "personality.reply_style":global_config.personality.reply_style}.get(key,default)
    curator.ctx.config.get=get_config
    return curator


@pytest.fixture
def native_review(prompts_loaded):
    async def review(situation,style,*,review_context=""):
        out=await prompts_loaded.review_before_upsert(situation=situation,style=style,
            review_context=review_context,review_enabled=True)
        result=out["modified_kwargs"]["suitability_review"]
        return result["suitable"],result["reason"],result.get("error")
    return review



def message(text, user_id="user", platform="qq"):
    return SimpleNamespace(platform=platform, processed_plain_text=text,
                           message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user_id)))


@pytest.mark.parametrize("suitable", [True, False, "false", "true", 0, 1, None])
def test_only_json_booleans_are_accepted(monkeypatch, suitable, native_review):
    model = AsyncMock(return_value=SimpleNamespace(response=json.dumps({"suitable": suitable, "reason": "按场景判断"})))
    monkeypatch.setattr(utils.judge_llm, "generate_response", model)
    passed, reason, error = asyncio.run(native_review("不同意", "拉倒吧", review_context="原文依据"))
    assert passed is (suitable is True)
    assert bool(error) is (not isinstance(suitable, bool))


def test_actual_prompt_contains_persona_style_and_source(monkeypatch,native_review):
    monkeypatch.setattr(global_config.personality, "personality", "TEST_PERSONA")
    monkeypatch.setattr(global_config.personality, "reply_style", "TEST_STYLE")
    model = AsyncMock(return_value=SimpleNamespace(response='{"suitable": true, "reason":"符合场景"}'))
    monkeypatch.setattr(utils.judge_llm, "generate_response", model)
    assert asyncio.run(native_review("拒绝", "拉倒吧", review_context="TEST_SOURCE"))[0]
    prompt = model.call_args.kwargs["prompt"]
    assert all(x in prompt for x in ["TEST_PERSONA", "TEST_STYLE", "TEST_SOURCE", "拉倒吧"])


def test_model_exception_is_a_review_error(monkeypatch,native_review):
    monkeypatch.setattr(utils.judge_llm, "generate_response", AsyncMock(side_effect=RuntimeError("offline")))
    passed, reason, error = asyncio.run(native_review("拒绝", "拉倒吧"))
    assert not passed and error == "RuntimeError"


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
    from pytests.plugin_runtime.test_arale_migration import plugin
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        p,_=plugin('Arale-expression-curator',Path(d))
        out=asyncio.run(p.filter_candidates(expressions=items))
    filtered=list(dict.fromkeys(filtered))[:out["modified_kwargs"]["max_candidates"]]
    assert filtered == [("场景", f"表达{i}") for i in range(5)]


def test_failed_candidate_does_not_abort_later_candidate(monkeypatch,prompts_loaded):
    monkeypatch.setattr(learner_module, "is_bot_self", lambda *args: False)
    monkeypatch.setattr(global_config.expression, "expression_self_reflect", True)
    learner = ExpressionLearner("test")
    learner._build_multi_learning_messages = AsyncMock(return_value=[])
    learner._log_learning_context_preview = lambda *args, **kwargs: None
    learner._get_session_display_name = lambda session: session
    async def hook(name,**kwargs):
        method={"expression.learn.after_extract":prompts_loaded.filter_candidates,
                "expression.learn.before_upsert":prompts_loaded.review_before_upsert}[name]
        out=await method(**kwargs)
        return SimpleNamespace(aborted=False,kwargs={**kwargs,**out.get("modified_kwargs",{})})
    learner._get_runtime_manager = lambda: SimpleNamespace(invoke_hook=hook)
    learner._upsert_expression_to_db = AsyncMock(return_value=object())
    learner._sync_expression_vector_index_batch = AsyncMock()
    candidates = [{"situation": "拒绝", "style": "拉倒吧", "id": "1"},
                  {"situation": "同意", "style": "说得对", "id": "2"}]
    monkeypatch.setattr(learner_module.express_learn_model, "generate_response_with_context",
                        AsyncMock(return_value=SimpleNamespace(response=json.dumps(candidates, ensure_ascii=False))))
    judge = AsyncMock(side_effect=[RuntimeError("offline"), SimpleNamespace(response='{"suitable":true,"reason":"有原文依据"}')])
    monkeypatch.setattr(utils.judge_llm, "generate_response", judge)
    logs = []
    monkeypatch.setattr(learner_module, "append_ai_review_log", lambda **kwargs: logs.append(kwargs))
    success = asyncio.run(learner._run_learning_batch([message("拉倒吧"), message("说得对")], learning_session_id="test"))
    assert success and judge.await_count == 2
    assert learner._upsert_expression_to_db.await_count == 1
    assert learner._upsert_expression_to_db.call_args.args == ("同意", "说得对")
    assert logs[0]["passed"] is False and logs[0]["error"]
    assert logs[1]["passed"] is True
    assert "拉倒吧" in judge.call_args_list[0].kwargs["prompt"]
