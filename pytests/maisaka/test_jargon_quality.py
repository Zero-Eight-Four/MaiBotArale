from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.common.data_models.message_component_data_model import TextComponent
from src.learners.jargon_learner import JargonLearner, JargonLearningSourceItem
from src.learners.jargon_matching import contains_jargon
from src.learners.jargon_miner import JargonMiner
from src.common.database.database_model import JargonCreatedBy

def _supported(result,evidence):
    import asyncio
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from pytests.plugin_runtime.test_arale_migration import plugin
    with TemporaryDirectory() as d:
        p,_=plugin('Arale-jargon-curator',Path(d))
        out=asyncio.run(p.finalize(content="烟测",meaning="本群用语",comparison_result=result,raw_content_list=[evidence],is_jargon=True))
        return out['modified_kwargs']['is_jargon']



def source(text):
    message = SimpleNamespace(
        platform="qq", session_id="test", message_id=text,
        is_notify=False, is_emoji=False,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id="human-test")),
        raw_message=SimpleNamespace(components=[TextComponent(text=text)]),
    )
    return JargonLearningSourceItem(message, text)


@pytest.mark.parametrize("text,term,expected", [
    ("烟测又来了", "烟测", True), ("学习capisce", "API", False),
    ("调用API接口", "api", True), ("API2接口", "api", False),
    ("词库只剩a", "a", False), ("编号1234", "123", False),
])
def test_term_boundaries(text, term, expected):
    assert contains_jargon(text, term) is expected


def test_cache_only_counts_actual_user_text():
    learner = JargonLearner("test")
    miner = SimpleNamespace(get_cached_jargons=lambda: ["烟测"])
    items = [source("普通文字"), source("又做烟测了")]
    assert learner._check_cached_jargons_in_messages(items, miner) == [("烟测", "2")]


def test_official_source_requires_a_real_message_and_session():
    with pytest.raises(TypeError):
        JargonLearningSourceItem(None, "伪造的来源")
    with pytest.raises(ValueError):
        JargonLearningSourceItem(SimpleNamespace(platform="qq",message_id="m",session_id=""), "无归属")



@pytest.mark.parametrize("changes", [{"is_similar": "false"}, {"has_special_meaning": "true"},
                                      {"confidence": "low"}, {"evidence_quote": "编造的聊天原文"}])
def test_uncertain_or_invented_classification_is_rejected(changes):
    result = dict(is_similar=False, has_special_meaning=True, confidence="high", evidence_quote="本群把烟测当作过度检查")
    result.update(changes)
    assert not _supported(result, "本群把烟测当作过度检查")


def test_supported_classification():
    assert _supported(
        dict(is_similar=False, has_special_meaning=True, confidence="high", evidence_quote="本群把烟测当作过度检查"),
        "前文 本群把烟测当作过度检查 后文")


@pytest.mark.asyncio
async def test_replayed_evidence_does_not_increment_count(monkeypatch):
    import src.learners.jargon_miner as module
    group = [{"platform": "qq", "message_id": "original-1"}]
    import json
    record = SimpleNamespace(created_by=JargonCreatedBy.AI, is_global=False,
                             session_id_dict='{"test": 1}', evidence_messages=json.dumps([group]))
    @contextmanager
    def db_session():
        yield SimpleNamespace(exec=lambda _: SimpleNamespace(all=lambda: [record]))
    monkeypatch.setattr(module, "get_db_session", db_session)
    monkeypatch.setattr(module.JargonConfigUtils, "resolve_jargon_group_scope", lambda _: ({"test"}, False))
    miner = JargonMiner("test", "测试")
    async def hook(*args, **kwargs):
        return SimpleNamespace(aborted=False, kwargs={})
    monkeypatch.setattr(miner, "_get_runtime_manager", lambda: SimpleNamespace(invoke_hook=hook))
    def unexpected(*args):
        pytest.fail("Repeated evidence must not update count")
    monkeypatch.setattr(miner, "_update_jargon", unexpected)
    assert await miner.process_extracted_entries([dict(content="烟测", raw_content={"烟测"}, evidence_messages=[group])]) == (0, 0)
