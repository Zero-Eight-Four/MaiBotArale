import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.common.data_models.message_component_data_model import TextComponent
from pytests.plugin_runtime.test_arale_migration import load, plugin
_admission=load('Arale-jargon-curator')[0]
from importlib import import_module
_helpers=import_module(_admission.__name__+'.admission')
parse_admission,plausible_term=_helpers.parse_admission,_helpers.plausible_term
from src.learners.jargon_learner import JargonLearner, JargonLearningSourceItem


def item(text, message_id):
    original = SimpleNamespace(platform='qq', session_id='test', message_id=message_id, is_notify=False, is_emoji=False,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id='real-human')),
        raw_message=SimpleNamespace(components=[TextComponent(text=text)]))
    return JargonLearningSourceItem(original,text)


def approved(term='木鱼工程',quote='我们把来回修改颜色叫木鱼工程'):
    return dict(content=term,approve=True,category='group_specific',confidence='high',
                meaning='本群调侃反复修改界面颜色的小任务',reason='原文明确给出了本群的特别称呼',evidence_quote=quote)


@pytest.mark.parametrize('term',['A!A!O desu!','⭕','123','a'])
def test_noise_rejected_before_model_call(term):
    assert not plausible_term(term)


def test_single_definition_can_be_admitted():
    quote='我们把来回修改颜色叫木鱼工程'
    assert parse_admission(json.dumps([approved()],ensure_ascii=False),[{'content':'木鱼工程','hits':[{'text':quote}]}]) == {'木鱼工程':'本群调侃反复修改界面颜色的小任务'}


@pytest.mark.parametrize('patch',[{'approve':'true'},{'category':'technical'},{'confidence':'low'},
                                  {'meaning':''},{'reason':''},{'evidence_quote':'我们把不存在的证据叫木鱼工程'}])
def test_unsupported_or_non_group_vocabulary_not_admitted(patch):
    decision=approved()
    decision.update(patch)
    assert parse_admission(json.dumps([decision],ensure_ascii=False),[{'content':'木鱼工程','hits':[{'text':'我们把来回修改颜色叫木鱼工程'}]}]) == {}


def test_single_verified_source_does_not_require_repeated_usage():
    quote='今天又在木鱼工程上浪费时间'
    assert parse_admission(json.dumps([approved(quote=quote)],ensure_ascii=False),[{'content':'木鱼工程','hits':[{'text':quote}]}]) == {'木鱼工程':'本群调侃反复修改界面颜色的小任务'}


def test_repeated_usage_requires_bounded_admission():
    candidates=[]
    results=[]
    for i in range(5):
        term=f'木鱼工程{i}'
        quote=f'又开始做{term}了'
        candidates.append({'content':term,'hits':[{'text':quote},{'text':quote+'第二回'}]})
        results.append(approved(term,quote))
    assert len(parse_admission(json.dumps(results,ensure_ascii=False),candidates)) == 3


@pytest.mark.asyncio
async def test_ordinary_candidates_need_model_approval(monkeypatch,tmp_path):
    from unittest.mock import AsyncMock
    curator,_=plugin('Arale-jargon-curator',tmp_path)
    curator.ctx.llm.generate=AsyncMock(return_value={"success":True,"content":"[]"})
    out=await curator.admit_before_persist(entries=[{
        "content":"路由","raw_content":["路由又不通了"],"evidence_messages":[[{"platform":"qq","message_id":"1"}]]}],
        sources=[{"text":"路由又不通了","platform":"qq","message_id":"1"}])
    assert out["modified_kwargs"]["entries"]==[]
    curator.ctx.llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_replayed_message_is_not_repeated_usage(monkeypatch,tmp_path):
    from unittest.mock import AsyncMock
    curator,_=plugin('Arale-jargon-curator',tmp_path)
    curator.ctx.llm.generate=AsyncMock(return_value={"success":True,"content":"[]"})
    source={"text":"今天又在木鱼工程上浪费时间","platform":"qq","message_id":"same"}
    await curator.admit_before_persist(entries=[{"content":"木鱼工程","raw_content":[source["text"]],
        "evidence_messages":[[{"platform":"qq","message_id":"same"}]]}],sources=[source,dict(source)])
    prompt=curator.ctx.llm.generate.await_args.args[0]
    # Inspect the real evidence envelope in the formatted prompt, not a removed learner helper.
    assert prompt.count('"message_id": "same"')==1


@pytest.mark.asyncio
async def test_admission_failure_does_not_persist_candidates(monkeypatch,tmp_path):
    curator,_=plugin('Arale-jargon-curator',tmp_path)
    async def failing(*args,**kwargs):
        raise RuntimeError('upstream unavailable')
    curator.ctx.llm.generate=failing
    entry={"content":"木鱼工程","raw_content":["我们把来回修改颜色叫木鱼工程"],
           "evidence_messages":[[{"platform":"qq","message_id":"real"}]]}
    # Hook is declared ABORT: errors propagate to official dispatch, which prevents persistence.
    with pytest.raises(RuntimeError,match='upstream unavailable'):
        await curator.admit_before_persist(entries=[entry],sources=[{
            "text":"我们把来回修改颜色叫木鱼工程","platform":"qq","message_id":"real"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("meaning,expected", [(None, 0), ("本群调侃反复修改颜色的任务", 1)])
async def test_new_candidates_require_explanation_and_remain_unconfirmed(monkeypatch, meaning, expected):
    from contextlib import contextmanager
    import src.learners.jargon_miner as module
    persisted = []
    @contextmanager
    def database(*args, **kwargs):
        yield SimpleNamespace(exec=lambda _: SimpleNamespace(all=lambda: []),
                              add=persisted.append, flush=lambda: None)
    monkeypatch.setattr(module, "get_db_session", database)
    monkeypatch.setattr(module.JargonConfigUtils, "resolve_jargon_group_scope", lambda _: ({"test"}, False))
    miner = module.JargonMiner("test", "测试群")
    async def hook(*args, **kwargs):
        return SimpleNamespace(aborted=False, kwargs={"initial_meanings":{"木鱼工程":meaning} if meaning else {}})
    monkeypatch.setattr(miner, "_get_runtime_manager", lambda: SimpleNamespace(invoke_hook=hook))
    group = [{"platform": "qq", "message_id": "definition-1"}]
    entries = [dict(content="木鱼工程", raw_content={"我们把修改颜色叫木鱼工程"}, evidence_messages=[group])]
    assert await miner.process_extracted_entries(entries) == (expected, 0)
    if expected:
        assert persisted[0].meaning == meaning
        assert persisted[0].is_jargon is False
