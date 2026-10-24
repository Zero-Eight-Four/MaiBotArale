from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("meaning,expected", [(None, 0), ("本群调侃反复修改颜色的任务", 1)])
async def test_new_candidates_require_explanation_and_remain_unconfirmed(monkeypatch, meaning, expected):
    import src.learners.jargon_miner as module

    persisted = []

    @contextmanager
    def database(*args, **kwargs):
        yield SimpleNamespace(
            exec=lambda _: SimpleNamespace(all=lambda: []),
            add=persisted.append,
            flush=lambda: None,
        )

    monkeypatch.setattr(module, "get_db_session", database)
    monkeypatch.setattr(module.JargonConfigUtils, "resolve_jargon_group_scope", lambda _: ({"test"}, False))
    miner = module.JargonMiner("test", "测试群")

    async def hook(*args, **kwargs):
        return SimpleNamespace(
            aborted=False,
            kwargs={"initial_meanings": {"木鱼工程": meaning} if meaning else {}},
        )

    monkeypatch.setattr(miner, "_get_runtime_manager", lambda: SimpleNamespace(invoke_hook=hook))
    group = [{"platform": "qq", "message_id": "definition-1"}]
    entries = [dict(content="木鱼工程", raw_content={"我们把修改颜色叫木鱼工程"}, evidence_messages=[group])]
    assert await miner.process_extracted_entries(entries) == (expected, 0)
    if expected:
        assert persisted[0].meaning == meaning
        assert persisted[0].is_jargon is False
