from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.tooling import ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.reasoning_engine import MaisakaReasoningEngine

MAX_CONSECUTIVE_WEB_READS = 12


def make_engine():
    registry = SimpleNamespace(
        invoke=AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="fetch_url",
                success=True,
                content="网页正文",
            )
        )
    )
    return MaisakaReasoningEngine(
        SimpleNamespace(
            session_id="test",
            _tool_registry=registry,
            _max_internal_rounds=10,
            log_prefix="[test]",
        )
    )


async def invoke(engine, name):
    return await engine._invoke_tool_with_extension_policy(
        ToolInvocation(tool_name=name, arguments={}),
        None,
    )


@pytest.mark.asyncio
async def test_read_above_limit_and_search_are_blocked_in_same_batch():
    engine = make_engine()
    results = [await invoke(engine, "fetch_url") for _ in range(MAX_CONSECUTIVE_WEB_READS + 1)]
    assert all(r.success for r in results[:MAX_CONSECUTIVE_WEB_READS])
    assert not results[-1].success and results[-1].metadata["extension_limited"]
    assert engine._runtime._tool_registry.invoke.await_count == MAX_CONSECUTIVE_WEB_READS
    assert not (await invoke(engine, "web_search")).success
    await invoke(engine, "reply")
    assert not (await invoke(engine, "fetch_url")).success
    assert engine._runtime._tool_registry.invoke.await_count == MAX_CONSECUTIVE_WEB_READS + 1


@pytest.mark.asyncio
async def test_search_and_discovery_cannot_reset_read_chain():
    engine = make_engine()
    for _ in range(MAX_CONSECUTIVE_WEB_READS):
        assert (await invoke(engine, "web_search")).success
        assert (await invoke(engine, "tool_search")).success
        assert (await invoke(engine, "fetch_url")).success
    assert not (await invoke(engine, "fetch_url")).success
    assert engine._runtime._tool_registry.invoke.await_count == MAX_CONSECUTIVE_WEB_READS * 3


@pytest.mark.asyncio
async def test_new_turn_gets_fresh_budget_and_failures_are_counted():
    engine = make_engine()
    engine._runtime._tool_registry.invoke.return_value = ToolExecutionResult(
        tool_name="fetch_url",
        success=False,
        error_message="403",
    )
    for _ in range(MAX_CONSECUTIVE_WEB_READS):
        await invoke(engine, "fetch_url")
    assert not (await invoke(engine, "fetch_url")).success
    engine._start_tool_policy_cycle()
    result = await invoke(engine, "fetch_url")
    assert not result.metadata.get("extension_limited")
    assert engine._runtime._tool_registry.invoke.await_count == MAX_CONSECUTIVE_WEB_READS + 1


@pytest.mark.asyncio
async def test_search_summary_counts_as_read_and_sessions_are_independent():
    engine, other = make_engine(), make_engine()
    for _ in range(MAX_CONSECUTIVE_WEB_READS):
        await engine._invoke_tool_with_extension_policy(
            ToolInvocation(tool_name="web_search", arguments={"mode": "summary"}),
            None,
        )
    assert not (await invoke(engine, "fetch_url")).success
    assert (await invoke(other, "fetch_url")).success


@pytest.mark.asyncio
async def test_non_web_work_breaks_chain_before_limit():
    engine = make_engine()
    for name in ("fetch_url", "fetch_url", "query_memory", "fetch_url"):
        await invoke(engine, name)
    for _ in range(MAX_CONSECUTIVE_WEB_READS - 1):
        assert (await invoke(engine, "fetch_url")).success
    assert not (await invoke(engine, "fetch_url")).success


@pytest.mark.asyncio
async def test_last_round_keeps_reply_and_wait_and_rejects_hidden_web_calls():
    engine = make_engine()
    definitions = [
        ToolSpec(name=n).to_llm_definition()
        for n in ("fetch_url", "web_search", "reply", "wait", "send_emoji", "query_memory")
    ]
    unchanged, hints = await engine._prepare_extension_tool_policy(definitions, round_index=8)
    assert unchanged == definitions and hints == []
    allowed, hints = await engine._prepare_extension_tool_policy(definitions, round_index=9)
    assert [d["name"] for d in allowed] == ["reply", "wait", "send_emoji", "query_memory"]
    assert hints and "不回复" in hints[0]
    assert not (await invoke(engine, "fetch_url")).success
    assert engine._runtime._tool_registry.invoke.await_count == 0
    assert (await invoke(engine, "reply")).success


@pytest.fixture(autouse=True)
def web_policy_bridge(monkeypatch):
    from src.plugin_runtime import policy_bridge

    read_tools = {"fetch_url", "fetch_urls"}
    web_tools = {"fetch_url", "fetch_urls", "web_search"}
    preserve_tools = {"tool_search"}
    states = {}

    async def resolve(feature, action, payload):
        if feature != "tools":
            return False, None
        cid = (payload.get("session_id"), payload.get("cycle_id"))
        st = states.setdefault(cid, {"consecutive": 0, "exhausted": False, "last_round": False})
        if action == "prepare":
            defs = list(payload.get("definitions") or [])
            if payload.get("final_round"):
                st["last_round"] = True
                return True, {
                    "definitions": [d for d in defs if d.get("function", d).get("name") not in web_tools],
                    "hints": ["当前已到达本轮最后一步，如信息不足请先礼貌说明或不回复。"],
                }
            return True, {"definitions": defs, "hints": []}
        if action == "before_execute":
            name = str(payload.get("name") or "")
            args = payload.get("arguments") or {}
            is_read = name in read_tools or (name == "web_search" and args.get("mode") == "summary")
            if st["last_round"] and name in web_tools:
                return True, {"blocked": True, "reason": "已达轮次上限", "metadata": {"extension_limited": True}}
            if st["exhausted"] and name in web_tools:
                return True, {"blocked": True, "reason": "连续读取已达上限", "metadata": {"extension_limited": True}}
            if is_read:
                if st["consecutive"] >= MAX_CONSECUTIVE_WEB_READS:
                    st["exhausted"] = True
                    return True, {
                        "blocked": True,
                        "reason": "连续读取已达上限",
                        "metadata": {"extension_limited": True},
                    }
                st["consecutive"] += 1
            elif name not in web_tools and name not in preserve_tools:
                st["consecutive"] = 0
            return True, {"blocked": False}
        return False, None

    monkeypatch.setattr(policy_bridge, "resolve_policy", resolve)
