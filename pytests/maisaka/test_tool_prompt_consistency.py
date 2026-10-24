"""工具声明与当前开关一致，参考资料不会覆盖回复意图。"""

from pathlib import Path
import importlib
import string
import asyncio
import sys
import types

import pytest

from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.config.config import global_config
from src.maisaka import chat_loop_service as service_module
from src.maisaka.builtin_tool import get_all_builtin_tool_specs
from src.maisaka.builtin_tool.reply import get_tool_spec as reply_spec
from src.maisaka.builtin_tool.wait import get_tool_spec as wait_spec
from src.maisaka.chat_loop_service import MaisakaChatLoopService


@pytest.fixture(autouse=True)
def extension_policies(monkeypatch,tmp_path):
    from pytests.plugin_runtime.arale_test_runtime import PolicyRuntime
    from src.plugin_runtime import policy_bridge
    runtime=PolicyRuntime(tmp_path)
    monkeypatch.setattr(policy_bridge,'resolve_policy',runtime.resolve)



def test_native_reply_reference_and_quote_contract():
    from src.plugin_runtime.tool_specs import extend_tool_spec
    native = reply_spec()
    extended = asyncio.run(extend_tool_spec(native))
    assert extended.parameters_schema['required'] == ['msg_id','reply_reference']
    assert ('set_quote' in extended.parameters_schema['properties']) == ('set_quote' in native.parameters_schema['properties'])


@pytest.mark.parametrize('reason,reference', [('意图', '证据'), ('意图', ''), ('', '证据'), ('同一内容', '同一内容'), ('', '')])
def test_reference_keeps_intent_without_duplicate_text(reason, reference):
    text = BaseMaisakaReplyGenerator._build_reply_reference_message(reason, reference)
    if reason:
        assert reason in text
    if reference:
        assert reference in text
    if reason == reference:
        assert text.count(reason) == 1 if reason else text == ""
    if reason or reference:
        assert 'host_internal_reply_context' in text


def test_wait_declares_buffering_and_early_wakeup_and_rejects_zero_in_schema():
    from src.plugin_runtime.tool_specs import extend_tool_spec
    spec = asyncio.run(extend_tool_spec(wait_spec()))
    assert '群聊普通消息缓存' in spec.description
    assert '纠正可唤醒' in spec.description
    assert '私聊新消息可唤醒' in spec.description
    assert 'set_reminder' in spec.description
    assert spec.parameters_schema['properties']['seconds']['minimum'] == 1


def test_google_auto_search_has_no_time_filter_even_for_latest_query():
    plugin_path = Path(__file__).resolve().parents[2] / 'plugins' / 'Arale-web' / 'search'
    package = types.ModuleType('tool_prompt_search_test')
    package.__path__ = [str(plugin_path)]
    sys.modules[package.__name__] = package
    freshness = importlib.import_module(package.__name__ + '.pipelines.freshness')
    google = importlib.import_module(package.__name__ + '.search_engines.google')
    engine = google.GoogleEngine()
    for question in ['今天新闻', '最新模型排名', '刻晴 倒装 话真多你', '历史榜单']:
        resolved = freshness.resolve_freshness(question)
        assert resolved == 'none'
        assert 'tbs=' not in engine._build_url(question, time_range=resolved)
    assert freshness.resolve_freshness('用户明确查最近一周', 'week') == 'week'
