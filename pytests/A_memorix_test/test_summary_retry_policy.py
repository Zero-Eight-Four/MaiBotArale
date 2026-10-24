import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.services import memory_flow_service as flow
from src.services.memory_service import MemoryWriteResult
from src.plugin_runtime import policy_bridge

path = Path(__file__).resolve().parents[2] / 'plugins/Arale-policy/memory/summary_backoff.py'
spec = importlib.util.spec_from_file_location('summary_backoff_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_transient_backoff_is_bounded_persistent_and_per_session(tmp_path):
    store = module.SummaryBackoff(tmp_path / 'state.json')
    store.finish('a', 100, 36, 0, False, 'timeout', 60, 600)
    assert not store.prepare('a', 101, 59)['allow']
    assert store.prepare('a', 102, 60)['allow']
    assert store.prepare('b', 1, 1)['allow']
    store = module.SummaryBackoff(store.path)
    assert not store.prepare('a', 101, 59)['allow']
    store.finish('a', 102, 36, 60, False, 'timeout', 60, 600)
    assert store.states['a']['retry_at'] == 180
    for _ in range(20):
        store.finish('a', 102, 36, 1000, False, 'timeout', 60, 600)
    assert store.states['a']['retry_at'] == 1600


@pytest.mark.asyncio
async def test_real_summary_worker_does_not_replay_terminated_history_or_advance_failed_progress(tmp_path, monkeypatch):
    service = flow.ChatSummaryWritebackService()
    count, now = [100], [1000]
    store = module.SummaryBackoff(tmp_path / 'state.json')
    service._states['s'] = flow.ChatSummaryWritebackState(last_trigger_message_count=0, last_trigger_time=0)
    service._resolve_session_id = lambda m: 's'
    service._extract_message_timestamp = lambda m: 1
    service._count_messages_until_trigger = lambda **kwargs: count[0]
    service._message_threshold = lambda: 1
    service._context_length = lambda: 36
    service._extract_session_user_id = service._extract_session_group_id = lambda m: ''

    async def resolve(feature, action, payload):
        if action == 'summary_prepare':
            return True, {**store.prepare('s', payload['message_count'], now[0]), 'timeout_seconds': 90}
        store.finish('s', payload['message_count'], payload['context_length'], now[0], payload['success'], payload['detail'], 60, 600)
        return True, {}

    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolve)
    ingest = AsyncMock(return_value=MemoryWriteResult(False, detail='LLM failed: empty_choices'))
    monkeypatch.setattr(flow.memory_service, 'ingest_summary', ingest)
    await service._handle_message(SimpleNamespace())
    assert service._states['s'].last_trigger_message_count == 0
    now[0] += 10000
    count[0] += 35
    await service._handle_message(SimpleNamespace())
    assert ingest.await_count == 1  # timer alone never resubmits a terminated window
    count[0] += 1
    ingest.return_value = MemoryWriteResult(True)
    await service._handle_message(SimpleNamespace())
    assert ingest.await_count == 2
    assert service._states['s'].last_trigger_message_count == 136
    assert 's' not in store.states

@pytest.mark.asyncio
async def test_queue_coalesces_pending_triggers_using_latest_chat_cursor(monkeypatch):
    monkeypatch.setattr(flow, 'global_config', SimpleNamespace(a_memorix=SimpleNamespace(
        integration=SimpleNamespace(chat_summary_writeback_enabled=True))))
    service = flow.ChatSummaryWritebackService()
    first = SimpleNamespace(session_id='a')
    latest = SimpleNamespace(session_id='a')
    other = SimpleNamespace(session_id='b')
    await service.enqueue(first)
    await service.enqueue(latest)
    await service.enqueue(other)
    assert service._queue.qsize() == 2
    assert service._pending_messages['a'] is latest
    assert service._pending_messages['b'] is other

@pytest.mark.asyncio
async def test_summary_deadline_cancels_underlying_work_and_keeps_progress(monkeypatch):
    service = flow.ChatSummaryWritebackService()
    service._states['s'] = flow.ChatSummaryWritebackState()
    service._resolve_session_id = lambda m: 's'
    service._extract_message_timestamp = lambda m: 1
    service._count_messages_until_trigger = lambda **kwargs: 10
    service._message_threshold = lambda: 1
    service._context_length = lambda: 36
    service._extract_session_user_id = service._extract_session_group_id = lambda m: ''
    finished, stopped = [], asyncio.Event()

    async def resolve(feature, action, payload):
        if action == 'summary_prepare':
            return True, {'allow': True, 'timeout_seconds': 1}
        finished.append(payload)
        return True, {}

    async def slow(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolve)
    monkeypatch.setattr(flow.memory_service, 'ingest_summary', slow)
    await service._handle_message(SimpleNamespace())
    assert stopped.is_set()
    assert finished[0]['detail'] == 'summary_timeout'
    assert not finished[0]['success']
    assert service._states['s'].last_trigger_message_count == 0
