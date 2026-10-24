import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.services import memory_flow_service as flow
from src.services.memory_service import MemoryWriteResult
from src.plugin_runtime import policy_bridge


@pytest.mark.asyncio
async def test_real_summary_worker_does_not_replay_terminated_history_or_advance_failed_progress(monkeypatch):
    service = flow.ChatSummaryWritebackService()
    count = [100]
    service._states['s'] = flow.ChatSummaryWritebackState(last_trigger_message_count=0, last_trigger_time=0)
    service._resolve_session_id = lambda m: 's'
    service._extract_message_timestamp = lambda m: 1
    service._count_messages_until_trigger = lambda **kwargs: count[0]
    service._message_threshold = lambda: 1
    service._context_length = lambda: 36
    service._extract_session_user_id = service._extract_session_group_id = lambda m: ''
    blocked_until_count = [0]

    async def resolve(feature, action, payload):
        if action == 'summary_prepare':
            allow = payload['message_count'] >= blocked_until_count[0]
            return True, {'allow': allow, 'timeout_seconds': 90}
        if not payload['success']:
            blocked_until_count[0] = payload['message_count'] + payload['context_length']
        else:
            blocked_until_count[0] = 0
        return True, {}

    monkeypatch.setattr(policy_bridge, 'resolve_policy', resolve)
    ingest = AsyncMock(return_value=MemoryWriteResult(False, detail='LLM failed: empty_choices'))
    monkeypatch.setattr(flow.memory_service, 'ingest_summary', ingest)
    await service._handle_message(SimpleNamespace())
    assert service._states['s'].last_trigger_message_count == 0
    count[0] += 35
    await service._handle_message(SimpleNamespace())
    assert ingest.await_count == 1  # timer alone never resubmits a terminated window
    count[0] += 1
    ingest.return_value = MemoryWriteResult(True)
    await service._handle_message(SimpleNamespace())
    assert ingest.await_count == 2
    assert service._states['s'].last_trigger_message_count == 136
    assert blocked_until_count[0] == 0

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
