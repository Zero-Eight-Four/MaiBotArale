from datetime import datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.services import memory_flow_service as module
from src.services.image_writeback_journal import ImageWritebackJournal


@pytest.fixture(autouse=True)
def memory_policy(monkeypatch,tmp_path):
    from pytests.plugin_runtime.test_arale_migration import plugin
    from src.plugin_runtime import policy_bridge
    p,_=plugin('Arale-memory-policy',tmp_path)
    async def resolve(feature,action,payload):
        out=await p.accept(feature,action,payload)
        return True,out['modified_kwargs']['result']
    monkeypatch.setattr(policy_bridge,'resolve_policy',resolve)


def picture(*, animated=False):
    stream = BytesIO()
    first = Image.new('RGB', (3, 3), 'red')
    if animated:
        first.save(stream, format='GIF', save_all=True, append_images=[Image.new('RGB', (3, 3), 'blue')], duration=100)
    else:
        first.save(stream, format='PNG')
    return SimpleNamespace(type='image', binary_data=stream.getvalue(), binary_hash='hash', content='')


def message(components):
    return SimpleNamespace(raw_message=SimpleNamespace(components=components), session_id='test-chat', message_id='test-message', timestamp=datetime.now())


@pytest.mark.asyncio
async def test_animated_only_job_completes_without_ingest(tmp_path, monkeypatch):
    image = picture(animated=True)
    source = message([image])
    service = module.ImageMemoryWritebackService(tmp_path / 'jobs.sqlite3')
    service._journal = ImageWritebackJournal(service._journal_path)
    service._journal.enqueue(source.session_id, source.message_id)
    monkeypatch.setattr(service, '_enabled', lambda: True)
    monkeypatch.setattr(module, 'find_messages', lambda **kwargs: [source])
    ingest = AsyncMock()
    monkeypatch.setattr(module.memory_service, 'image_memory', ingest)
    complete = service._journal.complete

    def complete_and_stop(chat_id, message_id):
        complete(chat_id, message_id)
        service._stopping = True

    monkeypatch.setattr(service._journal, 'complete', complete_and_stop)
    try:
        await service._worker_loop()
        job = service._journal.list_jobs('', 10, 0)['items'][0]
        assert job['status'] == 'done'
        assert job['attempts'] == 0
        ingest.assert_not_awaited()
        assert image.binary_data == b''
    finally:
        service._journal.close()


@pytest.mark.asyncio
async def test_mixed_message_keeps_static_image(monkeypatch):
    from src.chat.image_system.image_manager import image_manager
    animated, static = picture(animated=True), picture()
    ingest = AsyncMock(return_value={'success': True, 'content_hash': 'hash'})
    monkeypatch.setattr(module.memory_service, 'image_memory', ingest)
    monkeypatch.setattr(image_manager, 'get_cached_image_description', lambda _: None)
    assert await module.ImageMemoryWritebackService()._handle_message(message([animated, static]))
    assert ingest.await_count == 1
    assert ingest.call_args.kwargs['component_path'] == '1'
    assert animated.binary_data == static.binary_data == b''


@pytest.mark.asyncio
async def test_static_ingest_failure_still_raises(monkeypatch):
    image = picture()
    monkeypatch.setattr(module.memory_service, 'image_memory', AsyncMock(return_value={'success': False, 'error': 'upstream unavailable'}))
    with pytest.raises(ExceptionGroup) as failure:
        await module.ImageMemoryWritebackService()._handle_message(message([image]))
    assert 'upstream unavailable' in str(failure.value.exceptions[0])
    assert image.binary_data == b''


@pytest.mark.asyncio
async def test_invalid_image_is_not_silently_skipped(monkeypatch):
    image = picture()
    image.binary_data = b'invalid image bytes'
    ingest = AsyncMock()
    monkeypatch.setattr(module.memory_service, 'image_memory', ingest)
    with pytest.raises(ExceptionGroup):
        await module.ImageMemoryWritebackService()._handle_message(message([image]))
    ingest.assert_not_awaited()
    assert image.binary_data == b''
