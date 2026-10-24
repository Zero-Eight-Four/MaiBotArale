from types import SimpleNamespace
from unittest.mock import AsyncMock
import threading
import pytest
from src.chat.utils import utils
from src.plugin_runtime.host.supervisor import PluginRunnerSupervisor
from src.maisaka.monitor.event_store import _is_replayable_event
from src.webui.routers.websocket import unified


def test_explicit_zero_disables_plugin_auto_restart():
    supervisor = PluginRunnerSupervisor(plugin_dirs=[], max_restart_attempts=0)
    assert supervisor._max_restart_attempts == 0


def test_old_monitor_events_do_not_break_replay():
    assert not _is_replayable_event({'event': 'planner.progress', 'data': {}})
    assert _is_replayable_event({'event': 'planner.finalized', 'data': {'run_id': 'valid'}})
    assert _is_replayable_event({'event': 'message.sent', 'data': {}})


@pytest.mark.asyncio
async def test_failed_llm_splitter_preserves_reply(monkeypatch):
    monkeypatch.setattr(utils.global_config.response_post_process, 'enable_response_post_process', True)
    monkeypatch.setattr(utils.global_config.response_splitter, 'enable', True)
    monkeypatch.setattr(utils.global_config.response_splitter, 'mode', 'llm')
    monkeypatch.setattr(utils, 'split_text_with_llm', AsyncMock(side_effect=ValueError('rewritten output')))
    def process(sentences, _typo):
        assert sentences == [('原回复保持完整。', '')]
        return [utils.ProcessedResponseSegment('原回复保持完整。')]
    monkeypatch.setattr(utils, '_apply_response_post_processing', process)
    result = await utils.process_llm_response_segments_async('原回复保持完整。')
    assert [item.text for item in result] == ['原回复保持完整。']


@pytest.mark.asyncio
async def test_log_file_read_is_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    threads = []
    def read(*, limit):
        threads.append(threading.get_ident())
        return [{'message': 'test'}]
    manager = SimpleNamespace(subscribe=lambda *args, **kwargs: None, send_response=AsyncMock(), send_event=AsyncMock())
    monkeypatch.setattr(unified, 'websocket_manager', manager)
    monkeypatch.setattr(unified, 'load_recent_logs', read)
    await unified._handle_logs_subscribe('connection', 'request', {'replay': 10})
    assert threads and threads[0] != loop_thread
    assert manager.send_event.call_args.kwargs['data']['entries'] == [{'message': 'test'}]
