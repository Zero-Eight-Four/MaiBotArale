"""Run the real startup method with held plugin startup and inert services."""
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
import asyncio
import sys

import pytest

import src.main as main


def install(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize('mode', ['ready', 'disabled', 'failed'])
def test_vector_background_work_waits_for_plugin_startup_completion(monkeypatch, mode):
    async def run():
        events=[];entered=asyncio.Event();release=asyncio.Event()
        async def start_plugins():
            events.append('plugins_loading');entered.set()
            if mode!='disabled':await release.wait()
            if mode=='failed':raise RuntimeError('fixture plugin startup failure')
            events.append('plugins_complete')
        manager=SimpleNamespace(start=start_plugins,supervisors=[] if mode=='disabled' else [SimpleNamespace(_runner_process=object())])
        install(monkeypatch,'src.plugin_runtime.integration',get_plugin_runtime_manager=lambda:manager)
        install(monkeypatch,'src.A_memorix.host_service',a_memorix_host_service=SimpleNamespace(
            register_config_reload_callback=Mock(),start=AsyncMock()))
        install(monkeypatch,'src.mcp_module.service',get_mcp_service=lambda:SimpleNamespace(register_config_reload_callback=Mock()))
        install(monkeypatch,'src.services.expression_vector_service',expression_vector_service=SimpleNamespace(
            start=lambda:events.append('embedding_policy_request')))
        install(monkeypatch,'src.emoji_system.emoji_manager',emoji_manager=SimpleNamespace(load_emojis_from_db=Mock()))
        install(monkeypatch,'src.chat.message_receive.chat_manager',chat_manager=SimpleNamespace(
            initialize=AsyncMock(),regularly_save_sessions=AsyncMock()))
        install(monkeypatch,'src.services.memory_flow_service',memory_automation_service=SimpleNamespace(start=AsyncMock()))
        install(monkeypatch,'src.agent.service',agent_service=SimpleNamespace(start=AsyncMock()))
        install(monkeypatch,'src.core.event_bus',event_bus=SimpleNamespace(emit=AsyncMock()))
        install(monkeypatch,'src.core.types',EventType=SimpleNamespace(ON_START='start'))
        install(monkeypatch,'src.chat.utils.statistic',OnlineTimeRecordTask=object,StatisticOutputTask=object)
        install(monkeypatch,'src.common.remote',TelemetryHeartBeatTask=object,TelemetryStatsUploadTask=object)
        monkeypatch.setattr(main,'config_manager',SimpleNamespace(start_file_watcher=AsyncMock()))
        monkeypatch.setattr(main,'prompt_manager',SimpleNamespace(load_prompts=Mock()))
        monkeypatch.setattr(main,'async_task_manager',SimpleNamespace(add_task=AsyncMock()))
        monkeypatch.setattr(main,'diagnostic_stage',lambda *a,**kw:nullcontext())
        monkeypatch.setattr(main.MainSystem,'_start_webui_server',Mock())
        task=asyncio.create_task(main.MainSystem()._init_components())
        await entered.wait()
        started_too_early='embedding_policy_request' in events
        release.set()
        if mode=='failed':
            with pytest.raises(RuntimeError,match='fixture plugin startup failure'):await task
            assert 'embedding_policy_request' not in events
        else:
            await task
            assert not started_too_early, events
            assert events.index('plugins_complete')<events.index('embedding_policy_request')
    asyncio.run(run())
