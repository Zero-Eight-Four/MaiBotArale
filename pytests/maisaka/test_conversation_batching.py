"""Plugin-owned conversation policy exercised through the native scheduling interfaces."""
from types import SimpleNamespace
from unittest.mock import Mock,AsyncMock
import pytest
from pytests.plugin_runtime.test_arale_phase2 import policies,row
from src.chat.debounce import MessageDebouncer
from src.maisaka import runtime as runtime_module
from src.maisaka.turn_trigger import scheduler

@pytest.mark.asyncio
async def test_quiet_window_extends_for_normal_typing_gaps(monkeypatch):
    clock=[0.0];waits=[]
    debouncer=MessageDebouncer()
    debouncer.observe(author='owner', is_bare_image=False, is_pure_emoji=False, at_bot=False, received_at=0.0, wall_time=0.0, now=0.0, mode='natural')
    runtime=SimpleNamespace(_fixed_message_debouncer=debouncer,_running=True,_is_message_debounce_enabled=lambda:True)
    runtime._get_message_quiet_period_remaining_seconds=lambda:runtime_module.MaisakaHeartFlowChatting._get_message_quiet_period_remaining_seconds(runtime)
    async def wait_for(coro, timeout):
        coro.close()
        waits.append(timeout);clock[0]+=timeout
        if len(waits)==2:
            debouncer.observe(author='owner', is_bare_image=False, is_pure_emoji=False, at_bot=False, received_at=clock[0], wall_time=clock[0], now=clock[0], mode='natural')
        raise TimeoutError()
    monkeypatch.setattr(runtime_module.asyncio,'wait_for',wait_for)
    monkeypatch.setattr(runtime_module.time,'monotonic',lambda:clock[0])
    await runtime_module.MaisakaHeartFlowChatting._wait_for_message_quiet_period(runtime)
    assert round(clock[0], 3)==3.0

@pytest.mark.asyncio
async def test_busy_group_does_not_extend_past_eight_seconds(monkeypatch):
    clock=[0.0];waits=[]
    debouncer=MessageDebouncer()
    debouncer.observe(author='owner', is_bare_image=False, is_pure_emoji=False, at_bot=False, received_at=0.0, wall_time=0.0, now=0.0, mode='steady')
    runtime=SimpleNamespace(_fixed_message_debouncer=debouncer,_running=True,_is_message_debounce_enabled=lambda:True)
    runtime._get_message_quiet_period_remaining_seconds=lambda:runtime_module.MaisakaHeartFlowChatting._get_message_quiet_period_remaining_seconds(runtime)
    async def wait_for(coro, timeout):
        coro.close()
        waits.append(timeout);clock[0]+=timeout
        debouncer.observe(author=f'user-{len(waits)}', is_bare_image=False, is_pure_emoji=False, at_bot=False, received_at=clock[0], wall_time=clock[0], now=clock[0], mode='steady')
        raise TimeoutError()
    monkeypatch.setattr(runtime_module.asyncio,'wait_for',wait_for)
    monkeypatch.setattr(runtime_module.time,'monotonic',lambda:clock[0])
    await runtime_module.MaisakaHeartFlowChatting._wait_for_message_quiet_period(runtime)
    assert round(clock[0], 3)==8.0

@pytest.mark.asyncio
@pytest.mark.parametrize('urgent',[True,False])
async def test_wait_wakeup_preserves_receipt_and_single_enqueue(policies,monkeypatch,urgent):
    monkeypatch.setattr(scheduler.global_config.chat.reply_timing,'planner_message_steering',False)
    runtime=SimpleNamespace(session_id='s',log_prefix='test',chat_stream=SimpleNamespace(is_group_session=True),
        _STATE_WAIT='wait',_STATE_RUNNING='running',_agent_state='wait',_message_turn_scheduled=False,
        _is_reply_frequency_silent=lambda:False,_get_pending_message_count=lambda:1,
        _get_message_quiet_period_remaining_seconds=lambda:0.0,
        _get_message_turn_idle_backoff_remaining_seconds=lambda:0.0,
        _has_forced_turn_trigger=lambda:False,
        _idle_backoff=SimpleNamespace(get_delay_seconds=lambda count:0.0),
        message_cache=[row(correction=urgent)],_last_processed_index=0,
        _cancel_wait_timeout_task=Mock(),_get_effective_reply_frequency=lambda:.8,
        _pending_wait_tool_call_id='wait1',_update_stage_status=Mock(),_defer_message_turn_check=Mock())
    def enqueue():runtime._message_turn_scheduled=True
    runtime._enqueue_message_turn=Mock(side_effect=enqueue)
    runtime._enter_running_state=lambda:setattr(runtime,'_agent_state','running')
    async def decision(action):
        return (await policies.resolve('flow',action,{'session_id':'s','pending':runtime.message_cache,'target':row(),'is_group':True}))[1]
    runtime._conversation_policy=decision
    manager=scheduler.MessageTurnScheduler(runtime)
    manager._dynamic_reply_gate.evaluate_extension=AsyncMock(return_value=None)
    manager._dynamic_reply_gate.evaluate=Mock(return_value=scheduler.TurnGateResult(decision='trigger',detail='ok'))
    await manager.schedule_message_turn()
    assert runtime._enqueue_message_turn.call_count==int(urgent)
    assert runtime._pending_wait_tool_call_id=='wait1'

@pytest.mark.asyncio
async def test_consumption_advances_native_revision_and_not_plugin_state_in_core():
    runtime=SimpleNamespace(_last_processed_index=0,message_cache=['a','b'],_context_revision=0,
        _fixed_message_debouncer=SimpleNamespace(consume=Mock()),
        _conversation_policy=AsyncMock(),session_id='s')
    result=await runtime_module.MaisakaHeartFlowChatting._collect_pending_messages(runtime)
    assert result==['a','b'] and runtime._context_revision==1 and runtime._last_processed_index==2
    runtime._conversation_policy.assert_awaited_once_with('consumed')

@pytest.mark.asyncio
async def test_proactive_reminder_is_new_reply_opportunity(policies):
    base={'session_id':'s','revision':1,'pending':[],'target':row()}
    await policies.resolve('flow','sent',{**base,'parameters':{'event_id':'first'}})
    assert (await policies.resolve('flow','prepare_reply',base))[1]['decision']=='skip'
    await policies.resolve('flow','proactive',base)
    assert (await policies.resolve('flow','prepare_reply',base))[1]['decision']=='allow'
