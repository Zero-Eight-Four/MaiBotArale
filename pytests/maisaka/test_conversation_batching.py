"""Plugin-owned conversation policy exercised through the native scheduling interfaces."""
from types import SimpleNamespace
from unittest.mock import Mock,AsyncMock
import pytest
from pytests.plugin_runtime.test_arale_phase2 import policies,row
from src.maisaka import runtime as runtime_module
from src.maisaka.turn_trigger import scheduler

@pytest.mark.asyncio
async def test_quiet_window_extends_for_normal_typing_gaps(policies,monkeypatch):
    clock=[0.0];waits=[]
    base={'session_id':'s','revision':1,'message':row(),'pending':[],'is_group':True}
    await policies.resolve('flow','received',{**base,'now':0})
    runtime=SimpleNamespace(_message_debounce_required=True,_message_debounce_seconds=3,_last_message_received_at=0,_running=True)
    runtime._clear_message_debounce_required=lambda:setattr(runtime,'_message_debounce_required',False)
    async def decide(action):
        return (await policies.resolve('flow',action,{**base,'now':clock[0],'last_received_at':runtime._last_message_received_at}))[1]
    runtime._conversation_policy=decide
    async def sleep(seconds):
        waits.append(seconds)
        if len(waits)==1:runtime._last_message_received_at=2
        clock[0]+=seconds
    monkeypatch.setattr(runtime_module.asyncio,'sleep',sleep)
    await runtime_module.MaisakaHeartFlowChatting._wait_for_message_quiet_period(runtime)
    assert waits==[3,2]

@pytest.mark.asyncio
async def test_busy_group_does_not_extend_past_eight_seconds(policies,monkeypatch):
    clock=[0];waits=[]
    base={'session_id':'s','revision':1,'message':row(),'pending':[],'is_group':True}
    await policies.resolve('flow','received',{**base,'now':0})
    runtime=SimpleNamespace(_message_debounce_required=True,_message_debounce_seconds=3,_last_message_received_at=0,_running=True)
    runtime._clear_message_debounce_required=lambda:setattr(runtime,'_message_debounce_required',False)
    async def decide(action):return (await policies.resolve('flow',action,{**base,'now':clock[0],'last_received_at':clock[0]}))[1]
    runtime._conversation_policy=decide
    async def sleep(seconds):waits.append(seconds);clock[0]+=seconds
    monkeypatch.setattr(runtime_module.asyncio,'sleep',sleep)
    await runtime_module.MaisakaHeartFlowChatting._wait_for_message_quiet_period(runtime)
    assert sum(waits)==8

@pytest.mark.asyncio
@pytest.mark.parametrize('urgent',[True,False])
async def test_wait_wakeup_preserves_receipt_and_single_enqueue(policies,monkeypatch,urgent):
    monkeypatch.setattr(scheduler.focus_mode_manager,'can_decide',lambda *a,**k:True)
    monkeypatch.setattr(scheduler.global_config.experimental,'planner_message_steering',False)
    runtime=SimpleNamespace(session_id='s',log_prefix='test',chat_stream=SimpleNamespace(is_group_session=True),
        _STATE_WAIT='wait',_STATE_RUNNING='running',_agent_state='wait',_message_turn_scheduled=False,
        _is_reply_frequency_silent=lambda:False,_get_pending_message_count=lambda:1,
        message_cache=[row(correction=urgent)],_last_processed_index=0,
        _cancel_wait_timeout_task=Mock(),_get_effective_reply_frequency=lambda:.8,
        _pending_wait_tool_call_id='wait1')
    def enqueue():runtime._message_turn_scheduled=True
    runtime._enqueue_message_turn=Mock(side_effect=enqueue)
    runtime._enter_running_state=lambda:setattr(runtime,'_agent_state','running')
    async def decision(action):
        return (await policies.resolve('flow',action,{'session_id':'s','pending':runtime.message_cache,'target':row(),'is_group':True}))[1]
    runtime._conversation_policy=decision
    manager=scheduler.MessageTurnScheduler(runtime)
    await manager.schedule_message_turn()
    assert runtime._enqueue_message_turn.call_count==int(urgent)
    assert runtime._pending_wait_tool_call_id=='wait1'

@pytest.mark.asyncio
async def test_consumption_advances_native_revision_and_not_plugin_state_in_core():
    runtime=SimpleNamespace(_last_processed_index=0,message_cache=['a','b'],_context_revision=0,
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
