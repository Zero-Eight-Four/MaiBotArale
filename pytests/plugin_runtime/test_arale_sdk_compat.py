"""Real SDK result normalization and Hook snapshot compatibility."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from maibot_sdk.context import PluginContext
from pytests.plugin_runtime.test_arale_migration import plugin

def attach_sdk(p, responses):
    async def rpc(method, plugin_id, payload, timeout_ms=None):
        assert method == 'cap.call'
        return responses[payload['capability']]
    ctx = PluginContext('test', rpc_call=rpc)
    p._ctx.chat = ctx.chat
    p._ctx.message = ctx.message
    return ctx

@pytest.mark.asyncio
@pytest.mark.parametrize('user,group,account,allowed', [
    ('2730843725', '', 'bot', True), ('other', '', 'bot', False),
    ('2730843725', '959142150', 'bot', False), ('2730843725', '', 'otherbot', False),
])
async def test_permissions_use_normalized_streams(tmp_path,user,group,account,allowed):
    p,_=plugin('Arale-chat-policy',tmp_path)
    attach_sdk(p, {'chat.get_all_streams': {'success':True,'streams':[
        {'session_id':'opaque','user_id':user,'account_id':account,'group_id':group,'is_group_session':bool(group)}]}})
    message={'platform':'qq','account_id':'bot','session_id':'opaque','message_info':{}}
    assert (await p.outgoing(message=message))['action']==('continue' if allowed else 'abort')
    message['message_info']={'user_info':{'user_id':'bot'}}
    assert (await p.incoming(message=message))['action']==('continue' if allowed else 'abort')

@pytest.mark.asyncio
async def test_group_queries_use_sdk_lists_and_keep_scope(tmp_path):
    p,_=plugin('Arale-group-context',tmp_path)
    msg={'session_id':'g','account_id':'bot','message_id':'m','timestamp':datetime.now().isoformat(),
        'processed_plain_text':'游戏讨论','message_info':{'group_info':{'group_id':'959142150'},'user_info':{'user_id':'2730843725'}}}
    attach_sdk(p, {'chat.get_group_streams':{'success':True,'streams':[
        {'session_id':'g','account_id':'bot','group_id':'959142150'},
        {'session_id':'other','account_id':'otherbot','group_id':'959142150'}]},
        'message.get_by_time_in_chat':{'success':True,'messages':[msg]}})
    session=SimpleNamespace(platform='qq',account_id='bot')
    rows=await p._query_rows(session)
    assert len(rows)==1 and rows[0]['processed_plain_text']=='游戏讨论'
    rows=await p._query_candidate_rows(session,rows[0])
    assert len(rows)==1 and rows[0]['message_id']=='m'

@pytest.mark.asyncio
async def test_planner_hook_chain_keeps_all_host_fields(tmp_path):
    from src.llm_models.payload_content.context_item import ContextItemBuilder,RoleType
    from src.plugin_runtime.hook_payloads import serialize_prompt_items,deserialize_prompt_items
    b=ContextItemBuilder();b.set_role(RoleType.System);b.add_text_content('系统');original=[b.build()]
    kwargs={'items':serialize_prompt_items(original),'item_schema_version':1,'session_id':'opaque',
        'tool_definitions':[{'name':'reply'}],'selection_reason':'latest','selected_history_count':2}
    for ident,method in [('Arale-emoji-policy','hints'),('Arale-vision','inject_planner_rules'),('Arale-file-reader','inject_planner_rules')]:
        p,_=plugin(ident,tmp_path/ident)
        result=await getattr(p,method)(**kwargs)
        kwargs=result.get('modified_kwargs', kwargs)
        assert kwargs['session_id']=='opaque' and kwargs['tool_definitions']==[{'name':'reply'}]
        assert kwargs['selected_history_count']==2
    items=deserialize_prompt_items(kwargs['items'],item_schema_version=kwargs['item_schema_version'],original_items=original)
    assert len(items[0].parts)>=1

@pytest.mark.asyncio
async def test_asr_context_uses_sdk_message_list(tmp_path,monkeypatch):
    p,mod=plugin('Arale-asr',tmp_path)
    attach_sdk(p, {'message.get_by_time_in_chat':{'success':True,'messages':[{'processed_plain_text':'语音上下文'}]}})
    asset=tmp_path/'voice';asset.write_bytes(b'audio')
    monkeypatch.setattr(mod.pipeline,'get_voice_text',AsyncMock(return_value='识别结果'))
    result=await p.transcribe(feature='asr',action='transcribe',payload={'session_id':'s','asset_path':str(asset)})
    assert result['modified_kwargs']['result']=='识别结果'
    assert mod.pipeline._CONTEXTS['s']=='语音上下文'
