from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from datetime import datetime
import importlib.util
import json
import sys
import tomllib

import pytest

from src.plugin_runtime.runner.manifest_validator import PluginManifest
from src.plugin_runtime.component_query import ComponentQueryService
from src.core.tooling import ToolExecutionContext, ToolInvocation

ROOT=Path(__file__).resolve().parents[2]
IDS=['Arale-asr','Arale-policy','Arale-conversation-flow','Arale-learning',
     'Arale-file-reader','Arale-history-reader']
MODULES={**{f'Arale-{name}-policy':('Arale-policy',name)
            for name in ['chat','context','emoji','memory','model','runtime','web']},
         'Arale-reply-guard':('Arale-conversation-flow','guard'),
         'Arale-group-context':('Arale-history-reader','group'),
         'Arale-vision':('Arale-policy','vision'),
         'Arale-reverse-image':('Arale-web','reverse_image'),
         'Arale-expression-curator':('Arale-learning','expression'),
         'Arale-jargon-curator':('Arale-learning','jargon')}


def load(ident):
    base=ROOT/'plugins'/ident
    if ident in MODULES:
        owner,section=MODULES[ident]
        base=ROOT/'plugins'/owner/section
    name='arale_test_'+ident.replace('-','_')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,base/'plugin.py',submodule_search_locations=[str(base)])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name],base


def plugin(ident,tmp_path):
    mod,base=load(ident);p=mod.create_plugin()
    if ident in MODULES:
        owner,section=MODULES[ident]
        config=tomllib.loads((ROOT/'plugins'/owner/'config.toml').read_text())[section]
    else:
        config=tomllib.loads((base/'config.toml').read_text())
    if ident == 'Arale-policy':
        config['chat'].update(receive_all_chats=False,group_enabled=False,private_all_enabled=False)
    # Legacy access tests use explicit fixtures, independent of the live user's settings.
    if ident == 'Arale-chat-policy':config.update(receive_all_chats=False,group_enabled=False,private_all_enabled=False)
    if ident == 'Arale-group-context':config['groups']=['959142150','664646220']
    p.set_plugin_config(config)
    p._set_context(SimpleNamespace(logger=Mock(),config=SimpleNamespace(get=AsyncMock(return_value="")),paths=SimpleNamespace(data_dir=tmp_path,runtime_dir=tmp_path),
        llm=SimpleNamespace(generate=AsyncMock()),chat=SimpleNamespace(get_all_streams=AsyncMock(), get_private_streams=AsyncMock(return_value=[])),
        db=SimpleNamespace(query=AsyncMock(return_value=[])),
        message=SimpleNamespace(get_by_time_in_chat=AsyncMock()),
        maisaka=SimpleNamespace(context=SimpleNamespace(append=AsyncMock(return_value={'success':True})))))
    return p,mod


@pytest.mark.parametrize('ident',IDS)
def test_manifest_and_config_load(ident,tmp_path):
    _,base=load(ident);manifest=PluginManifest.model_validate(json.loads((base/'_manifest.json').read_text()))
    p,_=plugin(ident,tmp_path)
    assert manifest.id==ident and p.config is not None
    from src.plugin_runtime.runner.plugin_loader import PluginLoader
    PluginLoader._validate_sdk_plugin_contract(ident,p)



def test_host_context_overrides_forged_model_parameters():
    entry=SimpleNamespace(invoke_method='plugin.invoke_tool',plugin_id='Arale-group-context')
    context=ToolExecutionContext(session_id='real',stream_id='real',user_id='2730843725',platform='qq',is_group_chat=False,
        metadata={'logical_turn_id':'turn','account_id':'bot'})
    call=ToolInvocation(tool_name='recall_group_context',arguments={'_host_context':{'stream_id':'forged'},'stream_id':'forged'})
    out=ComponentQueryService._build_tool_invocation_payload(entry,call,context)
    assert out['_host_context']['stream_id']=='real'
    assert out['_host_context']['account_id']=='bot'


@pytest.mark.asyncio
async def test_web_budget_cannot_be_reset_after_limit(tmp_path):
    p,_=plugin('Arale-web-policy',tmp_path)
    base={'session_id':'s','cycle_id':'cycle'}
    for _ in range(p.config.max_consecutive_reads):
        out=await p.resolve('tools','before_execute',{**base,'name':'fetch_url'})
        assert not out['modified_kwargs']['result']['blocked']
    await p.resolve('tools','before_execute',{**base,'name':'query_memory'})
    out=await p.resolve('tools','before_execute',{**base,'name':'web_search'})
    assert out['modified_kwargs']['result']['blocked']
    out=await p.resolve('tools','prepare',{**base,'definitions':[{'name':'reply'},{'name':'fetch_url'}]})
    assert out['modified_kwargs']['result']['definitions']==[{'name':'reply'}]
    out=await p.resolve('tools','before_execute',{'session_id':'s','cycle_id':'new','name':'fetch_url'})
    assert not out['modified_kwargs']['result']['blocked']


@pytest.mark.asyncio
async def test_expression_rejects_string_booleans(tmp_path):
    p,_=plugin('Arale-expression-curator',tmp_path)
    p.ctx.llm.generate.return_value={'success':True,'content':'{"suitable":"false","reason":"no"}'}
    out=await p.review('expression','review',{'situation':'拒绝','style':'拉倒吧'})
    result=out['modified_kwargs']['result']
    assert not result['suitable'] and result['error']
    p.ctx.llm.generate.return_value={'success':True,'content':'{"suitable":true,"reason":"符合原文场景"}'}
    out=await p.review('expression','review',{'situation':'拒绝','style':'拉倒吧','review_context':'原文'})
    assert out['modified_kwargs']['result']['suitable']


@pytest.mark.asyncio
async def test_group_reference_survives_plugin_reload(tmp_path):
    p,mod=plugin('Arale-group-context',tmp_path);await p.on_load()
    rows=[{'message_id':'m','session_id':'g','group_id':'959142150','group_name':'群名','timestamp':datetime.now(),
           'user_id':'2730843725','user_nickname':'owner','processed_plain_text':'游戏角色','reply_to':None}]
    p._service.query_rows=AsyncMock(return_value=rows);p._service.query_candidate_rows=AsyncMock(return_value=rows)
    context={'stream_id':'s','account_id':'bot','user_id':'2730843725','platform':'qq','logical_turn_id':'turn','is_group_chat':False}
    found=await p.recall(action='lookup',query='游戏',_host_context=context)
    assert found['success'];ref=json.loads(found['content'])['candidates'][0]['candidate_ref']
    new,_=plugin('Arale-group-context',tmp_path);await new.on_load();new._service.query_candidate_rows=AsyncMock(return_value=rows)
    result=await new.recall(action='read',candidate_ref=ref,_host_context=context)
    assert result['success'] and json.loads(result['content'])['messages'][0]['text']=='游戏角色'
    bad=await new.recall(action='read',candidate_ref=ref,_host_context={**context,'stream_id':'other'})
    assert not bad['success']
    bad=await new.recall(action='read',candidate_ref=ref,_host_context={**context,'is_group_chat':True})
    assert not bad['success']
    assert not (await new.recall(action='read',candidate_ref=ref))['success']
    injected=await new.inject_background(reply_tool_args={'_host_context':context})
    assert '游戏角色' in injected['modified_kwargs']['extra_prompt']
    stale=await new.inject_background(reply_tool_args={'_host_context':{**context,'logical_turn_id':'new'}})
    assert 'modified_kwargs' not in stale


@pytest.mark.asyncio
async def test_conversation_classifies_requests_and_uses_single_window(tmp_path):
    p,_=plugin('Arale-conversation-flow',tmp_path)
    p.config.plugin.legacy_debounce_enabled = False
    p.config.plugin.debounce_seconds = 3.0
    p.config.plugin.max_wait_seconds = 8.0
    out=await p.conversation_policy('conversation','classify',{'text':'不对，我说的是这个'})
    data=out['modified_kwargs']['result']
    assert data['request_or_correction'] and data['debounce_seconds']==3 and data['max_wait_seconds']==8
    assert not p.config.plugin.legacy_debounce_enabled


@pytest.mark.asyncio
async def test_chat_restrictions_and_priority_account(tmp_path):
    p,_=plugin('Arale-chat-policy',tmp_path)
    assert not p.allowed(True,'2730843725') and not p.allowed(False,'other') and p.allowed(False,'2730843725')
    out=await p.classify('chat','classify',{'platform':'qq','user_id':'2730843725','group_id':'959142150','is_group_chat':True})
    assert out['modified_kwargs']['result']['bypass']
    out=await p.classify('chat','classify',{'platform':'qq','user_id':'2730843725','group_id':'664646220','is_group_chat':True})
    assert not out['modified_kwargs']['result']['bypass']


@pytest.mark.asyncio
@pytest.mark.parametrize('mentioned,at', [(False, False), (True, False), (False, True), (True, True)])
async def test_explicit_group_triggers_do_not_enable_owner_bypass(tmp_path, mentioned, at):
    p,_=plugin('Arale-chat-policy',tmp_path)
    p.config.priority_disabled_groups = ['qq:664646220']
    p.config.mentioned_reply_groups = ['qq:664646220'] if mentioned else []
    p.config.at_reply_groups = ['qq:664646220'] if at else []
    for user in ['2730843725', 'other']:
        out = await p.classify('chat', 'classify', {'platform':'qq', 'user_id':user,
                              'group_id':'664646220', 'is_group_chat':True})
        policy = out['modified_kwargs']['result']
        assert policy == {'bypass':False, 'force_disabled':not (mentioned or at),
                          'mentioned_bot_reply':mentioned, 'inevitable_at_reply':at}


@pytest.mark.asyncio
async def test_emoji_prompt_preserves_context_snapshot_protocol(tmp_path):
    from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType
    from src.plugin_runtime.hook_payloads import serialize_prompt_items,deserialize_prompt_items
    p,_=plugin('Arale-emoji-policy',tmp_path)
    b=ContextItemBuilder();b.set_role(RoleType.System);b.add_text_content('系统');original=[b.build()]
    out=await p.hints(items=serialize_prompt_items(original))
    restored=deserialize_prompt_items(out['modified_kwargs']['items'],original_items=original)
    assert len(restored[0].parts)==2
