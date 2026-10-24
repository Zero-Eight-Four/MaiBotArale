from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
import asyncio,json
import pytest
import pytest_asyncio
from pytests.plugin_runtime.arale_test_runtime import PolicyRuntime
from src.plugin_runtime import policy_bridge
from src.plugin_runtime.runner.plugin_loader import PluginLoader

@pytest_asyncio.fixture
async def policies(tmp_path,monkeypatch):
    runtime=PolicyRuntime(tmp_path);await runtime.start()
    monkeypatch.setattr(policy_bridge,'resolve_policy',runtime.resolve)
    return runtime

@pytest.mark.parametrize('ident',['Arale-context-policy','Arale-model-policy'])
def test_new_plugins_contract(tmp_path,ident):
    from pytests.plugin_runtime.test_arale_migration import plugin
    p,_=plugin(ident,tmp_path);PluginLoader._validate_sdk_plugin_contract(ident,p)

@pytest.mark.asyncio
async def test_all_policy_routes_are_unique(policies):
    from fnmatch import fnmatchcase
    for route,owner,fn in policies.routes:
        if '*' not in route:assert sum(fnmatchcase(route,p) for p,_,_ in policies.routes)==1


@pytest.mark.asyncio
async def test_capacity_is_native_and_selection_is_checked(policies):
    handled, capacity = await policies.resolve('emoji', 'capacity', {})
    assert not handled and capacity is None
    handled, selection = await policies.resolve('emoji', 'selection_result', {
        'candidate_count': 2, 'content': '{"emoji_index":2,"reason":"合适"}'})
    assert handled and not selection['skip']
    assert json.loads(selection['content'])['emoji_index'] == 2
    handled, invalid = await policies.resolve('emoji', 'selection_result', {
        'candidate_count': 2, 'content': '{"emoji_index":99}'})
    assert handled and invalid['skip']

@pytest.mark.asyncio
@pytest.mark.parametrize('locale',['zh-CN','en-US','ja-JP'])
@pytest.mark.parametrize('name',['maisaka_chat','maisaka_chat_focus','maisaka_replyer','learn_style','learn_jargon','jargon_inference_with_context','jargon_compare_inference'])
async def test_template_ownership(policies,name,locale):
    handled,result=await policies.resolve('prompt',name,{'locale':locale})
    assert handled and result['template']

@pytest.mark.asyncio
async def test_reply_schema_preserves_official_required_reference_and_quote_policy(policies,monkeypatch):
    from src.maisaka.builtin_tool.reply import get_tool_spec
    from src.plugin_runtime.tool_specs import extend_tool_spec
    spec=await extend_tool_spec(get_tool_spec())
    props=spec.parameters_schema['properties']
    assert spec.parameters_schema['required']==['msg_id','reply_reference']
    assert 'set_quote' not in props
    assert 'supplement_reason' in props


def row(user='owner',text='',at=False,correction=False,direct=False):
    return {'message_id':'m','user_id':user,'platform':'qq','is_self':False,'is_at':at,'is_mentioned':False,
            'traits':{'request_or_correction':correction,'direct_request':direct}}

@pytest.mark.asyncio
async def test_flow_batch_reply_and_correction(policies):
    base={'session_id':'s','now':0,'revision':1,'is_group':True,'pending':[],'target':row()}
    async def call(action,**kw):return (await policies.resolve('flow',action,{**base,**kw}))[1]
    assert (await call('prepare_reply'))['decision']=='allow'
    await call('sent',parameters={'event_id':'send1'})
    assert (await call('prepare_reply'))['decision']=='skip'
    assert (await call('prepare_reply',parameters={'supplement_reason':'important_correction'}))['decision']=='allow'
    assert (await call('prepare_reply',revision=2))['decision']=='allow'
    assert (await call('prepare_reply',pending=[row('other')]))['decision']=='skip'
    result=await call('prepare_reply',pending=[row(correction=True)])
    assert result['decision']=='replan' and not result['pause_execution']
    await call('prepare_reply',pending=[row()])
    assert (await call('prepare_reply',pending=[row()]))['pause_execution']

@pytest.mark.asyncio
async def test_flow_quiet_cap_wait_and_reset(policies):
    base={'session_id':'s','now':0,'revision':1,'is_group':True,'message':row(),'pending':[],'target':row()}
    async def call(action,**kw):return (await policies.resolve('flow',action,{**base,**kw}))[1]
    await call('received')
    assert (await call('quiet_window',now=2,last_received_at=2))['remaining']==3
    assert (await call('quiet_window',now=7,last_received_at=7))['remaining']==1
    assert (await call('quiet_window',now=8,last_received_at=8))['remaining']==0
    assert not (await call('wake_wait',pending=[row()]))['wake']
    assert (await call('wake_wait',pending=[row(correction=True)]))['wake']
    assert not (await call('wake_wait',pending=[row('other',correction=True)]))['wake']
    assert (await call('wake_wait',is_group=False,pending=[row()]))['wake']
    await call('consumed');await call('received',now=10)
    assert (await call('quiet_window',now=10,last_received_at=10))['remaining']==3

@pytest.mark.asyncio
async def test_priority_accounting_is_idempotent_and_separate(policies):
    p=policies.plugins['Arale-chat-policy']
    base={'session_id':'s','now':1,'frequency':.8,'probability':.35}
    m={'id':'m','new':True,'policy':{'bypass':True}}
    out=(await policies.resolve('chat','admission',{**base,'messages':[m]}))[1]
    assert out['decision']=='trigger'
    out=(await policies.resolve('chat','admission',{**base,'messages':[{**m,'new':False}]}))[1]
    assert out['decision']=='normal'
    payload={**base,'platform':'qq','user_id':'2730843725','is_group':True,'group_id':'959142150','event_id':'send1'}
    for _ in range(2):out=(await policies.resolve('chat','reply_account',payload))[1]
    assert not out['include_feedback'] and out['statistics']=={'expected':.35,'actual':1}
    out=(await policies.resolve('chat','reply_account',{**payload,'group_id':'664646220','event_id':'send2'}))[1]
    assert out['include_feedback']

@pytest.mark.asyncio
async def test_model_cooldown_persists_without_keys(policies,tmp_path):
    p=policies.plugins['Arale-model-policy']
    details={'error':{'details':[{'@type':'type.googleapis.com/google.rpc.RetryInfo','retryDelay':'60s'}]}}
    out=(await policies.resolve('model','embedding_error',{'credential_fingerprint':'opaque','details':details}))[1]
    assert out['retry_at']
    from pytests.plugin_runtime.test_arale_migration import plugin
    new,_=plugin('Arale-model-policy',p.ctx.paths.data_dir);await new.on_load()
    out=await new.resolve('model','before_embedding',{'credential_fingerprint':'opaque'})
    assert out['modified_kwargs']['result']['retry_at']
    assert 'opaque' in (p.ctx.paths.data_dir/'cooldowns.json').read_text()

@pytest.mark.asyncio
async def test_context_filters_only_real_recall_receipt(policies):
    from src.plugin_runtime.context_views import select_history
    items=[SimpleNamespace(source_kind='guided_reply',message_id='x',content='old'),
           SimpleNamespace(tool_name='recall_message',success=True,content='已撤回消息 x。'),
           SimpleNamespace(source_kind='user',message_id='u',content='已撤回消息 u。')]
    assert await select_history(items)==items[1:]

@pytest.mark.asyncio
async def test_jargon_finalize_strict_boolean_and_evidence(policies):
    p=policies.plugins['Arale-jargon-curator']
    evidence='本群把来回改颜色叫木鱼工程'
    good={'is_similar':False,'has_special_meaning':True,'confidence':'high','evidence_quote':evidence}
    for result,expected in [(good,True),({**good,'has_special_meaning':'true'},False),({**good,'evidence_quote':'不存在的证据'},False)]:
        out=await p.finalize(comparison_result=result,raw_content_list=[evidence],is_jargon=True)
        assert out['modified_kwargs']['is_jargon'] is expected

@pytest.mark.asyncio
async def test_rendered_prompt_uses_plugin_template(policies):
    from src.prompt.prompt_manager import prompt_manager
    prompt_manager.load_prompts()
    prompt=prompt_manager.get_prompt('learn_jargon');prompt.add_context('chat_history','sample')
    # 使用模板真实字段填充，而不是猜测字段名。
    from string import Formatter
    handled,override=await policies.resolve('prompt','learn_jargon',{'locale':'zh-CN'})
    for _,name,_,_ in Formatter().parse(override['template']):
        if name and name not in prompt.prompt_render_context:prompt.add_context(name,'sample')
    text=await prompt_manager.render_prompt(prompt)
    assert 'sample' in text


@pytest.mark.asyncio
async def test_planner_template_renders_with_current_official_interface(policies, monkeypatch):
    from src.maisaka.chat_loop_service import MaisakaChatLoopService
    from src.config.config import global_config
    service = object.__new__(MaisakaChatLoopService)
    service._get_chat_prompt_name = lambda: 'maisaka_chat'
    service._build_bot_nickname_notice = lambda *args, **kwargs: ''
    service.build_prompt_template_context = lambda **kwargs: {
        'bot_name': '阿拉蕾', 'bot_alias_notice': '', 'behavior_style': '简短自然', 'bot_nickname_notice': '',
        'group_chat_attention_block': '', 'query_memory_rule': '', 'subagent_rule': '后台子代理', 'emoji_send_rule': '每轮最多一张'}
    text = await service._render_extension_planner_prompt([])
    assert '阿拉蕾的行为风格：简短自然' in text
    assert 'Planner负责调度' in text
    assert '{bot_name}' not in text and '{behavior_style}' not in text
