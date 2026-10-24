from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from pytests.plugin_runtime.test_arale_migration import load


@pytest.mark.asyncio
@pytest.mark.parametrize("name",["Arale-policy","Arale-learning","Arale-conversation-flow","Arale-history-reader","Arale-web"])
async def test_parent_disable_survives_submodule_reload(name,monkeypatch):
    module,_=load(name);p=module.create_plugin();config=p.config_model().model_dump(mode="python");config["plugin"]["enabled"]=False;p.set_plugin_config(config)
    p._ctx=SimpleNamespace(logger=Mock())
    if name in {"Arale-policy","Arale-learning"}:
        children=p.modules
    else:
        key={"Arale-conversation-flow":"guard","Arale-history-reader":"group","Arale-web":"reverse_image"}[name]
        children={key:getattr(p,key)}
        monkeypatch.setattr(type(p).__mro__[1],"on_config_update",AsyncMock())
    for child in children.values():
        child.config.plugin.enabled=True
        child.on_config_update=AsyncMock()
    await p.on_config_update("bot",{},"new")
    for child in children.values():
        assert child.config.plugin.enabled is False
        assert child.on_config_update.await_args.args[1]["plugin"]["enabled"] is False


@pytest.mark.asyncio
async def test_can_send_reuses_one_lookup_without_changing_group_and_private_permissions(tmp_path):
    from pytests.plugin_runtime.test_arale_migration import plugin
    p,_=plugin("Arale-chat-policy",tmp_path)
    p.config.group_enabled=True;p.config.group_send_whitelist=["allowed"]
    p.ctx.chat.get_all_streams.return_value=[
        {"session_id":"g","group_id":"allowed","account_id":"bot","is_group_session":True},
        {"session_id":"blocked","group_id":"other","account_id":"bot","is_group_session":True},
        {"session_id":"p","user_id":"2730843725","account_id":"bot"},
        {"session_id":"stranger","user_id":"unlisted","account_id":"bot"}]
    for sid,allowed in [("g",True),("blocked",False),("p",True),("stranger",False)]:
        assert (await p.can_send(stream_id=sid,account_id="bot"))["allowed"] is allowed
    p.ctx.chat.get_all_streams.assert_awaited_once()
