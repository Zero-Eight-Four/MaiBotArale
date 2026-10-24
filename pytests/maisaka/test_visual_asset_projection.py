"""Missing media stays readable without attempting to hydrate absent assets."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.data_models.message_component_data_model import EmojiComponent, TextComponent
from src.maisaka.reasoning_engine import MaisakaReasoningEngine
import src.plugin_runtime.policy_bridge as bridge


@pytest.mark.asyncio
async def test_missing_emoji_becomes_description_without_binary_load(monkeypatch):
    async def policy(feature, action, payload):
        assert (feature, action) == ("context", "visual_assets")
        return True, {"indices": []}
    monkeypatch.setattr(bridge, "resolve_policy", policy)
    loader = AsyncMock(side_effect=ValueError("asset absent"))
    monkeypatch.setattr(EmojiComponent, "load_emoji_binary", loader)
    components = [EmojiComponent(binary_hash="a" * 64, content="笑着摆手")]
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = SimpleNamespace(session_id="test", log_prefix="[test]")
    await engine._hydrate_visual_components(components)
    assert isinstance(components[0], TextComponent)
    assert "笑着摆手" in components[0].text
    loader.assert_not_awaited()


@pytest.mark.asyncio
async def test_available_emoji_still_loads_normally(monkeypatch):
    async def policy(*args):
        return True, {"indices": [0]}
    monkeypatch.setattr(bridge, "resolve_policy", policy)
    loader = AsyncMock()
    monkeypatch.setattr(EmojiComponent, "load_emoji_binary", loader)
    components = [EmojiComponent(binary_hash="a" * 64, content="笑着摆手")]
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = SimpleNamespace(session_id="test", log_prefix="[test]")
    await engine._hydrate_visual_components(components)
    assert isinstance(components[0], EmojiComponent)
    loader.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_visual_selection_cannot_add_components(monkeypatch):
    from src.plugin_runtime.context_views import select_visual_assets
    async def policy(*args):
        return True, {"indices": [1]}
    monkeypatch.setattr(bridge, "resolve_policy", policy)
    with pytest.raises(ValueError, match="Invalid visual asset"):
        await select_visual_assets([EmojiComponent(binary_hash="a" * 64)])


@pytest.mark.asyncio
async def test_asset_plugin_uses_sdk_and_rechecks_restored_files(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path("plugins/Arale-context-policy/assets.py")
    spec = importlib.util.spec_from_file_location("asset_lookup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "plugins" / "policy" / "assets.py"))
    image = tmp_path / "data" / "emoji.gif"
    image.parent.mkdir()
    query = AsyncMock(return_value={"full_path": "data/emoji.gif", "no_file_flag": False})
    from maibot_sdk import PluginContext
    async def rpc(method, plugin_id, payload, **kwargs):
        assert method == "cap.call" and payload["capability"] == "database.query"
        args = payload["args"]
        return await query(args["model_name"], filters=args["filters"], single_result=args["single_result"])
    ctx = PluginContext("Arale-context-policy", rpc_call=rpc)
    assets = [{"hash": "a" * 64, "kind": "emoji"}]
    assert await module.available_assets(ctx, assets) == []
    image.write_bytes(b"original asset")
    assert await module.available_assets(ctx, assets) == [0]
    query.assert_awaited_with("Images", filters={"image_hash": "a" * 64, "image_type": "EMOJI"}, single_result=True)
