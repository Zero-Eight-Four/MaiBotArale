"""Merged plugin registration, independent switches and reverse-search persistence."""
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import importlib
import json

from PIL import Image
import pytest

from pytests.plugin_runtime.test_arale_migration import load
from src.plugin_runtime.runner.manifest_validator import PluginManifest


def configured(name, tmp_path):
    module, _ = load(name)
    plugin = module.create_plugin()
    plugin.set_plugin_config(plugin.get_default_config())
    ctx = SimpleNamespace(logger=Mock(), config=SimpleNamespace(get=AsyncMock(return_value="")),
        paths=SimpleNamespace(data_dir=tmp_path / name, runtime_dir=tmp_path / 'runtime' / name),
        db=SimpleNamespace(query=AsyncMock(return_value=[])),
        llm=SimpleNamespace(generate=AsyncMock(return_value={'success': True, 'response': '{"suitable":true,"reason":"符合"}'})),
        message=SimpleNamespace(get_by_id=AsyncMock()))
    plugin._set_context(ctx)
    return plugin, module


@pytest.mark.parametrize('name', ['Arale-policy', 'Arale-web', 'Arale-learning'])
def test_real_factory_registers_stable_unique_components_before_load(name, tmp_path):
    p, _ = configured(name, tmp_path)
    manifest = json.loads((Path(__file__).resolve().parents[2] / 'plugins' / name / '_manifest.json').read_text())
    assert PluginManifest.model_validate(manifest).id == name
    components = p.get_components()
    assert components == p.get_components()
    assert len(components) == len({(c['type'], c['name']) for c in components})
    assert all(callable(getattr(p, c['metadata']['handler_name'])) for c in components)
    if name == 'Arale-web':
        assert len([c for c in components if c['type'] == 'TOOL' and c['name'] == 'reverse_image_search']) == 1
    if name == 'Arale-policy':
        assert not any(c['type'] == 'TOOL' and c['name'] == 'reverse_image_search' for c in components)


@pytest.mark.asyncio
async def test_learning_independent_switches_and_reload(tmp_path):
    p, _ = configured('Arale-learning', tmp_path)
    await p.on_load()
    declarations = p.get_components()
    selector = next(c for c in declarations if c['name'] == 'expression__selector_prompt')
    result = await getattr(p, selector['metadata']['handler_name'])(feature='expression', action='selection_prompt',
        payload={'candidates': [{'id': 1, 'situation': '打招呼', 'style': '早呀'}], 'max_selected': 2, 'persona': '简洁', 'style': '自然'})
    assert 'selected_ids' in result['modified_kwargs']['result']['prompt']
    config = p.get_plugin_config_data()
    config['expression']['plugin']['enabled'] = False
    p.set_plugin_config(config)
    await p.on_config_update('self', config, 'test')
    assert not p.modules['expression'].config.enabled
    assert p.modules['jargon'].config.enabled
    result = await getattr(p, selector['metadata']['handler_name'])(feature='expression', action='selection_prompt', payload={})
    assert result == {'action': 'continue'}
    await p.on_unload()
    await p.on_load()
    assert p.get_components() == declarations
    await p.on_unload()


@pytest.mark.asyncio
async def test_reverse_search_cache_survives_owner_reload_and_disabled_tool_never_reads_image(tmp_path, monkeypatch):
    p, module = configured('Arale-web', tmp_path)
    # Exercise the new wrapper and original search cache, with network/media lifecycle isolated.
    for name in ['on_load', 'on_unload', 'on_config_update']:
        monkeypatch.setattr(module.AraleWeb, name, AsyncMock())
    await p.on_load()
    search_module = importlib.import_module(module.__name__ + '.reverse_image.search')
    data = BytesIO()
    Image.new('RGB', (32, 32), 'blue').save(data, format='PNG')
    raw = data.getvalue()
    key = search_module.cache_key('chat', raw, 'anime', p.reverse_image.config.search)
    p.reverse_image._cache.put(key, {'success': True, 'provider': 'test', 'candidates': [], 'verified': False}, 600)
    await p.on_unload()
    new, _ = configured('Arale-web', tmp_path)
    await new.on_load()
    new.ctx.message.get_by_id.return_value = {'session_id': 'chat',
        'raw_message': [{'type': 'image', 'binary_data_base64': b64encode(raw).decode()}]}
    component = next(c for c in new.get_components() if c['type'] == 'TOOL' and c['name'] == 'reverse_image_search')
    handler = getattr(new, component['metadata']['handler_name'])
    result = await handler(msg_id='message', stream_id='chat')
    assert json.loads(result['content'])['cached'] is True
    cfg = new.get_plugin_config_data()
    cfg['reverse_image']['plugin']['enabled'] = False
    new.set_plugin_config(cfg)
    await new.on_config_update('self', cfg, 'test')
    new.ctx.message.get_by_id.reset_mock()
    assert not (await handler(msg_id='message', stream_id='chat'))['success']
    new.ctx.message.get_by_id.assert_not_awaited()
    component = next(c for c in new.get_components() if c['type'] == 'TOOL' and c['name'] == 'reverse_image_search')
    assert component['metadata']['enabled'] is False
    await new.on_unload()
