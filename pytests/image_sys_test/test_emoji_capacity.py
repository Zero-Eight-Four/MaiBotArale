import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.emoji_system import emoji_manager as module


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "count,replace,expected",
    [(99, False, 1), (100, True, 1), (100, False, 0), (101, True, 1)],
)
async def test_maintenance_scans_at_capacity_only_when_replacement_enabled(
    tmp_path, monkeypatch, count, replace, expected
):
    for name in ["a.png", "b.png"]:
        (tmp_path / name).write_bytes(b"image")
    config = SimpleNamespace(
        check_interval=10, steal_emoji=True, max_reg_num=100,
        do_replace=replace, max_emoji_size_mb=5,
    )
    monkeypatch.setattr(module, "global_config", SimpleNamespace(emoji=config))
    monkeypatch.setattr(module, "EMOJI_DIR", tmp_path)
    monkeypatch.setattr(module, "_ensure_directories", lambda: None)

    @contextmanager
    def db_session():
        yield SimpleNamespace(exec=lambda query: SimpleNamespace(all=lambda: []))

    monkeypatch.setattr(module, "get_db_session", db_session)
    monkeypatch.setattr(
        module.asyncio, "wait_for", AsyncMock(side_effect=[TimeoutError(), asyncio.CancelledError()])
    )
    manager = SimpleNamespace(
        _emoji_num=count,
        _maintenance_wakeup_event=SimpleNamespace(wait=Mock(return_value=None), clear=Mock()),
        register_emoji_by_filename=AsyncMock(return_value="registered"),
        check_emoji_file_integrity=Mock(),
    )
    with pytest.raises(asyncio.CancelledError):
        await module.EmojiManager.periodic_emoji_maintenance(manager)
    assert manager.register_emoji_by_filename.await_count == expected
    manager.check_emoji_file_integrity.assert_called_once()


@pytest.mark.asyncio
async def test_full_library_without_replacement_does_not_register_or_analyze(tmp_path, monkeypatch):
    path = tmp_path / "new.png"
    path.write_bytes(b"image")
    emoji = SimpleNamespace(
        full_path=path, file_hash="new", description="",
        calculate_hash_format=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(module, "MaiEmoji", lambda **kwargs: emoji)
    monkeypatch.setattr(module, "global_config", SimpleNamespace(emoji=SimpleNamespace(max_reg_num=100, do_replace=False)))

    @contextmanager
    def db_session():
        yield SimpleNamespace(exec=lambda query: SimpleNamespace(first=lambda: None))

    monkeypatch.setattr(module, "get_db_session_manual", db_session)
    manager = SimpleNamespace(
        _emoji_num=100, get_emoji_by_hash=Mock(return_value=None),
        build_emoji_description=AsyncMock(), register_emoji_to_db=Mock(),
    )
    assert await module.EmojiManager.register_emoji_by_filename(manager, path) == "skipped"
    manager.build_emoji_description.assert_not_awaited()
    manager.register_emoji_to_db.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure,cycles,expected', [('exception',1,1), ('failed',1,1), ('failed',2,2)])
async def test_official_maintenance_attempts_only_one_candidate_per_cycle(tmp_path, monkeypatch, failure, cycles, expected):
    for index in range(25):
        (tmp_path / f'{index}.png').write_bytes(b'image')
    cfg=SimpleNamespace(check_interval=10,steal_emoji=True,max_reg_num=100,do_replace=True,max_emoji_size_mb=5)
    monkeypatch.setattr(module,'global_config',SimpleNamespace(emoji=cfg))
    monkeypatch.setattr(module,'EMOJI_DIR',tmp_path)
    monkeypatch.setattr(module,'_ensure_directories',lambda:None)
    @contextmanager
    def db_session():
        yield SimpleNamespace(exec=lambda query:SimpleNamespace(all=lambda:[]))
    monkeypatch.setattr(module,'get_db_session',db_session)
    wait=AsyncMock(side_effect=[TimeoutError() for _ in range(cycles)]+[asyncio.CancelledError()])
    monkeypatch.setattr(module.asyncio,'wait_for',wait)
    register=AsyncMock(side_effect=RuntimeError('upstream accounts limited')) if failure=='exception' else AsyncMock(return_value='failed')
    manager=SimpleNamespace(_emoji_num=100,
        _maintenance_wakeup_event=SimpleNamespace(wait=Mock(return_value=None),clear=Mock()),
        register_emoji_by_filename=register,check_emoji_file_integrity=Mock())
    with pytest.raises(asyncio.CancelledError):
        await module.EmojiManager.periodic_emoji_maintenance(manager)
    assert register.await_count==expected
    assert 1 <= len({call.args[0].name for call in register.await_args_list}) <= expected
    assert all(call.kwargs['timeout']==600 for call in wait.await_args_list)
    assert manager.check_emoji_file_integrity.call_count==cycles
