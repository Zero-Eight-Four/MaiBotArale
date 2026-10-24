from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import StaticPool
from sqlmodel import Session, create_engine, select

from src.common.database import database
from src.common.database.database_model import Messages
from src.common.utils.utils_message import MessageUtils


@pytest.fixture
def isolated_store(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Messages.__table__.create(engine)

    @contextmanager
    def get_session(auto_commit=True):
        with Session(engine) as session:
            yield session
            if auto_commit:
                session.commit()

    monkeypatch.setattr(database, "get_db_session", get_session)
    monkeypatch.setattr(MessageUtils, "fill_reply_frequency_if_available", lambda message: None)
    return get_session


def message(mid="42", user="bot", account="bot", session_id="group"):
    def to_db():
        return Messages(message_id=mid, timestamp=datetime.now(), platform="qq", account_id=account,
                        user_id=user, user_nickname=user, session_id=session_id, raw_content=b"",
                        processed_plain_text="same text")
    return SimpleNamespace(message_id=mid, platform="qq", account_id=account, session_id=session_id,
                           is_notify=False, message_info=SimpleNamespace(user_info=SimpleNamespace(user_id=user)),
                           raw_message=SimpleNamespace(components=[]), to_db_instance=to_db)


def test_self_echo_is_idempotent_and_scoped(isolated_store):
    assert MessageUtils.store_message_to_db(message()) is True
    assert MessageUtils.store_message_to_db(message()) is False
    assert MessageUtils.store_message_to_db(message(mid="43")) is True
    assert MessageUtils.store_message_to_db(message(session_id="other")) is True
    assert MessageUtils.store_message_to_db(message(user="otherbot", account="otherbot")) is True
    assert MessageUtils.store_message_to_db(message(user="human")) is True
    assert MessageUtils.store_message_to_db(message(user="human")) is True
    with isolated_store() as session:
        assert len(session.exec(select(Messages)).all()) == 6


def test_echo_arriving_before_receipt_is_merged(isolated_store):
    MessageUtils.store_message_to_db(message(mid="temporary"))
    MessageUtils.store_message_to_db(message(mid="42"))
    # 相同消息 ID 在其他会话出现时不能误删。
    MessageUtils.store_message_to_db(message(mid="42", session_id="other"))
    assert MessageUtils.update_message_id("temporary", "42") is True
    with isolated_store() as session:
        rows = session.exec(select(Messages)).all()
        assert len(rows) == 2
        assert all(row.message_id == "42" for row in rows)
    assert MessageUtils.store_message_to_db(message()) is False


def test_receipt_arriving_before_echo_is_idempotent(isolated_store):
    MessageUtils.store_message_to_db(message(mid="temporary"))
    assert MessageUtils.update_message_id("temporary", "42") is True
    assert MessageUtils.store_message_to_db(message()) is False
    with isolated_store() as session:
        assert len(session.exec(select(Messages)).all()) == 1


def test_reading_old_echoes_does_not_delete_history(isolated_store, monkeypatch):
    from src.common import message_repository

    # 模拟修复前的库内容。
    with isolated_store() as session:
        session.add(message().to_db_instance())
        session.add(message().to_db_instance())
        session.add(message(user="human").to_db_instance())
        session.add(message(user="human").to_db_instance())
    monkeypatch.setattr(message_repository, "get_db_session", isolated_store)
    monkeypatch.setattr(message_repository, "_message_to_instance", lambda row: row)
    rows = message_repository.find_messages(session_id="group")
    assert len(rows) == 3
    with isolated_store() as session:
        assert len(session.exec(select(Messages)).all()) == 4


def test_self_report_uses_sent_history_instead_of_planner_input():
    import asyncio
    from unittest.mock import Mock
    from src.maisaka.runtime import MaisakaHeartFlowChatting

    runtime = SimpleNamespace(_focus_group=None, append_sent_message_to_chat_history=Mock(return_value=True))
    echo = message()
    asyncio.run(MaisakaHeartFlowChatting._register_message(runtime, echo))
    runtime.append_sent_message_to_chat_history.assert_called_once_with(echo)


def test_manual_qq_send_with_new_id_is_kept_even_with_same_text(isolated_store):
    assert MessageUtils.store_message_to_db(message(mid='reply-id')) is True
    assert MessageUtils.store_message_to_db(message(mid='reply-id')) is False
    assert MessageUtils.store_message_to_db(message(mid='manual-id')) is True
    with isolated_store() as session:
        assert len(session.exec(select(Messages)).all()) == 2
