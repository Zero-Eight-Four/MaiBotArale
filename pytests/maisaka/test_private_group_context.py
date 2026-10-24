import json
from types import SimpleNamespace
import pytest
from src.services import conversation_privacy as privacy
from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext

def private(user="2730843725", sid="private-a", account="bot"):
    return SimpleNamespace(
        is_group_session=False, platform="qq", user_id=user, session_id=sid, account_id=account, group_id=""
    )


def group(sid="group-a"):
    return SimpleNamespace(
        is_group_session=True, platform="qq", user_id="", session_id=sid, account_id="bot", group_id="959142150"
    )


@pytest.fixture
def scoped_privacy(monkeypatch):
    sessions = {"group-a": group(), "private-a": private()}
    monkeypatch.setattr(privacy.chat_manager, "get_existing_session_by_session_id", lambda sid: sessions.get(sid))
    monkeypatch.setattr(privacy, "_store", lambda: None)


def test_group_recall_rejects_private_mixed_unknown_sources(scoped_privacy):
    hits = [
        {"content": "public", "metadata": {"chat_id": "group-a"}},
        {"content": "secret", "metadata": {"chat_id": "private-a"}},
        {"content": "mixed", "metadata": {"chat_ids": ["group-a", "private-a"]}},
        {"content": "unknown", "metadata": {}},
        {"content": "import", "source": "web_import:doc"},
    ]
    out = privacy.filter_search({"hits": hits, "summary": "secret summary"}, "group-a")
    assert [h["content"] for h in out["hits"]] == ["public", "import"] and not out["summary"]
    assert privacy.filter_search({"hits": hits}, "private-a")["hits"] == hits


def test_group_profile_rebuilds_only_from_group_evidence(scoped_privacy):
    profile = {
        "success": True,
        "profile_text": "global secret",
        "summary": "global secret",
        "uncertain_candidates": [{"text": "private"}],
        "evidence": [
            {"content": "群里公开说过", "metadata": {"chat_id": "group-a"}},
            {"content": "私聊秘密", "metadata": {"chat_id": "private-a"}},
        ],
    }
    out = privacy.filter_profile(profile, "group-a")
    assert out["summary"] == "群里公开说过" and not out["uncertain_candidates"]
    assert "global secret" not in json.dumps(out) and "私聊秘密" not in json.dumps(out, ensure_ascii=False)


@pytest.mark.asyncio
async def test_profile_query_uses_group_evidence_and_preserves_admin(monkeypatch, scoped_privacy):
    from src.services.memory_service import MemoryService

    service = MemoryService()
    calls = []

    async def invoke(component, **args):
        calls.append(args)
        return {
            "success": True,
            "profile_text": "private-derived summary",
            "evidence": [{"content": "群聊证据", "metadata": {"chat_id": "group-a"}}],
        }

    monkeypatch.setattr(service, "_invoke_admin", invoke)
    group_result = await service.profile_admin(action="query", chat_id="group-a", person_id="person")
    assert calls[-1]["action"] == "evidence" and group_result["profile_text"] == "群聊证据"
    private_result = await service.profile_admin(action="query", chat_id="private-a", person_id="person")
    assert calls[-1]["action"] == "query" and private_result["profile_text"] == "private-derived summary"
    await service.profile_admin(action="query", person_id="person")
    assert calls[-1]["action"] == "query"


def test_relation_with_private_support_is_rejected(scoped_privacy):
    from unittest.mock import Mock

    store = Mock()
    store.get_paragraphs_by_relation_hashes.return_value = {
        "r": [
            {"hash": "public", "metadata": {"chat_id": "group-a"}},
            {"hash": "secret", "metadata": {"chat_id": "private-a"}},
        ]
    }
    store.get_paragraph.side_effect = lambda key: {
        "hash": key,
        "metadata": {"chat_id": "group-a" if key == "public" else "private-a"},
    }
    assert not privacy.safe_group_evidence({"hash": "r", "type": "relation"}, group(), store)


def test_episode_with_private_underlying_paragraph_is_rejected(scoped_privacy):
    from unittest.mock import Mock

    store = Mock()
    store.get_episode_by_id.return_value = {"metadata": {"chat_id": "group-a"}}
    store.get_episode_paragraphs.return_value = [{"metadata": {"chat_id": "private-a"}}]
    assert not privacy.safe_group_evidence({"type": "episode", "episode_id": "episode"}, group(), store)

