"""MaiBot-side provenance checks: private material must not enter group recall."""

from typing import Any
import json

from src.chat.message_receive.chat_manager import chat_manager
from src.A_memorix.host_service import a_memorix_host_service


def group_session(chat_id: str) -> Any:
    session = chat_manager.get_existing_session_by_session_id(chat_id) if chat_id else None
    return session if session and session.is_group_session else None


def _store() -> Any:
    kernel = a_memorix_host_service._kernel
    return kernel.metadata_store if kernel is not None else None


def safe_group_evidence(item: dict, session: Any, store: Any, visited=None) -> bool:
    """Require positive group provenance; reject mixed, unknown and private origins."""
    visited = set() if visited is None else visited
    metadata = item.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return False
    origins = set()
    groups = set()
    private = False

    def walk(value, key=""):
        nonlocal private
        if isinstance(value, dict):
            if value.get("private_only") or value.get("is_private") or value.get("scope_type") == "private":
                private = True
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and value.strip():
            if key in {
                "chat_id",
                "chat_ids",
                "session_id",
                "session_ids",
                "stream_id",
                "stream_ids",
                "source_chat_id",
                "source_chat_ids",
                "origin_chat_id",
                "origin_chat_ids",
                "source_session_id",
                "source_session_ids",
                "origin_session_ids",
                "origin_session_id",
            }:
                origins.add(value.strip())
            if key in {"group_id", "group_ids"}:
                groups.add(value.strip())
            if key == "source":
                for prefix in ("chat_summary:", "chat_stream:", "maibot.chat_history:"):
                    if value.startswith(prefix):
                        origins.add(value[len(prefix) :])

    walk({**item, "metadata": metadata})
    if private or any(group != session.group_id for group in groups):
        return False
    for origin in origins:
        known = chat_manager.get_existing_session_by_session_id(origin)
        if (
            not known
            or not known.is_group_session
            or known.platform != session.platform
            or known.group_id != session.group_id
            or known.account_id != session.account_id
        ):
            return False
    # Validate underlying records too: graph hits/snapshots can summarize several sources.
    kind = item.get("type") or item.get("hit_type")
    key = str((item.get("episode_id") or item.get("hash") or "") if kind == "episode" else (item.get("hash") or ""))
    if key and key not in visited and store is not None:
        visited.add(key)
        if kind == "relation":
            paragraphs = store.get_paragraphs_by_relation_hashes([key]).get(key, [])
            return bool(paragraphs) and all(safe_group_evidence(p, session, store, visited.copy()) for p in paragraphs)
        if kind == "episode":
            record = store.get_episode_by_id(key)
            paragraphs = store.get_episode_paragraphs(key, limit=101)
            if len(paragraphs) > 100 or any(
                not safe_group_evidence(p, session, store, visited.copy()) for p in paragraphs
            ):
                return False
            return bool(record) and safe_group_evidence(record, session, store, visited)
        record = store.get_paragraph(key)
        if record:
            return safe_group_evidence(record, session, store, visited)
    source_paragraph = metadata.get("source_paragraph") if isinstance(metadata, dict) else None
    if source_paragraph and store is not None and source_paragraph not in visited:
        visited.add(source_paragraph)
        paragraph = store.get_paragraph(source_paragraph)
        return bool(paragraph) and safe_group_evidence(paragraph, session, store, visited)
    if origins or groups:
        return True
    source = str(item.get("source") or metadata.get("source", "")) if isinstance(metadata, dict) else ""
    return source.startswith("web_import:")


def filter_search(payload: dict, chat_id: str) -> dict:
    session = group_session(chat_id)
    if session is None:
        return payload
    store = _store()
    hits = [h for h in payload.get("hits", []) if isinstance(h, dict) and safe_group_evidence(h, session, store)]
    # A generated summary may have been composed from hits that were rejected.
    return {
        **payload,
        "hits": hits,
        "summary": "",
        "filtered": len(hits) != len(payload.get("hits", [])) or payload.get("filtered", False),
    }


def profile_view_fields(evidence: list[dict[str, Any]], *, limit: int = 220) -> dict[str, Any]:
    """Render only the evidence already admitted by the caller's provenance boundary."""
    text = "\n".join(str(item.get("content") or item.get("text") or "")[:limit] for item in evidence)
    return {"profile_text": text, "summary": text, "traits": [line for line in text.splitlines() if line]}


def filter_profile(payload: dict, chat_id: str) -> dict:
    session = group_session(chat_id)
    if session is None:
        return payload
    store = _store()
    evidence = payload.get("evidence") or payload.get("vector_evidence") or []
    evidence = [e for e in evidence if isinstance(e, dict) and safe_group_evidence(e, session, store)]
    # Never reuse the global synthesized profile, even if a subset of evidence is safe.
    return {
        "success": bool(payload.get("success", True)),
        "person_id": payload.get("person_id", ""),
        **profile_view_fields(evidence),
        "evidence": evidence,
        "uncertain_candidates": [],
        "profile_source": "current_group_evidence",
    }
