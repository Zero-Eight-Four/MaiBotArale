"""Host-validated, subset-only context projections for extension policies."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import time

if TYPE_CHECKING:
    from src.chat.message_receive.message import SessionMessage
    from src.maisaka.runtime import MaisakaHeartFlowChatting


def _validate_indices(indices: Any, count: int, *, error: str) -> list[int]:
    if (
        not isinstance(indices, list)
        or any(type(index) is not int or not 0 <= index < count for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(error)
    return indices


@dataclass(frozen=True)
class HistoryProjection:
    """A host-owned selection snapshot, bound to one session and exact records."""

    session_id: str
    messages: tuple[Any, ...]


async def prepare_history(messages, session_id="") -> HistoryProjection:
    return HistoryProjection(session_id, tuple(await select_history(messages, session_id)))


async def select_history(messages, session_id="", *, projection: HistoryProjection | None = None):
    if projection is not None:
        if (
            not isinstance(projection, HistoryProjection)
            or projection.session_id != session_id
            or len(messages) != len(projection.messages)
            or any(a is not b for a, b in zip(messages, projection.messages, strict=True))
        ):
            raise ValueError("History projection does not match the session or records")
        return list(projection.messages)
    from src.plugin_runtime.policy_bridge import resolve_mapping_policy

    rows = [
        {
            "message_id": str(getattr(m, "message_id", "") or ""),
            "source_kind": getattr(m, "source_kind", ""),
            "tool_name": getattr(m, "tool_name", ""),
            "success": getattr(m, "success", None),
            "content": str(getattr(m, "content", "") or "")[:2000],
        }
        for m in messages
    ]
    result = await resolve_mapping_policy("context", "select_history", {"session_id": session_id, "messages": rows})
    if result is None:
        return list(messages)
    indices = _validate_indices(result.get("indices"), len(messages), error="Invalid extension context projection")
    return [messages[i] for i in indices]


async def select_visual_assets(components, session_id=""):
    """Allow a plugin to replace unavailable media with text on the context copy only."""
    from src.common.data_models.message_component_data_model import EmojiComponent, ImageComponent, TextComponent
    from src.plugin_runtime.policy_bridge import resolve_mapping_policy

    candidates = [
        (i, c)
        for i, c in enumerate(components)
        if isinstance(c, (EmojiComponent, ImageComponent)) and not c.binary_data
    ]
    if not candidates:
        return list(components)
    rows = [{"hash": c.binary_hash, "kind": c.format_name} for _, c in candidates]
    result = await resolve_mapping_policy("context", "visual_assets", {"session_id": session_id, "assets": rows})
    if result is None:
        return list(components)
    indices = _validate_indices(result.get("indices"), len(candidates), error="Invalid visual asset projection")
    selected = set(indices)
    out = list(components)
    for j, (i, component) in enumerate(candidates):
        if j not in selected:
            out[i] = TextComponent(f"[{component.format_name}: {component.content or '媒体资源已不可用'}]")
    return out


async def select_memory(payload, chat_id, kind):
    from src.plugin_runtime.policy_bridge import resolve_mapping_policy
    from src.services.conversation_privacy import group_session, profile_view_fields

    is_group = group_session(chat_id) is not None
    key = "hits" if kind == "search" else "evidence"
    if kind == "profile" and key not in payload and "vector_evidence" in payload:
        key = "vector_evidence"
    records = payload.get(key, [])
    result = await resolve_mapping_policy(
        "context", "memory_view", {"session_id": chat_id, "kind": kind, "is_group": is_group, "records": records}
    )
    if result is None:
        return payload
    indices = _validate_indices(
        result.get("indices", list(range(len(records)))), len(records), error="Invalid memory projection"
    )
    out = {**payload, key: [records[i] for i in indices]}
    if is_group and kind == "profile":
        limit = max(1, min(1000, int(result.get("max_chars_per_record", 220))))
        out.update(profile_view_fields(out[key], limit=limit), uncertain_candidates=[])
    return out


def conversation_payload(
    runtime: "MaisakaHeartFlowChatting",
    *,
    message: "SessionMessage | None" = None,
    target: "SessionMessage | None" = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture trusted Flow input synchronously before the first policy await."""
    from src.chat.utils.utils import is_bot_self

    def record(item: "SessionMessage") -> dict[str, Any]:
        user_id = str(item.message_info.user_info.user_id)
        return {
            "message_id": str(item.message_id),
            "platform": item.platform,
            "user_id": user_id,
            "is_self": is_bot_self(item.platform, user_id),
            "is_at": item.is_at,
            "is_mentioned": item.is_mentioned,
            "traits": dict(getattr(item, "extension_traits", {}) or {}),
        }

    return {
        "session_id": runtime.session_id,
        "now": time.time(),
        "revision": runtime._context_revision,
        "input_version": runtime._input_version,
        "event_sequence": runtime._extension_sequence,
        "is_group": runtime.chat_stream.is_group_session,
        "last_received_at": runtime._last_message_received_at,
        "message": record(message) if message is not None else None,
        "target": record(target) if target is not None else None,
        "pending": [record(item) for item in runtime.message_cache[runtime._last_processed_index :]],
        "parameters": deepcopy(parameters) if parameters is not None else {},
    }


def reply_host_context(
    runtime: "MaisakaHeartFlowChatting", *, logical_turn_id: str | None, target_message_id: str
) -> dict[str, Any]:
    """Build the Replyer identity envelope from host state, never tool arguments."""
    stream = runtime.chat_stream
    return {
        "stream_id": runtime.session_id,
        "account_id": stream.account_id,
        "user_id": stream.user_id,
        "platform": stream.platform,
        "is_group_chat": stream.is_group_session,
        "logical_turn_id": logical_turn_id,
        "reply_target_message_id": target_message_id,
    }


def window_reply_history(messages: list[Any], *, target: "SessionMessage | None", limit: int) -> list[Any]:
    """Keep the recent window plus its original target and directly quoted messages."""
    from src.maisaka.context.planner_messages import extract_quote_ids_from_message_sequence

    if len(messages) <= limit:
        return messages
    pinned = set()
    if target is not None:
        if target.message_id.strip():
            pinned.add(target.message_id.strip())
        pinned.update(qid.strip() for qid in extract_quote_ids_from_message_sequence(target.raw_message) if qid.strip())
    start = len(messages) - limit
    return [
        item
        for index, item in enumerate(messages)
        if index >= start or str(getattr(item, "message_id", "") or "").strip() in pinned
    ]
