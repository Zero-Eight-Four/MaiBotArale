"""Host-validated, subset-only context projections for extension policies."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HistoryProjection:
    """A host-owned selection snapshot, bound to one session and exact records."""
    session_id: str
    messages: tuple[Any, ...]


async def prepare_history(messages, session_id="") -> HistoryProjection:
    return HistoryProjection(session_id, tuple(await select_history(messages, session_id)))


async def select_history(messages,session_id="", *, projection: HistoryProjection | None = None):
    if projection is not None:
        if (not isinstance(projection, HistoryProjection) or projection.session_id != session_id
                or len(messages) != len(projection.messages)
                or any(a is not b for a, b in zip(messages, projection.messages))):
            raise ValueError("History projection does not match the session or records")
        return list(projection.messages)
    from src.plugin_runtime.policy_bridge import resolve_policy
    rows=[{"message_id":str(getattr(m,"message_id","") or ""),"source_kind":getattr(m,"source_kind",""),
        "tool_name":getattr(m,"tool_name",""),"success":getattr(m,"success",None),
        "content":str(getattr(m,"content","") or "")[:2000]} for m in messages]
    handled,result=await resolve_policy("context","select_history",{"session_id":session_id,"messages":rows})
    if not handled:return list(messages)
    indices=result.get("indices")
    if not isinstance(indices,list) or any(type(i) is not int or i<0 or i>=len(messages) for i in indices) or len(set(indices))!=len(indices):
        raise ValueError("Invalid extension context projection")
    return [messages[i] for i in indices]

async def select_visual_assets(components, session_id=""):
    """Allow a plugin to replace unavailable media with text on the context copy only."""
    from src.common.data_models.message_component_data_model import EmojiComponent, ImageComponent, TextComponent
    from src.plugin_runtime.policy_bridge import resolve_policy

    candidates = [(i, c) for i, c in enumerate(components)
                  if isinstance(c, (EmojiComponent, ImageComponent)) and not c.binary_data]
    if not candidates:
        return list(components)
    rows = [{"hash": c.binary_hash, "kind": c.format_name} for _, c in candidates]
    handled, result = await resolve_policy("context", "visual_assets", {"session_id": session_id, "assets": rows})
    if not handled:
        return list(components)
    indices = result.get("indices")
    if (not isinstance(indices, list)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Invalid visual asset projection")
    selected = set(indices)
    out = list(components)
    for j, (i, component) in enumerate(candidates):
        if j not in selected:
            out[i] = TextComponent(f"[{component.format_name}: {component.content or '媒体资源已不可用'}]")
    return out


async def select_memory(payload,chat_id,kind):
    from src.plugin_runtime.policy_bridge import resolve_policy
    from src.services.conversation_privacy import group_session
    is_group=group_session(chat_id) is not None
    key="hits" if kind=="search" else "evidence"
    if kind=="profile" and key not in payload and "vector_evidence" in payload:key="vector_evidence"
    records=payload.get(key,[])
    handled,result=await resolve_policy("context","memory_view",{"session_id":chat_id,"kind":kind,"is_group":is_group,"records":records})
    if not handled:return payload
    indices=result.get("indices",list(range(len(records))))
    if not isinstance(indices,list) or any(type(i) is not int or i<0 or i>=len(records) for i in indices) or len(set(indices))!=len(indices):
        raise ValueError("Invalid memory projection")
    out={**payload,key:[records[i] for i in indices]}
    if is_group and kind=="profile":
        limit=max(1,min(1000,int(result.get("max_chars_per_record",220))))
        text="\n".join(str(e.get("content") or e.get("text") or "")[:limit] for e in out[key])
        out.update(profile_text=text,summary=text,traits=text.splitlines(),uncertain_candidates=[])
    return out
