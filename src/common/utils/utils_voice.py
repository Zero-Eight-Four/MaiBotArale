"""Native ASR entry point; optional quality policies are implemented by SDK plugins."""
from pathlib import Path
from typing import Optional
import asyncio
import base64
import tempfile

from src.config.config import config_manager, global_config
from src.plugin_runtime.policy_bridge import resolve_policy
from src.services.llm_service import LLMServiceClient


async def get_voice_text(voice_bytes: bytes, *, session_id: str = "", message_id: str = "") -> Optional[str]:
    if not global_config.voice.enable_asr or not voice_bytes:
        return None
    # 以临时资产传递大型音频，不把 Base64 塞入插件 RPC 帧。
    with tempfile.NamedTemporaryFile(suffix=".audio") as asset:
        await asyncio.to_thread(asset.write, voice_bytes)
        asset.flush()
        signature = str(config_manager.get_model_config().model_task_config.voice.model_list)
        handled, result = await resolve_policy("asr", "transcribe", {
            "asset_path": str(Path(asset.name).resolve()), "session_id": session_id,
            "message_id": message_id, "model_signature": signature,
        })
        if handled:
            return result if isinstance(result, str) else None
    response = await LLMServiceClient(task_name="voice", request_type="audio").transcribe_audio(
        base64.b64encode(voice_bytes).decode(), session_id=session_id,
    )
    return response.text or None
