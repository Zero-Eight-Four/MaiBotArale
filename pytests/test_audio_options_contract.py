import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from openai.types.chat import ChatCompletion
import pytest

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.model_client.base_client import AudioTranscriptionOptions, AudioTranscriptionRequest
from src.llm_models.model_client.openai_client import OpenaiClient
from src.llm_models.model_client.gemini_client import GeminiClient


def audio_request(**options):
    return AudioTranscriptionRequest(
        model_info=ModelInfo(name="voice", model_type="llm", model_identifier="voice", api_provider="test"),
        audio_base64=base64.b64encode(b"fixture-audio").decode(),
        extra_params=options,
    )


def openai_client():
    c = object.__new__(OpenaiClient)
    c.api_provider = APIProvider(name="test", api_key="test", client_type="openai", base_url="https://example.invalid")
    c._omit_sdk_authorization = False
    raw = ChatCompletion(
        id="fixture",
        object="chat.completion",
        created=1,
        model="voice",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "transcript"}}],
    )
    c.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=raw))),
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(text="transcript")))
        ),
    )
    return c


@pytest.mark.asyncio
@pytest.mark.parametrize("mime,fmt", [("audio/wav", "wav"), ("audio/mpeg", "mp3")])
async def test_audio_chat_respects_format_and_keeps_options_out_of_extra_body(mime, fmt):
    c = openai_client()
    response, _ = await c._execute_audio_transcription_request(
        audio_request(
            audio_transcription_mode="chat",
            audio_mime_type=mime,
            transcription_prompt="literal words",
            audio_temperature=0.2,
        )
    )
    assert response.content == "transcript"
    call = c.client.chat.completions.create.await_args.kwargs
    assert call["messages"][0]["content"][1]["input_audio"]["format"] == fmt
    assert call["messages"][0]["content"][0]["text"] == "literal words"
    assert call["temperature"] == 0.2 and call["extra_body"] is None


@pytest.mark.asyncio
async def test_transcription_upload_uses_matching_container_name():
    c = openai_client()
    await c._execute_audio_transcription_request(audio_request(audio_mime_type="audio/mpeg", audio_language="zh"))
    call = c.client.audio.transcriptions.create.await_args.kwargs
    assert call["file"][0] == "audio.mp3" and call["language"] == "zh"


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [OpenaiClient, GeminiClient])
async def test_invalid_temperature_is_rejected_before_network_and_saved(monkeypatch, cls):
    module = __import__(cls.__module__, fromlist=["save_failed_request_snapshot"])
    saved = Mock(return_value=Path("fixture-snapshot"))
    monkeypatch.setattr(module, "save_failed_request_snapshot", saved)
    c = object.__new__(cls)
    c.api_provider = APIProvider(name="test", api_key="test", base_url="https://example.invalid")
    with pytest.raises(ValueError, match="有限数值"):
        await c._execute_audio_transcription_request(audio_request(audio_temperature=float("nan")))
    saved.assert_called_once()


def test_provider_specific_format_support_is_not_erased():
    options = AudioTranscriptionOptions.from_extra_params({"audio_mime_type": "audio/flac"})
    assert options.openai_file_name() == "audio.flac"
    with pytest.raises(ValueError, match="WAV或MP3"):
        options.openai_audio_format()
