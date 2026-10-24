"""Exercise Gemini's native embedding wire format without network access."""

from pathlib import Path
from typing import Any

import base64
import json

from google import genai
from google.genai.types import HttpOptions
import httpx
import pytest

from src.config.model_configs import APIProvider, ModelInfo
from src.llm_models.exceptions import RespNotOkException, RespParseException
from src.llm_models.model_client.base_client import EmbeddingRequest, ImageEmbeddingRequest
from src.llm_models.model_client.gemini_client import GeminiClient
import src.llm_models.model_client.gemini_client as gemini_module


def make_client(handler: Any) -> GeminiClient:
    client = object.__new__(GeminiClient)
    client.api_provider = APIProvider(
        name="test",
        api_key="test-key",
        client_type="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    client.client = genai.Client(
        api_key="test-key",
        http_options=HttpOptions(
            api_version="v1beta",
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )
    return client


def make_request(data: bytes = b"image-bytes", dimensions: int = 64) -> ImageEmbeddingRequest:
    return ImageEmbeddingRequest(
        model_info=ModelInfo(name="embedding", model_type="image_embedding", model_identifier="gemini-embedding-2", api_provider="test"),
        image_bytes=data,
        mime_type="image/png",
        preprocess_version="identity_v1",
        extra_params={"output_dimensionality": dimensions},
    )


@pytest.mark.asyncio
async def test_image_wire_payload_and_vector_space_fingerprint() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-embedding-2:batchEmbedContents"
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"embeddings": [{"values": [0.25, -0.5]}]})

    client = make_client(handler)
    try:
        first = await client.get_image_embedding(make_request())
        changed_image = await client.get_image_embedding(make_request(b"another-image"))
        changed_dimensions = await client.get_image_embedding(make_request(dimensions=128))
    finally:
        await client.client.aio.aclose()
        client.client.close()

    payload = bodies[0]["requests"][0]
    assert payload["content"]["parts"][0]["inline_data"] == {
        "mime_type": "image/png",
        "data": base64.b64encode(b"image-bytes").decode(),
    }
    assert payload["outputDimensionality"] == 64
    assert "taskType" not in payload  # Embedding 2 has no default task_type.
    assert first.embedding == [0.25, -0.5]
    assert first.request_protocol_hash
    assert first.request_protocol_hash == changed_image.request_protocol_hash
    assert first.request_protocol_hash != changed_dimensions.request_protocol_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-embedding-2", "gemini-embedding-001"])
async def test_text_embedding_remains_supported(model: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)["requests"][0]
        assert payload["content"]["parts"] == [{"text": "hello"}]
        if model == "gemini-embedding-2":
            assert "taskType" not in payload
        else:
            assert payload["taskType"] == "SEMANTIC_SIMILARITY"
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})

    client = make_client(handler)
    try:
        response = await client.get_embedding(
            EmbeddingRequest(
                model_info=ModelInfo(name="text", model_type="embedding", model_identifier=model, api_provider="test"),
                embedding_input="hello",
            )
        )
    finally:
        await client.client.aio.aclose()
        client.client.close()
    assert response.embedding == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, response_body, error",
    [
        (400, {"error": {"code": 400, "message": "bad image", "status": "INVALID_ARGUMENT"}}, RespNotOkException),
        (200, {"embeddings": []}, RespParseException),
        (200, {"embeddings": [{"values": []}]}, RespParseException),
        (200, {"embeddings": [{"values": ["NaN"]}]}, RespParseException),
    ],
)
async def test_failed_image_requests_save_only_redacted_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    response_body: dict,
    error: type[Exception],
) -> None:
    snapshots: list[dict] = []

    def save(**kwargs: Any) -> Path:
        snapshots.append(kwargs)
        return Path("/tmp/test-image-snapshot.json")

    monkeypatch.setattr(gemini_module, "save_failed_request_snapshot", save)
    client = make_client(lambda request: httpx.Response(status, json=response_body))
    try:
        with pytest.raises(error):
            await client.get_image_embedding(make_request())
    finally:
        await client.client.aio.aclose()
        client.client.close()
    assert snapshots[0]["internal_request"]["request_kind"] == "image_embedding"
    assert "image-bytes" not in str(snapshots)
    assert base64.b64encode(b"image-bytes").decode() not in str(snapshots)
