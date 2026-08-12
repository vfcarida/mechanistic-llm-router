"""Unit and Integration Tests for LiteLLM Dispatcher & FastAPI REST Gateway Server."""

import pytest
from fastapi.testclient import TestClient
from mechanistic_router.gateway.dispatcher import LiteLLMDispatcher
from mechanistic_router.gateway.server import app
from mechanistic_router.schemas.openai import ChatCompletionRequest, ChatCompletionMessage

client = TestClient(app)


@pytest.mark.asyncio
async def test_litellm_dispatcher_fallback() -> None:
    """Test LiteLLM dispatcher execution and graceful fallback handling."""
    dispatcher = LiteLLMDispatcher(max_retries=1)
    req = ChatCompletionRequest(
        model="SLM-BERTau-Local",
        messages=[ChatCompletionMessage(role="user", content="Hello test prompt")],
    )

    res = await dispatcher.dispatch("SLM-BERTau-Local", req)
    assert res.model == "SLM-BERTau-Local"
    assert len(res.choices) == 1
    assert res.choices[0].message.role == "assistant"


def test_gateway_healthz() -> None:
    """Test /healthz REST endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_gateway_list_models() -> None:
    """Test /v1/models REST endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    assert any(m["id"] == "mechanistic-auto" for m in data)


def test_gateway_chat_completions() -> None:
    """Test /v1/chat/completions drop-in OpenAI REST endpoint."""
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "What is my credit card balance?"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "choices" in res_data
    assert len(res_data["choices"]) > 0
    assert res_data["choices"][0]["message"]["role"] == "assistant"
