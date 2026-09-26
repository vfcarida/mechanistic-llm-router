"""Unit and Integration Tests for LiteLLM Dispatcher & FastAPI REST Gateway Server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mechanistic_router.gateway.dispatcher import LiteLLMDispatcher, ProviderDispatchError
from mechanistic_router.gateway.server import app, get_router
from mechanistic_router.schemas.openai import ChatCompletionMessage, ChatCompletionRequest
from mechanistic_router.schemas.routing import RoutingDecision

client = TestClient(app)


@pytest.mark.asyncio
async def test_litellm_dispatcher_error_propagation() -> None:
    """Test LiteLLM dispatcher raises ProviderDispatchError on provider failure."""
    dispatcher = LiteLLMDispatcher(max_retries=1)
    req = ChatCompletionRequest(
        model="SLM-BERTau-Local",
        messages=[ChatCompletionMessage(role="user", content="Hello test prompt")],
    )

    with pytest.raises(ProviderDispatchError) as exc_info:
        await dispatcher.dispatch("SLM-BERTau-Local", req)

    assert exc_info.value.target_model == "SLM-BERTau-Local"
    assert exc_info.value.status_code == 502


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


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_gateway_chat_completions(mock_acompletion: AsyncMock) -> None:
    """Test /v1/chat/completions drop-in OpenAI REST endpoint with mocked upstream provider."""
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-test-123"
    mock_choice = MagicMock()
    mock_choice.message.content = "Your balance is R$ 1.500,00."
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=12, completion_tokens=15, total_tokens=27)
    mock_acompletion.return_value = mock_response

    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "What is my credit card balance?"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-router-key"},
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "choices" in res_data
    assert len(res_data["choices"]) > 0
    assert res_data["choices"][0]["message"]["role"] == "assistant"
    assert res_data["choices"][0]["message"]["content"] == "Your balance is R$ 1.500,00."
    assert "cost_saved_usd" in res_data
    assert response.headers.get("x-cost-saved-usd") is not None


def test_gateway_missing_api_key_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure gateway returns 503 when ROUTER_API_KEY is not configured in the environment."""
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "What is my balance?"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer any-key"},
    )
    assert response.status_code == 503
    data = response.json()
    assert data["error"]["code"] == "missing_router_api_key"
    assert "Gateway API key is not configured" in data["error"]["message"]


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_gateway_x_router_strategy_header(mock_acompletion: AsyncMock) -> None:
    """Ensure X-Router-Strategy header selects the corresponding router strategy."""
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-test-semantic"
    mock_choice = MagicMock()
    mock_choice.message.content = "Semantic response"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
    mock_acompletion.return_value = mock_response

    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Consulta rápida de fatura"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={
            "Authorization": "Bearer test-router-key",
            "X-Router-Strategy": "semantic",
        },
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "SemanticRouter" in res_data.get("router_strategy", "")


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_gateway_dependency_override(mock_acompletion: AsyncMock) -> None:
    """Ensure FastAPI dependency injection allows overriding router
    with app.dependency_overrides.
    """
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-override"
    mock_choice = MagicMock()
    mock_choice.message.content = "Overridden router response"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
    mock_acompletion.return_value = mock_response

    mock_custom_router = MagicMock()
    mock_custom_router.route = AsyncMock(
        return_value=RoutingDecision(
            selected_model="SLM-BERTau-Local",
            confidence=0.99,
            latency_ms=1.2,
            estimated_cost_usd=0.001,
            strategy_used="MockInjectedRouter",
            signals={},
        )
    )

    app.dependency_overrides[get_router] = lambda: mock_custom_router
    try:
        payload = {
            "model": "mechanistic-auto",
            "messages": [{"role": "user", "content": "Test prompt with dependency override"}],
        }
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-router-key"},
        )
        assert response.status_code == 200
        res_data = response.json()
        assert "MockInjectedRouter" in res_data.get("router_strategy", "")
        mock_custom_router.route.assert_called_once()
    finally:
        app.dependency_overrides.pop(get_router, None)


@patch("litellm.acompletion")
def test_gateway_streaming_response(mock_acompletion: MagicMock) -> None:
    """Ensure stream=True yields Server-Sent Events (SSE) tokens and final [DONE] event."""

    async def mock_async_stream(*args, **kwargs):
        chunk1 = MagicMock()
        chunk1.id = "chatcmpl-stream-1"
        delta1 = MagicMock()
        delta1.content = "Streaming "
        choice1 = MagicMock()
        choice1.delta = delta1
        chunk1.choices = [choice1]
        yield chunk1

        chunk2 = MagicMock()
        chunk2.id = "chatcmpl-stream-1"
        delta2 = MagicMock()
        delta2.content = "tokens."
        choice2 = MagicMock()
        choice2.delta = delta2
        chunk2.choices = [choice2]
        yield chunk2

    mock_acompletion.return_value = mock_async_stream()

    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-router-key"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    text = response.text
    assert "data: " in text
    assert "Streaming " in text
    assert "tokens." in text
    assert "data: [DONE]" in text
    assert "x-router-strategy" in response.headers
    assert "x-cost-saved-usd" in response.headers


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_gateway_multi_turn_conversation(mock_acompletion: AsyncMock) -> None:
    """Ensure multi-turn conversation messages are accepted and routed properly."""
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-multiturn"
    mock_choice = MagicMock()
    mock_choice.message.content = "Multi-turn assistant answer"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=30, completion_tokens=10, total_tokens=40)
    mock_acompletion.return_value = mock_response

    payload = {
        "model": "mechanistic-auto",
        "messages": [
            {"role": "system", "content": "You are a quantitative finance expert."},
            {"role": "user", "content": "Explain Black-Scholes formula."},
            {"role": "assistant", "content": "The Black-Scholes formula models option pricing."},
            {"role": "user", "content": "Now derive the heat equation transformation."},
        ],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-router-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Multi-turn assistant answer"
