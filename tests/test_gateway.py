"""Unit and Integration Tests for LiteLLM Dispatcher & FastAPI REST Gateway Server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mechanistic_router.gateway.dispatcher import LiteLLMDispatcher, ProviderDispatchError
from mechanistic_router.gateway.server import app
from mechanistic_router.schemas.openai import ChatCompletionMessage, ChatCompletionRequest

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
