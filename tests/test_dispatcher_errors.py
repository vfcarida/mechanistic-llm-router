"""Verification tests for provider error propagation and zero fabricated responses (MLR-T04)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from mechanistic_router.gateway.dispatcher import LiteLLMDispatcher, ProviderDispatchError
from mechanistic_router.gateway.server import app
from mechanistic_router.schemas.openai import ChatCompletionMessage, ChatCompletionRequest

client = TestClient(app)


@pytest.mark.asyncio
@patch("litellm.acompletion", side_effect=RuntimeError("Connection refused by upstream provider"))
async def test_dispatcher_raises_typed_error(mock_acompletion: AsyncMock) -> None:
    """Ensure dispatcher raises ProviderDispatchError when upstream fails without fallbacks."""
    dispatcher = LiteLLMDispatcher(max_retries=2, cooldown_seconds=0.01)
    req = ChatCompletionRequest(
        model="SLM-BERTau-Local",
        messages=[ChatCompletionMessage(role="user", content="Test query")],
    )

    with pytest.raises(ProviderDispatchError) as exc_info:
        await dispatcher.dispatch("SLM-BERTau-Local", req)

    assert exc_info.value.target_model == "SLM-BERTau-Local"
    assert exc_info.value.status_code == 502
    assert "Connection refused" in exc_info.value.message
    assert mock_acompletion.call_count == 2


@patch("litellm.acompletion", side_effect=Exception("Upstream 500 Internal Server Error"))
def test_gateway_endpoint_returns_502_without_fabricated_content(
    mock_acompletion: AsyncMock,
) -> None:
    """Ensure gateway returns HTTP 502 on failure and does NOT synthesize assistant message."""
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Consultar limite disponível."}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-router-key"},
    )

    assert response.status_code == 502
    res_data = response.json()

    # Verify no fabricated completion choices or assistant messages exist in the response
    assert "choices" not in res_data
    assert "Router Dispatcher Fallback" not in response.text
    assert "error" in res_data
    assert res_data["error"]["type"] == "provider_dispatch_error"
    assert res_data["error"]["code"] == "upstream_provider_failure"


def test_gateway_endpoint_rejects_invalid_key_before_dispatch() -> None:
    """Ensure requests with invalid credentials receive 401 without dispatching or fallbacks."""
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Consulta teste"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer invalid-secret-token"},
    )
    assert response.status_code == 401
    assert "choices" not in response.json()
    assert response.json()["error"]["type"] == "authentication_error"
