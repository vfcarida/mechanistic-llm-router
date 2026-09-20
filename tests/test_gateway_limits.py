"""Verification tests for gateway security limits: auth, rate limiting, and size (MLR-T04)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mechanistic_router.gateway.server import app, rate_limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter buckets before each test."""
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_missing_api_key_returns_401() -> None:
    """Ensure missing API key header returns HTTP 401 Unauthorized."""
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Hello world"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 401
    res_data = response.json()
    assert res_data["error"]["type"] == "authentication_error"
    assert res_data["error"]["code"] == "invalid_api_key"


def test_invalid_api_key_returns_401() -> None:
    """Ensure incorrect API key returns HTTP 401 Unauthorized."""
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Hello world"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer bad-key"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_x_api_key_header_accepted() -> None:
    """Ensure X-API-Key header is accepted as valid authentication."""
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test-key"
        mock_choice = MagicMock()
        mock_choice.message.content = "OK"
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        mock_acompletion.return_value = mock_response

        payload = {
            "model": "mechanistic-auto",
            "messages": [{"role": "user", "content": "Saldo da conta"}],
        }
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-API-Key": "test-router-key"},
        )
        assert response.status_code == 200


def test_oversize_prompt_returns_413() -> None:
    """Ensure prompt exceeding max_prompt_chars returns HTTP 413 Payload Too Large."""
    huge_prompt = "x" * 10_005
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": huge_prompt}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-router-key"},
    )
    assert response.status_code == 413
    res_data = response.json()
    assert res_data["error"]["code"] == "prompt_too_large"
    assert "exceeds maximum allowed limit" in res_data["error"]["message"]


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_burst_rate_limit_returns_429(mock_acompletion: AsyncMock) -> None:
    """Ensure rapid request bursts exceeding bucket capacity return HTTP 429."""
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-burst"
    mock_choice = MagicMock()
    mock_choice.message.content = "Response"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    mock_acompletion.return_value = mock_response

    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Ping"}],
    }
    headers = {"Authorization": "Bearer test-router-key"}

    # Temporarily set small burst capacity with minimal refill for deterministic testing
    orig_cap = rate_limiter.capacity
    orig_rate = rate_limiter.refill_rate
    rate_limiter.capacity = 3.0
    rate_limiter.refill_rate = 0.01
    rate_limiter.reset()

    try:
        responses = [
            client.post("/v1/chat/completions", json=payload, headers=headers)
            for _ in range(6)
        ]
        status_codes = [r.status_code for r in responses]

        # At least one trailing burst request must trigger 429
        assert 429 in status_codes
    finally:
        rate_limiter.capacity = orig_cap
        rate_limiter.refill_rate = orig_rate
        rate_limiter.reset()
    idx_429 = status_codes.index(429)
    res_429 = responses[idx_429].json()
    assert res_429["error"]["type"] == "rate_limit_error"
    assert res_429["error"]["code"] == "rate_limit_exceeded"
