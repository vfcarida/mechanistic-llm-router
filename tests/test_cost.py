"""Verification tests for dynamic cost savings calculation (MLR-T04 / MLR-F10)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mechanistic_router.gateway.server import app, compute_cost_savings, rate_limiter
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TargetModel, TaskComplexity

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_limiter():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_compute_cost_savings_default_pool() -> None:
    """Verify cost_saved_usd equals price(strong) - price(selected) for default model pool."""
    strong_model = max(MODEL_POOL.values(), key=lambda m: m.cost)
    strong_cost = strong_model.cost
    assert strong_cost == 1.50

    # Routine SLM: 1.50 - 0.02 = 1.48
    slm_savings = compute_cost_savings("SLM-BERTau-Local", MODEL_POOL)
    expected_slm = round(strong_cost - MODEL_POOL["SLM-BERTau-Local"].cost, 6)
    assert slm_savings == expected_slm
    assert slm_savings == 1.48

    # Mid-Tier: 1.50 - 0.25 = 1.25
    mid_savings = compute_cost_savings("LLM-Mid-Tier", MODEL_POOL)
    expected_mid = round(strong_cost - MODEL_POOL["LLM-Mid-Tier"].cost, 6)
    assert mid_savings == expected_mid
    assert mid_savings == 1.25

    # Frontier Oracle itself: 1.50 - 1.50 = 0.0
    oracle_savings = compute_cost_savings("LLM-Frontier-Oracle", MODEL_POOL)
    assert oracle_savings == 0.0


def test_compute_cost_savings_is_dynamic_not_constant() -> None:
    """Verify calculation adapts dynamically to custom pricing rather than hardcoded $1.50."""
    custom_pool = {
        "BudgetModel": TargetModel(
            name="BudgetModel",
            cost=0.05,
            base_accuracy=0.80,
            complexity_ceiling=TaskComplexity.ROUTINE,
        ),
        "FlagshipModel": TargetModel(
            name="FlagshipModel",
            cost=4.80,
            base_accuracy=0.99,
            complexity_ceiling=TaskComplexity.COMPLEX,
        ),
    }

    # If static 1.50 were used, this would produce 1.50 - 0.05 = 1.45.
    # Dynamic computation yields 4.80 - 0.05 = 4.75.
    savings = compute_cost_savings("BudgetModel", custom_pool)
    assert savings == 4.75


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_gateway_chat_completions_reports_dynamic_cost_savings(
    mock_acompletion: AsyncMock,
) -> None:
    """Verify /v1/chat/completions calculates and reports honest cost savings."""
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-cost-test"
    mock_choice = MagicMock()
    mock_choice.message.content = "Saldo disponível: R$ 500,00"
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=8, completion_tokens=10, total_tokens=18)
    mock_acompletion.return_value = mock_response

    # Prompt routed to SLM-BERTau-Local (cost $0.02)
    payload = {
        "model": "mechanistic-auto",
        "messages": [{"role": "user", "content": "Qual meu saldo atual?"}],
    }
    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer test-router-key"},
    )
    assert response.status_code == 200
    res_data = response.json()

    # SLM route should save 1.50 - 0.02 = 1.48 USD
    assert res_data["cost_saved_usd"] == 1.48
    assert response.headers["x-cost-saved-usd"] == "1.48"
