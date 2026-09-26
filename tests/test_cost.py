"""Verification tests for dynamic cost savings calculation (MLR-T04 / MLR-F10)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mechanistic_router.gateway.server import app, compute_cost_savings, rate_limiter
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TargetModel, TaskComplexity
from mechanistic_router.utils.metrics import normalized_accuracy, normalized_inverse_cost

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


def test_normalized_inverse_cost_uniform_and_invalid_pricing() -> None:
    """Test normalized inverse cost with uniform pricing and invalid boundary values."""
    # When cost_min == cost_max (uniform pricing), denominator is 0 -> returns 0.5 neutral score
    assert normalized_inverse_cost(cost=0.10, cost_min=0.10, cost_max=0.10) == 0.5
    assert normalized_inverse_cost(cost=1.50, cost_min=1.50, cost_max=1.50) == 0.5

    # Invalid cost arguments must raise ValueError
    with pytest.raises(ValueError, match="strictly positive"):
        normalized_inverse_cost(cost=0.0, cost_min=0.02, cost_max=1.50)

    with pytest.raises(ValueError, match="strictly positive"):
        normalized_inverse_cost(cost=-0.50, cost_min=0.02, cost_max=1.50)

    with pytest.raises(ValueError, match="strictly positive"):
        normalized_inverse_cost(cost=0.50, cost_min=0.0, cost_max=1.50)


def test_normalized_accuracy_edge_cases() -> None:
    """Test normalized accuracy calculations across edge cases and zero range."""
    # When accuracy floor equals ceiling -> returns 1.0
    assert normalized_accuracy(accuracy=0.85, accuracy_floor=0.85, accuracy_ceiling=0.85) == 1.0

    # Normal scaling
    assert normalized_accuracy(
        accuracy=0.80, accuracy_floor=0.70, accuracy_ceiling=0.90
    ) == pytest.approx(0.5)

    # Clamping outside bounds
    assert normalized_accuracy(accuracy=0.50, accuracy_floor=0.70, accuracy_ceiling=0.90) == 0.0
    assert normalized_accuracy(accuracy=1.00, accuracy_floor=0.70, accuracy_ceiling=0.90) == 1.0
