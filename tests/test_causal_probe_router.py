"""Unit tests for CausalProbeRouter and LinearActivationProbe promotion."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.routers.causal_probe import CausalProbeRouter
from mechanistic_router.schemas.routing import RoutingRequest


def test_linear_activation_probe_fit_and_predict() -> None:
    """Test fitting LinearActivationProbe on synthetic activation vectors."""
    rng = np.random.RandomState(42)
    # 20 samples, 16 dimensions
    X = rng.randn(20, 16).astype(np.float32)
    # Class 0 if first component < 0, else 1
    y = (X[:, 0] > 0).astype(int)

    probe = LinearActivationProbe(random_state=42)
    probe.fit(X, y)

    assert probe.direction_vector.shape == (16,)
    assert probe.direction_norm > 0.0

    probs = probe.predict_proba(X)
    assert probs.shape == (20,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = probe.predict(X, threshold=0.5)
    assert preds.shape == (20,)


@pytest.mark.asyncio
async def test_causal_probe_router_routing() -> None:
    """Test CausalProbeRouter decision pipeline with mocked extractor and probe."""
    mock_extractor = MagicMock()
    mock_probe = MagicMock()

    # Low difficulty query -> low prob of needing strong model
    mock_extractor.extract_one.return_value = np.zeros(16, dtype=np.float32)
    mock_probe.predict_proba.return_value = np.array([0.15], dtype=np.float32)
    mock_probe.direction_norm = 1.0

    router = CausalProbeRouter(
        extractor=mock_extractor,
        probe=mock_probe,
        model_pool=MODEL_POOL,
        config=DEFAULT_CONFIG,
        threshold=0.5,
    )

    decision = await router.route(RoutingRequest(prompt="Consulta simples de saldo"))
    assert decision.strategy_used == "CausalProbeRouter"
    assert decision.selected_model == router.cheap_model
    assert "prob_strong" in decision.signals[router.cheap_model].extra_metadata
    assert decision.estimated_cost_usd > 0.0

    # High difficulty query -> high prob of needing strong model
    mock_probe.predict_proba.return_value = np.array([0.85], dtype=np.float32)
    decision_hard = await router.route("Calculo avançado de CET com juros compostos")
    assert decision_hard.selected_model == router.strong_model


@pytest.mark.asyncio
async def test_causal_probe_router_strict_validation() -> None:
    """Test CausalProbeRouter rejects invalid request payloads."""
    mock_extractor = MagicMock()
    mock_probe = MagicMock()

    router = CausalProbeRouter(
        extractor=mock_extractor,
        probe=mock_probe,
        model_pool=MODEL_POOL,
        config=DEFAULT_CONFIG,
    )

    with pytest.raises(TypeError, match="request must be an instance of RoutingRequest or str"):
        await router.route(999)  # type: ignore[arg-type]
