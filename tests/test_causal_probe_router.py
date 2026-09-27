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


def test_linear_activation_probe_save_and_load(tmp_path) -> None:
    """Test persisting LinearActivationProbe via compressed NumPy archive and loading it back."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 8).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    probe = LinearActivationProbe(C=0.5, random_state=42, max_iter=500)
    probe.fit(X, y)

    save_path = tmp_path / "probe_checkpoint.npz"
    probe.save(save_path)
    assert save_path.is_file()

    loaded_probe = LinearActivationProbe.load(save_path)

    # Verify structural and parameter fidelity
    assert loaded_probe.C == 0.5
    assert loaded_probe.random_state == 42
    assert loaded_probe.max_iter == 500
    np.testing.assert_allclose(loaded_probe.weights, probe.weights, rtol=1e-5)
    np.testing.assert_allclose(loaded_probe.direction_vector, probe.direction_vector, rtol=1e-5)
    assert abs(loaded_probe.direction_norm - probe.direction_norm) < 1e-5
    assert abs(loaded_probe.intercept - probe.intercept) < 1e-5

    # Verify prediction parity
    np.testing.assert_allclose(loaded_probe.predict_proba(X), probe.predict_proba(X), rtol=1e-5)
    np.testing.assert_array_equal(loaded_probe.predict(X), probe.predict(X))
    np.testing.assert_allclose(
        loaded_probe.decision_function(X), probe.decision_function(X), rtol=1e-5
    )


def test_linear_activation_probe_unfitted_save_and_access_errors(tmp_path) -> None:
    """Test unfitted LinearActivationProbe errors gracefully on save or property access."""
    probe = LinearActivationProbe()

    with pytest.raises(ValueError, match="Cannot save an unfitted probe"):
        probe.save(tmp_path / "fail.npz")

    with pytest.raises(ValueError, match="Probe has not been fitted yet"):
        _ = probe.direction_vector

    with pytest.raises(ValueError, match="Probe has not been fitted yet"):
        _ = probe.weights


def test_linear_activation_probe_load_missing_file(tmp_path) -> None:
    """Test loading non-existent file raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.npz"
    with pytest.raises(FileNotFoundError, match="Probe weights archive not found"):
        LinearActivationProbe.load(missing_file)


def test_linear_activation_probe_pure_numpy_inference(tmp_path) -> None:
    """Test LinearActivationProbe performs accurate vector inference without scikit-learn."""
    rng = np.random.RandomState(42)
    X = rng.randn(25, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    probe = LinearActivationProbe()
    probe.fit(X, y)

    save_path = tmp_path / "pure_numpy.npz"
    probe.save(save_path)

    loaded = LinearActivationProbe.load(save_path)
    # Strip classifier object to simulate environment where scikit-learn is not installed
    loaded.classifier = None

    probs = loaded.predict_proba(X)
    preds = loaded.predict(X)
    margin = loaded.decision_function(X)

    assert probs.shape == (25,)
    assert preds.shape == (25,)
    assert margin.shape == (25,)
    np.testing.assert_allclose(probs, probe.predict_proba(X), rtol=1e-5)
    np.testing.assert_array_equal(preds, probe.predict(X))


@pytest.mark.asyncio
async def test_causal_probe_router_from_saved_probe(tmp_path) -> None:
    """Test instantiating CausalProbeRouter via from_saved_probe factory."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 8).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    probe = LinearActivationProbe()
    probe.fit(X, y)
    probe_path = tmp_path / "router_probe.npz"
    probe.save(probe_path)

    mock_extractor = MagicMock()
    # Query activation that strongly fires class 1
    mock_extractor.extract_one.return_value = np.array(
        [2.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    router = CausalProbeRouter.from_saved_probe(
        probe_path=probe_path,
        extractor=mock_extractor,
        model_pool=MODEL_POOL,
        config=DEFAULT_CONFIG,
    )

    decision = await router.route("Complex query requiring advanced reasoning")
    assert decision.strategy_used == "CausalProbeRouter"
    assert decision.selected_model == router.strong_model
    assert "margin" in decision.signals[router.strong_model].extra_metadata


def test_causal_probe_router_fit_and_save(tmp_path) -> None:
    """Test CausalProbeRouter offline training and persistence methods."""
    mock_extractor = MagicMock()
    rng = np.random.RandomState(42)
    mock_extractor.extract_batch.return_value = rng.randn(10, 8).astype(np.float32)

    probe = LinearActivationProbe()
    router = CausalProbeRouter(
        extractor=mock_extractor,
        probe=probe,
        model_pool=MODEL_POOL,
        config=DEFAULT_CONFIG,
    )

    prompts = [f"Prompt {i}" for i in range(10)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    router.fit(prompts, labels)

    save_path = tmp_path / "persisted_probe.npz"
    router.save_probe(save_path)
    assert save_path.is_file()

    # Load back into router
    new_probe_path = tmp_path / "persisted_probe.npz"
    router.load_probe(new_probe_path)
    assert router.probe.weights.shape == (8,)


def test_simulated_router_alias() -> None:
    """Test SimulatedRouter is an exact alias for MechanisticRouter."""
    from mechanistic_router import MechanisticRouter, SimulatedRouter
    from mechanistic_router.routers.mechanistic import (
        MechanisticRouter as MR,
    )
    from mechanistic_router.routers.mechanistic import (
        SimulatedRouter as SR,
    )

    assert SimulatedRouter is MechanisticRouter
    assert SR is MR
