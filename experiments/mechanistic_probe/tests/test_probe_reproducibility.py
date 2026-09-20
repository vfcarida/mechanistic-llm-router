"""Tests probe training reproducibility and deterministic convergence."""

import numpy as np
import pytest

from experiments.mechanistic_probe.probe import LinearActivationProbe


def test_probe_reproducibility_fixed_seed() -> None:
    """Verifies that fitting probe with identical data and seed produces identical parameters."""
    rng = np.random.default_rng(123)
    N, D = 100, 32
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    probe1 = LinearActivationProbe(random_state=42)
    probe1.fit(X, y)

    probe2 = LinearActivationProbe(random_state=42)
    probe2.fit(X, y)

    # Verify bitwise identical direction vectors and intercept
    np.testing.assert_allclose(probe1.direction_vector, probe2.direction_vector, rtol=1e-6)
    assert probe1.direction_norm == pytest.approx(probe2.direction_norm, rel=1e-6)
    assert probe1.intercept == pytest.approx(probe2.intercept, rel=1e-6)

    # Verify identical probability outputs
    X_test = rng.standard_normal((20, D)).astype(np.float32)
    probs1 = probe1.predict_proba(X_test)
    probs2 = probe2.predict_proba(X_test)
    np.testing.assert_allclose(probs1, probs2, rtol=1e-6)


def test_probe_unfitted_raises_error() -> None:
    """Verifies that accessing properties before fitting raises clear ValueError."""
    probe = LinearActivationProbe()
    X = np.zeros((5, 10))

    with pytest.raises(ValueError, match="Probe has not been fitted yet"):
        _ = probe.direction_vector

    with pytest.raises(ValueError, match="Probe has not been fitted yet"):
        probe.predict_proba(X)

    with pytest.raises(ValueError, match="Probe has not been fitted yet"):
        probe.decision_function(X)


def test_probe_single_class_error() -> None:
    """Verifies that attempting to fit on single-class data raises ValueError."""
    probe = LinearActivationProbe()
    X = np.zeros((10, 5))
    y = np.ones(10, dtype=int)

    with pytest.raises(ValueError, match="Expected at least 2 distinct classes"):
        probe.fit(X, y)
