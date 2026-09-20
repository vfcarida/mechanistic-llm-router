"""Tests causal directional ablation harness on synthetic controls."""

import numpy as np
import pytest

from experiments.mechanistic_probe.causal_validator import CausalAblationHarness
from experiments.mechanistic_probe.probe import LinearActivationProbe


def test_ablate_direction_orthogonality() -> None:
    """Verifies that ablating direction v yields an activation matrix orthogonal to v."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 16))
    v = rng.standard_normal(16)
    v /= np.linalg.norm(v)

    X_ablated = CausalAblationHarness.ablate_direction(X, v)

    # Every row of X_ablated must have zero dot product with v
    dot_products = X_ablated @ v
    np.testing.assert_allclose(dot_products, np.zeros(50), atol=1e-6)


def test_causal_ablation_planted_ground_truth_control() -> None:
    """Verifies that causal ablation correctly validates a true causal direction."""
    rng = np.random.default_rng(999)
    N, D = 300, 20
    # Planted causal direction: coordinate 0
    # True relationship: y = 1 if x_0 > 0 else 0
    # All other coordinates are Gaussian noise
    X = rng.standard_normal((N, D)).astype(np.float32)
    # Give strong signal along coordinate 0
    X[:, 0] *= 2.0
    y = (X[:, 0] > 0).astype(int)

    # Split into train (200) and test (100)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    probe = LinearActivationProbe(random_state=42)
    probe.fit(X_train, y_train)

    # Check probe unablated accuracy is high
    unablated_acc = float(np.mean(probe.predict(X_test) == y_test))
    assert unablated_acc > 0.85

    harness = CausalAblationHarness(num_controls=40, n_bootstraps=500, seed=42)
    result = harness.validate(probe, X_test, y_test)

    # 1. Unablated accuracy should match probe
    assert result.unablated_accuracy == pytest.approx(unablated_acc, abs=1e-4)

    # 2. Target ablation (removing probe direction) must drop accuracy to near chance (~50%)
    assert result.target_ablated_accuracy < 0.65
    assert result.target_delta < -0.25

    # 3. Control ablation (removing random noise directions) should stay high (>80%)
    assert result.control_ablated_accuracy > 0.80

    # 4. Causal effect size must be strongly negative and CI must exclude zero
    assert result.causal_effect_size < -0.20
    assert result.causal_effect_ci[1] < 0.0
    assert result.is_causal is True


def test_causal_ablation_rejects_spurious_non_causal_direction() -> None:
    """Verifies that ablating an orthogonal non-causal direction is rejected as non-causal."""
    rng = np.random.default_rng(555)
    N, D = 300, 20
    # True signal on coordinate 0
    X = rng.standard_normal((N, D)).astype(np.float32)
    X[:, 0] *= 2.0
    y = (X[:, 0] > 0).astype(int)

    probe = LinearActivationProbe(random_state=42)
    probe.fit(X[:200], y[:200])

    # Construct a probe with a spurious direction (coordinate 15, which is pure noise)
    class SpuriousProbe:
        def __init__(self, original_probe: LinearActivationProbe):
            self.orig = original_probe
            # Set direction to noise coordinate
            spurious_dir = np.zeros(D, dtype=np.float32)
            spurious_dir[15] = 1.0
            self.direction_vector = spurious_dir

        def predict(self, X_eval: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            return self.orig.predict(X_eval, threshold=threshold)

    spurious = SpuriousProbe(probe)
    harness = CausalAblationHarness(num_controls=40, n_bootstraps=500, seed=42)
    result = harness.validate(spurious, X[200:], y[200:])  # type: ignore[arg-type]

    # Ablating noise coordinate 15 does not degrade accuracy compared to random control
    assert result.is_causal is False
    assert result.causal_effect_ci[1] >= -0.05  # CI overlaps or near zero
