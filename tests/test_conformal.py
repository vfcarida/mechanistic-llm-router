"""Unit and property tests for ConformalRiskController and multi-tier CausalProbeRouter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mechanistic_router.evaluation.baselines import MODEL_POOL
from mechanistic_router.probing.causal_validator import CausalAblationHarness
from mechanistic_router.probing.conformal import (
    ConformalCalibrationSummary,
    ConformalRiskController,
    compute_epistemic_uncertainty,
)
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.routers.causal_probe import CausalProbeRouter
from mechanistic_router.schemas.routing import RoutingRequest


class SyntheticExtractor:
    """Deterministic activation extractor for testing multi-tier conformal routing."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def extract_one(self, prompt: str) -> np.ndarray:
        words = len(prompt.split())
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = (words - 10.0) / 5.0
        rng = np.random.default_rng(abs(hash(prompt)) % (2**31))
        vec[1:] = rng.normal(0.0, 0.1, size=self.dim - 1).astype(np.float32)
        return vec

    def extract_batch(self, prompts: list[str]) -> np.ndarray:
        return np.stack([self.extract_one(p) for p in prompts], axis=0)


def test_compute_epistemic_uncertainty_properties() -> None:
    """Verify binary Shannon entropy properties: boundary zeroes, symmetry, and maximum at 0.5."""
    probs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    entropies = [compute_epistemic_uncertainty(p) for p in probs]

    assert entropies[0] == pytest.approx(0.0, abs=1e-5)
    assert entropies[-1] == pytest.approx(0.0, abs=1e-5)
    assert entropies[3] == pytest.approx(1.0, abs=1e-5)
    # Symmetry: H(p) == H(1 - p)
    assert entropies[1] == pytest.approx(entropies[5], abs=1e-5)
    assert entropies[2] == pytest.approx(entropies[4], abs=1e-5)
    # Monotonicity on [0, 0.5]
    assert entropies[0] < entropies[1] < entropies[2] < entropies[3]


def test_conformal_risk_controller_finite_sample_bound() -> None:
    """Verify finite-sample Conformal Risk Control bound: (n/(n+1))*R_hat + B/(n+1) <= alpha."""
    rng = np.random.default_rng(42)
    n_cal = 200
    probs = rng.uniform(0.0, 1.0, size=n_cal)
    losses = np.where(probs > 0.35, rng.uniform(0.2, 0.8, size=n_cal), 0.0)

    alpha = 0.10
    crc = ConformalRiskController(alpha=alpha, max_loss_bound=1.0)
    summary = crc.calibrate(probs, losses)

    assert isinstance(summary, ConformalCalibrationSummary)
    assert summary.upper_risk_bound_low <= alpha
    assert summary.calibration_size == n_cal
    expected_bound = (n_cal / (n_cal + 1.0)) * summary.empirical_risk_low + (1.0 / (n_cal + 1.0))
    assert summary.upper_risk_bound_low == pytest.approx(expected_bound, abs=1e-6)
    assert 0.35 <= summary.low_threshold <= 0.60
    assert crc.summary == summary


def test_conformal_risk_controller_multitier_and_abstention(tmp_path: Path) -> None:
    """Verify 3-tier calibration, epistemic abstention escalation, and .npz persistence."""
    n_cal = 150
    probs = np.linspace(0.01, 0.99, n_cal)
    losses_cheap = np.where(probs > 0.30, 0.5, 0.0)
    losses_mid = np.where(probs > 0.70, 0.5, 0.0)

    crc = ConformalRiskController(alpha=0.10, max_loss_bound=1.0)
    summary = crc.calibrate(probs, cheap_losses=losses_cheap, mid_losses=losses_mid)

    assert summary.low_threshold <= summary.high_threshold
    assert 0.30 <= summary.low_threshold <= 0.55
    assert 0.70 <= summary.high_threshold <= 0.92

    # Low probability -> tier 0 (cheap)
    tier_low, abstained_low = crc.select_tier_index(0.02, num_tiers=3, max_uncertainty=0.92)
    assert tier_low == 0
    assert abstained_low is False

    # Mid probability with low uncertainty -> tier 1 (mid)
    tier_mid, abstained_mid_clean = crc.select_tier_index(0.65, num_tiers=3, max_uncertainty=0.99)
    assert tier_mid == 1
    assert abstained_mid_clean is False

    # High probability -> tier 2 (strong)
    tier_high, _ = crc.select_tier_index(0.95, num_tiers=3, max_uncertainty=0.92)
    assert tier_high == 2

    # Maximum epistemic uncertainty at p = 0.5 (H_2(0.5) = 1.0 > max_uncertainty=0.92)
    tier_uncertain, abstained_mid = crc.select_tier_index(0.50, num_tiers=3, max_uncertainty=0.92)
    assert abstained_mid is True
    assert tier_uncertain == 2

    # Test 1-tier and 2-tier branches
    assert crc.select_tier_index(0.50, num_tiers=1) == (0, False)
    assert crc.select_tier_index(0.10, num_tiers=2) == (0, False)
    assert crc.select_tier_index(0.90, num_tiers=2) == (1, False)

    # Save and load calibration state
    artifact_path = tmp_path / "conformal_state.npz"
    crc.save(artifact_path)
    loaded = ConformalRiskController.load(artifact_path)
    assert loaded.alpha == pytest.approx(crc.alpha)
    assert loaded.low_threshold == pytest.approx(crc.low_threshold)
    assert loaded.high_threshold == pytest.approx(crc.high_threshold)


def test_conformal_controller_validation_and_edge_cases(tmp_path: Path) -> None:
    """Verify input validation on invalid alpha, bounds, uncalibrated state, and small n."""
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalRiskController(alpha=0.0)
    with pytest.raises(ValueError, match="max_loss_bound B must be > 0"):
        ConformalRiskController(alpha=0.1, max_loss_bound=-1.0)
    with pytest.raises(ValueError, match="grid_size must be >= 11"):
        ConformalRiskController(alpha=0.1, grid_size=5)

    crc = ConformalRiskController(alpha=0.1)
    assert crc.is_calibrated is False
    with pytest.raises(ValueError, match="has not been calibrated yet"):
        _ = crc.low_threshold
    with pytest.raises(ValueError, match="has not been calibrated yet"):
        _ = crc.high_threshold
    with pytest.raises(ValueError, match="has not been calibrated yet"):
        _ = crc.summary
    with pytest.raises(ValueError, match="Cannot save an uncalibrated"):
        crc.save(tmp_path / "uncalibrated.npz")
    with pytest.raises(FileNotFoundError, match="Conformal calibration archive not found"):
        ConformalRiskController.load(tmp_path / "missing.npz")

    with pytest.raises(ValueError, match="cheap_losses must have the same shape"):
        crc.calibrate(np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(ValueError, match="mid_losses must have the same shape"):
        crc.calibrate(np.array([0.1, 0.2]), np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(ValueError, match="probe_probs must be a non-empty 1D array"):
        crc.calibrate(np.array([]), np.array([]))

    # Small n where B / (n + 1) > alpha forces safe threshold = 0.0
    small_summary = crc.calibrate(np.array([0.2, 0.5]), np.array([0.0, 0.0]))
    assert small_summary.low_threshold == 0.0


def test_causal_validator_summary_formatting() -> None:
    """Verify CausalAblationHarness and CausalValidationResult.summary() formatting."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    probe = LinearActivationProbe(random_state=42).fit(X[:40], y[:40])
    harness = CausalAblationHarness(num_controls=10, n_bootstraps=50, seed=42)
    res = harness.validate(probe, X[40:], y[40:])
    text = res.summary()
    assert "Verdict:" in text
    assert "Causal Effect Size:" in text


@pytest.mark.asyncio
async def test_causal_probe_router_multitier_and_conformal_calibration() -> None:
    """Verify CausalProbeRouter 3-tier cascade routing and conformal calibration."""
    extractor = SyntheticExtractor(dim=16)
    probe = LinearActivationProbe(random_state=42)

    train_prompts = [
        "short query one",
        "tiny check two",
        "brief status three",
        "simple balance four",
        (
            "analyze complete multi-jurisdictional corporate restructuring "
            "debt covenant exposure and derivatives"
        ),
        (
            "derive stochastic volatility partial differential equation "
            "with jump diffusion boundary conditions"
        ),
        "evaluate cross-border regulatory capital adequacy under Basel III stress test scenarios",
        (
            "synthesize multi-stage merger arbitrage valuation with "
            "regulatory antitrust divestiture risk"
        ),
    ]
    X_train = extractor.extract_batch(train_prompts)
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probe.fit(X_train, y_train)

    router = CausalProbeRouter(
        model_pool=MODEL_POOL,
        extractor=extractor,  # type: ignore[arg-type]
        probe=probe,
        threshold=0.75,
        mid_threshold=0.25,
        max_uncertainty=0.98,
    )

    # 1. Route routine query -> cheap model
    dec_cheap = await router.route(RoutingRequest(prompt="short query"))
    assert dec_cheap.selected_model == router.cheap_model
    sig_cheap = dec_cheap.signals[dec_cheap.selected_model]
    assert "epistemic_uncertainty" in sig_cheap.extra_metadata
    assert "prob_strong" in sig_cheap.extra_metadata

    # 2. Route complex query -> strong model
    dec_strong = await router.route(
        RoutingRequest(
            prompt=(
                "derive stochastic volatility partial differential equation "
                "with jump diffusion boundary conditions and regulatory capital"
            )
        )
    )
    assert dec_strong.selected_model == router.strong_model

    # 3. Calibrate conformal controller directly on router
    cal_prompts = train_prompts * 5
    losses_cheap = np.array([0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.8, 0.9] * 5, dtype=np.float32)
    losses_mid = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6] * 5, dtype=np.float32)

    summary = router.calibrate_conformal(
        prompts=cal_prompts,
        cheap_losses=losses_cheap,
        mid_losses=losses_mid,
        alpha=0.15,
    )
    assert summary.upper_risk_bound_low <= 0.15
    assert router.conformal_controller is not None

    dec_calibrated = await router.route("short query")
    sig_cal = dec_calibrated.signals[dec_calibrated.selected_model]
    assert sig_cal.extra_metadata["conformal_alpha"] == pytest.approx(0.15)
