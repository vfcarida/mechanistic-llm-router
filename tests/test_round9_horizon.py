"""Verification suite for KV-Cache Policy, Martingale E-Process & L2 Robustness Certification.

Covers:
- Mechanistic KV-Cache Budget & Heavy-Hitter Token Eviction Policy
  (MechanisticKVCachePolicy, KVCacheAllocationPlan)
- Martingale E-Process & Anytime-Valid Sequential Risk Monitor
  (BettingMartingaleMonitor, EProcessStatus)
- Local Jacobian Sensitivity & L2 Decision Robustness Certifier
  (LocalRobustnessCertifier, RobustnessCertificate)
- Gateway Endpoints (/v1/routing/kv-cache-plan,
  /v1/routing/robustness-certificate, /v1/routing/sla-eprocess)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mechanistic_router.config import RouterConfig
from mechanistic_router.gateway.server import create_app
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.bilinear_probe import LowRankBilinearProbe
from mechanistic_router.probing.eprocess_monitor import (
    BettingMartingaleMonitor,
    EProcessStatus,
)
from mechanistic_router.probing.kv_cache_policy import (
    KVCacheAllocationPlan,
    MechanisticKVCachePolicy,
)
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.probing.robustness_certifier import (
    LocalRobustnessCertifier,
    RobustnessCertificate,
)


# ---------------------------------------------------------------------------
# Mechanistic KV-Cache Budget & Heavy-Hitter Token Eviction Policy
# ---------------------------------------------------------------------------
def test_mechanistic_kv_cache_policy_h2o_and_sinks() -> None:
    """Verifies Attention Sink retention, Heavy-Hitter selection, and perturbation bounds."""
    with pytest.raises(ValueError, match="num_sink_tokens"):
        MechanisticKVCachePolicy(num_sink_tokens=-1)
    with pytest.raises(ValueError, match="recent_window_size"):
        MechanisticKVCachePolicy(recent_window_size=-1)
    with pytest.raises(ValueError, match="Retention ratios"):
        MechanisticKVCachePolicy(min_retention_ratio=0.8, max_retention_ratio=0.4)

    policy = MechanisticKVCachePolicy(
        num_sink_tokens=2,
        recent_window_size=2,
        min_retention_ratio=0.30,
        max_retention_ratio=1.0,
    )

    # Dynamic retention ratio rises monotonically with complexity and entropy
    r_low = policy.compute_dynamic_retention_ratio(
        prob_complex=0.10, mean_head_entropy=0.15, conformal_alpha=0.20
    )
    r_high = policy.compute_dynamic_retention_ratio(
        prob_complex=0.90, mean_head_entropy=0.85, conformal_alpha=0.05
    )
    assert 0.30 <= r_low < r_high <= 1.0

    # Construct synthetic 2-head, 10-token causal attention matrix where token 4 and 5
    # are strong Heavy-Hitters across all subsequent queries.
    seq_len = 10
    attn = np.zeros((2, seq_len, seq_len), dtype=np.float64)
    for t in range(seq_len):
        attn[:, t, 0] = 0.25  # attention sink
        if t >= 5:
            attn[:, t, 4] = 0.40  # heavy hitter 1
            attn[:, t, 5] = 0.25  # heavy hitter 2
            attn[:, t, t] = 0.10
        else:
            attn[:, t, : t + 1] = 1.0 / float(t + 1)

    scores = policy.compute_token_importance_scores(torch.tensor(attn))
    assert scores.shape == (seq_len,)
    assert np.isclose(np.sum(scores), 1.0, atol=1e-6)
    assert scores[4] > scores[2]
    assert scores[5] > scores[3]

    # Plan with explicit 60% retention (6 of 10 tokens: 2 sinks [0,1], 2 recent [8,9], 2 HH [4,5])
    plan = policy.plan_eviction(attn, explicit_retention_ratio=0.60)
    assert isinstance(plan, KVCacheAllocationPlan)
    assert plan.seq_len == 10
    assert plan.num_retained_tokens == 6
    assert plan.num_evicted_tokens == 4
    assert plan.sink_token_indices == [0, 1]
    assert plan.recent_window_indices == [8, 9]
    assert plan.heavy_hitter_indices == [4, 5]
    assert plan.retained_token_indices == [0, 1, 4, 5, 8, 9]
    assert plan.evicted_token_indices == [2, 3, 6, 7]
    assert 0.80 <= plan.retained_attention_mass <= 1.0
    assert 0.0 <= plan.perturbation_error_bound <= 2.0 * (1.0 - plan.retained_attention_mass) + 1e-5
    assert np.isclose(plan.memory_saved_ratio, 0.40, atol=1e-6)
    assert "retained_token_indices" in plan.to_dict()

    # Plan from 1D and 2D hidden states
    rng = np.random.default_rng(42)
    h_1d = rng.normal(size=(64,))
    plan_1d = policy.plan_from_hidden_states(h_1d, prob_complex=0.45)
    assert plan_1d.seq_len >= 4
    assert plan_1d.num_retained_tokens + plan_1d.num_evicted_tokens == plan_1d.seq_len

    # Invalid attention shapes raise ValueError
    with pytest.raises(ValueError, match="Expected 2D"):
        policy.compute_token_importance_scores(np.ones((2, 2, 2, 2, 2)))
    with pytest.raises(ValueError, match="non-empty square"):
        policy.compute_token_importance_scores(np.ones((2, 4, 5)))


# ---------------------------------------------------------------------------
# Martingale E-Process & Anytime-Valid Sequential Risk Monitor
# ---------------------------------------------------------------------------
def test_betting_martingale_eprocess_ville_inequality_and_circuit_breaker() -> None:
    """Verifies supermartingale behavior under H0 and exponential growth under SLA violation H1."""
    with pytest.raises(ValueError, match="alpha"):
        BettingMartingaleMonitor(alpha=1.0)
    with pytest.raises(ValueError, match="significance_delta"):
        BettingMartingaleMonitor(significance_delta=0.0)
    with pytest.raises(ValueError, match="max_bet_multiplier"):
        BettingMartingaleMonitor(max_bet_multiplier=1.0)
    with pytest.raises(ValueError, match="min_exploration_bet"):
        BettingMartingaleMonitor(min_exploration_bet=-0.1)

    monitor = BettingMartingaleMonitor(alpha=0.10, significance_delta=0.05)
    init_status = monitor.status()
    assert isinstance(init_status, EProcessStatus)
    assert init_status.e_value_low == 1.0
    assert init_status.anytime_p_value == 1.0
    assert not init_status.sla_violated
    assert not init_status.circuit_breaker_tripped

    # 1. Under H0 (incurred loss <= alpha = 0.10), E-value stays bounded below 1/delta = 20
    for _ in range(30):
        status_h0 = monitor.update(incurred_loss_low=0.04, incurred_loss_high=0.05)
    assert status_h0.e_value_low < 1.0
    assert status_h0.e_value_high < 1.0
    assert not status_h0.sla_violated
    assert not status_h0.circuit_breaker_tripped

    # 2. Under H1 (severe SLA violation: incurred loss = 0.85 >> alpha = 0.10),
    # capital grows exponentially and trips the Ville barrier 1/delta = 20.0
    monitor.reset()
    for _ in range(25):
        status_h1 = monitor.update(incurred_loss_low=0.85, incurred_loss_high=0.80)
    assert status_h1.e_value_low >= monitor.rejection_threshold
    assert status_h1.anytime_p_value <= 0.05
    assert status_h1.sla_violated
    assert status_h1.circuit_breaker_tripped
    assert "anytime_p_value" in status_h1.to_dict()

    # Reset restores clean initial state
    reset_status = monitor.reset()
    assert reset_status.num_observations == 0
    assert reset_status.joint_e_value == 1.0
    assert not reset_status.circuit_breaker_tripped


# ---------------------------------------------------------------------------
# Local Jacobian Sensitivity & L2 Decision Robustness Certifier
# ---------------------------------------------------------------------------
def test_local_robustness_certifier_linear_and_bilinear() -> None:
    """Verifies exact hyperplane L2 radius and 2nd-order Taylor bilinear certificate."""
    with pytest.raises(ValueError, match="smoothing_sigma"):
        LocalRobustnessCertifier(smoothing_sigma=0.0)
    with pytest.raises(ValueError, match="num_smoothing_samples"):
        LocalRobustnessCertifier(num_smoothing_samples=4)
    with pytest.raises(ValueError, match="confidence_alpha"):
        LocalRobustnessCertifier(confidence_alpha=0.6)
    with pytest.raises(ValueError, match="min_relative_radius"):
        LocalRobustnessCertifier(min_relative_radius=-0.1)

    rng = np.random.default_rng(123)
    X = rng.normal(size=(80, 16)).astype(np.float64)
    true_w = np.zeros(16, dtype=np.float64)
    true_w[0] = 3.0
    true_w[1] = -2.0
    y = ((X @ true_w + 0.5 * X[:, 0] * X[:, 8]) > 0.0).astype(int)

    lin_probe = LinearActivationProbe(C=1.0)
    lin_probe.fit(X, y)

    bilin_probe = LowRankBilinearProbe(rank=3)
    bilin_probe.fit(X, y)

    certifier = LocalRobustnessCertifier(
        smoothing_sigma=0.10,
        num_smoothing_samples=128,
        confidence_alpha=0.05,
        seed=42,
    )

    # Sample far from boundary (high confidence)
    strong_sample = np.zeros(16, dtype=np.float64)
    strong_sample[0] = 4.0
    strong_sample[1] = -3.0

    lin_cert = certifier.certify(
        activation=strong_sample,
        probe=lin_probe,
        low_threshold=0.35,
        high_threshold=0.70,
    )
    assert isinstance(lin_cert, RobustnessCertificate)
    assert lin_cert.selected_tier == "LLM-Frontier-Oracle"
    assert lin_cert.hessian_spectral_bound == 0.0
    assert lin_cert.certified_l2_radius > 0.5
    assert lin_cert.smoothed_certified_radius > 0.0
    assert lin_cert.is_certified_stable

    # Verify exact mathematical boundary property for LinearActivationProbe:
    # Perturbation delta with ||delta||_2 = 0.95 * certified_l2_radius must not cross boundary
    w_vec = np.asarray(lin_probe.weights, dtype=np.float64)
    unit_adv = -w_vec / np.linalg.norm(w_vec)
    safe_perturbed = strong_sample + 0.95 * lin_cert.certified_l2_radius * unit_adv
    logit_after = float(lin_probe.decision_function(safe_perturbed[None, :])[0])
    assert logit_after > lin_cert.nearest_boundary_logit

    # Test 2nd-order Taylor certificate on LowRankBilinearProbe
    bilin_cert = certifier.certify(
        activation=strong_sample,
        probe=bilin_probe,
        low_threshold=0.35,
        high_threshold=0.70,
    )
    assert bilin_cert.hessian_spectral_bound > 0.0
    assert bilin_cert.certified_l2_radius > 0.0
    assert len(bilin_cert.worst_case_direction_top_dims) == 5
    assert "certified_l2_radius" in bilin_cert.to_dict()


# ---------------------------------------------------------------------------
# Gateway Endpoints (kv-cache-plan, robustness-certificate, sla-eprocess)
# ---------------------------------------------------------------------------
def test_gateway_round9_endpoints_kv_cache_robustness_and_eprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies FastAPI gateway endpoints for KV-cache, robustness certificate, and E-process."""
    monkeypatch.setenv("ROUTER_API_KEY", "test-round9-key")
    app = create_app(RouterConfig(seed=42))
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-round9-key"}

    # 1. Test POST /v1/routing/kv-cache-plan
    kv_resp = client.post(
        "/v1/routing/kv-cache-plan",
        headers=headers,
        json={
            "prompt": "Prove by induction that the sum of the first n odd integers equals n^2.",
            "num_sink_tokens": 2,
            "recent_window_size": 2,
            "explicit_retention_ratio": 0.65,
        },
    )
    assert kv_resp.status_code == 200
    kv_data = kv_resp.json()
    assert kv_data["seq_len"] >= 4
    assert kv_data["num_retained_tokens"] + kv_data["num_evicted_tokens"] == kv_data["seq_len"]
    assert 0.0 <= kv_data["retained_attention_mass"] <= 1.0
    assert 0.0 <= kv_data["perturbation_error_bound"] <= 1.0

    # 2. Test POST /v1/routing/robustness-certificate (linear & bilinear modes)
    rob_lin_resp = client.post(
        "/v1/routing/robustness-certificate",
        headers=headers,
        json={
            "prompt": "Summarize this short status update email in one sentence.",
            "smoothing_sigma": 0.12,
            "num_smoothing_samples": 64,
            "use_bilinear_probe": False,
        },
    )
    assert rob_lin_resp.status_code == 200
    rob_lin = rob_lin_resp.json()
    assert rob_lin["selected_tier"] in MODEL_POOL
    assert rob_lin["certified_l2_radius"] >= 0.0
    assert rob_lin["hessian_spectral_bound"] == 0.0

    rob_bilin_resp = client.post(
        "/v1/routing/robustness-certificate",
        headers=headers,
        json={
            "prompt": "Derive the Hamilton-Jacobi-Bellman equation for stochastic optimal control.",
            "smoothing_sigma": 0.12,
            "num_smoothing_samples": 64,
            "use_bilinear_probe": True,
        },
    )
    assert rob_bilin_resp.status_code == 200
    rob_bilin = rob_bilin_resp.json()
    assert rob_bilin["selected_tier"] in MODEL_POOL
    assert rob_bilin["hessian_spectral_bound"] > 0.0

    # 3. Test POST /v1/routing/feedback coupling with GET /v1/routing/sla-eprocess
    for _ in range(3):
        fb_resp = client.post(
            "/v1/routing/feedback",
            headers=headers,
            json={"incurred_loss_low": 0.02, "incurred_loss_high": 0.01, "step_size": 0.02},
        )
        assert fb_resp.status_code == 200

    eproc_resp = client.get("/v1/routing/sla-eprocess", headers=headers)
    assert eproc_resp.status_code == 200
    eproc_data = eproc_resp.json()
    assert eproc_data["num_observations"] == 3
    assert 0.0 < eproc_data["anytime_p_value"] <= 1.0
    assert not eproc_data["sla_violated"]

    # Reset via query param
    eproc_reset = client.get("/v1/routing/sla-eprocess?reset=true", headers=headers)
    assert eproc_reset.status_code == 200
    assert eproc_reset.json()["num_observations"] == 0
