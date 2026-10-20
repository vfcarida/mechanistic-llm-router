"""Verification Test Suite for Federated Conformal, Attention Probing, Cache & Drift Monitor.

Covers:
- Differentially Private Federated Conformal Calibration
  (FederatedConformalAggregator, DPClientCalibrationSummary, FederatedCalibrationResult).
- Attention-Head Routing Entropy & Induction Circuit Probing
  (AttentionCircuitProbe, AttentionProbeSignals, MechanisticRouter integration).
- Semantic-Mechanistic Activation Cache with SAE Circuit Safety
  (MechanisticActivationCache, /v1/chat/completions X-Mechanistic-Cache, /v1/routing/cache-stats).
- Population-Level Activation Drift Monitor via Kernel MMD &
  Counterfactual Explainability Endpoint (PopulationDriftMonitor, /v1/routing/counterfactual).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.gateway.activation_cache import MechanisticActivationCache
from mechanistic_router.gateway.server import create_app
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.attention_probe import AttentionCircuitProbe
from mechanistic_router.probing.conformal import ConformalRiskController
from mechanistic_router.probing.drift_monitor import PopulationDriftMonitor
from mechanistic_router.probing.federated_conformal import FederatedConformalAggregator
from mechanistic_router.routers.mechanistic import MechanisticRouter


def test_h17_federated_conformal_dp_calibration_and_controller_update() -> None:
    """Verify Gaussian and Laplace DP federated conformal calibration across silos."""
    rng = np.random.default_rng(101)
    scores_1 = rng.uniform(0.0, 1.0, size=120)
    cheap_loss_1 = (scores_1 > 0.35).astype(np.float64) * 0.8

    scores_2 = rng.uniform(0.0, 1.0, size=180)
    cheap_loss_2 = (scores_2 > 0.40).astype(np.float64) * 0.7
    mid_loss_2 = (scores_2 > 0.70).astype(np.float64) * 0.4

    summary_1 = FederatedConformalAggregator.compute_client_dp_summary(
        client_id="hospital-silo-a",
        scores=scores_1,
        cheap_losses=cheap_loss_1,
        epsilon=4.0,
        delta=1e-5,
        grid_size=51,
        seed=11,
    )
    summary_2 = FederatedConformalAggregator.compute_client_dp_summary(
        client_id="bank-silo-b",
        scores=scores_2,
        cheap_losses=cheap_loss_2,
        mid_losses=mid_loss_2,
        epsilon=4.0,
        delta=0.0,  # Pure epsilon-DP Laplace mechanism
        grid_size=51,
        seed=22,
    )

    assert len(summary_1.lambda_grid) == 51
    assert summary_1.to_dict()["client_id"] == "hospital-silo-a"
    # Verify monotone non-decreasing property of privatized risk curves
    diffs_1 = np.diff(summary_1.noisy_cheap_risk_curve)
    assert np.all(diffs_1 >= -1e-12)

    result = FederatedConformalAggregator.aggregate(
        summaries=[summary_1, summary_2],
        alpha=0.15,
        confidence_level=0.90,
    )
    assert 0.0 <= result.lambda_1 <= result.lambda_2 <= 1.0
    assert result.total_samples == 300
    assert result.num_clients == 2
    assert result.to_dict()["num_clients"] == 2

    ctrl = ConformalRiskController(alpha=0.10)
    FederatedConformalAggregator.apply_to_controller(ctrl, result)
    assert ctrl.is_calibrated
    assert ctrl.low_threshold == pytest.approx(result.lambda_1)
    assert ctrl.high_threshold == pytest.approx(result.lambda_2)


def test_h17_federated_conformal_validation_errors() -> None:
    """Verify input validation in FederatedConformalAggregator."""
    scores = np.array([0.1, 0.5, 0.9])
    losses = np.array([0.0, 0.2, 1.0])

    with pytest.raises(ValueError, match="client_id"):
        FederatedConformalAggregator.compute_client_dp_summary("", scores, losses)
    with pytest.raises(ValueError, match="epsilon"):
        FederatedConformalAggregator.compute_client_dp_summary("silo", scores, losses, epsilon=0.0)
    with pytest.raises(ValueError, match="delta"):
        FederatedConformalAggregator.compute_client_dp_summary("silo", scores, losses, delta=1.0)
    with pytest.raises(ValueError, match="grid_size"):
        FederatedConformalAggregator.compute_client_dp_summary("silo", scores, losses, grid_size=5)
    with pytest.raises(ValueError, match="At least 2"):
        FederatedConformalAggregator.compute_client_dp_summary(
            "silo", np.array([0.5]), np.array([0.1])
        )
    with pytest.raises(ValueError, match="identical length"):
        FederatedConformalAggregator.compute_client_dp_summary("silo", scores, np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="identical length"):
        FederatedConformalAggregator.compute_client_dp_summary(
            "silo", scores, losses, mid_losses=np.array([0.1])
        )
    with pytest.raises(ValueError, match="non-empty"):
        FederatedConformalAggregator.aggregate([])


@pytest.mark.asyncio
async def test_h18_attention_circuit_probe_and_mechanistic_router() -> None:
    """Verify AttentionCircuitProbe on exact induction patterns, sinks, and MechanisticRouter."""
    probe = AttentionCircuitProbe(hidden_dim=64, num_heads=4, seed=42)

    # Construct a 2-head attention matrix of sequence length 6 with repeated token pattern
    # Tokens: [10, 20, 30, 10, 20, 30]
    attn = np.zeros((2, 6, 6), dtype=np.float64)
    for t in range(6):
        # Head 0: strong attention sink at token 0
        attn[0, t, 0] = 1.0
        # Head 1: strong induction head attending to p+1 when repeated tokens occur
        attn[1, t, max(0, t - 1)] = 1.0
    # At t=3 (token 10 matches p=0 -> p+1=1), t=4 (p=1 -> p+1=2), t=5 (p=2 -> p+1=3)
    attn[1, 3, :] = [0.0, 0.9, 0.1, 0.0, 0.0, 0.0]
    attn[1, 4, :] = [0.0, 0.0, 0.95, 0.05, 0.0, 0.0]
    attn[1, 5, :] = [0.0, 0.0, 0.0, 0.92, 0.08, 0.0]

    signals = probe.analyze_attention_weights(attn, token_ids=[10, 20, 30, 10, 20, 30])
    assert 0.0 <= signals.mean_head_entropy <= 1.0
    assert signals.attention_sink_ratio > 0.3
    assert signals.induction_head_score > 0.8
    assert signals.max_induction_head_idx == 1
    assert 0.0 <= signals.structural_complexity_score <= 1.0
    assert signals.to_dict()["num_heads"] == 2

    # 4D and 2D attention weight inputs + 1D and mismatched hidden_dim inputs
    sig_4d = probe.analyze_attention_weights(torch.tensor(attn).unsqueeze(0))
    assert sig_4d.num_heads == 2
    sig_2d = probe.analyze_attention_weights(attn[0])
    assert sig_2d.num_heads == 1
    sig_1d_hidden = probe.analyze_hidden_states(np.ones(32, dtype=np.float64))
    assert sig_1d_hidden.seq_len == 1

    # Single-token edge case
    single_sig = probe.analyze_hidden_states(torch.randn(1, 1, 64))
    assert single_sig.seq_len == 1
    assert single_sig.attention_sink_ratio == 1.0

    with pytest.raises(ValueError, match="hidden_dim"):
        AttentionCircuitProbe(hidden_dim=0, num_heads=4)
    with pytest.raises(ValueError, match="divisible"):
        AttentionCircuitProbe(hidden_dim=65, num_heads=4)

    # Verify MechanisticRouter integration populates attention telemetry
    enc = SharedTrunkEncoder(DEFAULT_CONFIG)
    mech_router = MechanisticRouter(enc, MODEL_POOL, DEFAULT_CONFIG)
    decision = await mech_router.route(
        "Analyze the induction circuit and spectral entropy of this repeated sequence."
    )
    first_sig = next(iter(decision.signals.values()))
    assert "attention_mean_entropy" in first_sig.extra_metadata
    assert "induction_head_score" in first_sig.extra_metadata
    assert "structural_complexity_score" in first_sig.extra_metadata


def test_h19_mechanistic_activation_cache_triple_gate() -> None:
    """Verify MechanisticActivationCache enforces cosine, SAE circuit, and uncertainty gates."""
    cache = MechanisticActivationCache(
        max_entries=2,
        similarity_threshold=0.98,
        max_uncertainty_threshold=0.70,
        ttl_seconds=60.0,
    )
    vec_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    vec_near = np.array([0.999, 0.04, 0.0, 0.0], dtype=np.float64)

    # 1. Reject insertion when is_ood=True, uncertainty > threshold, or zero norm
    assert not cache.put(
        vec_a,
        selected_model="SLM-BERTau-Local",
        estimated_cost_usd=0.0001,
        is_ood=True,
    )
    assert not cache.put(
        vec_a,
        selected_model="SLM-BERTau-Local",
        estimated_cost_usd=0.0001,
        epistemic_uncertainty=0.85,
    )
    assert not cache.put(
        np.zeros(4),
        selected_model="SLM-BERTau-Local",
        estimated_cost_usd=0.0001,
    )

    # 2. Valid insertion
    assert cache.put(
        vec_a,
        selected_model="SLM-BERTau-Local",
        estimated_cost_usd=0.0001,
        circuit_type="formal_math_and_logic",
        sae_features=[1, 4],
        complexity_score=0.20,
        epistemic_uncertainty=0.25,
        response_content="Cached answer 42",
    )

    # 3. Reject lookup when SAE circuit_type mismatches despite >0.99 cosine similarity
    miss_circuit = cache.lookup(
        vec_near,
        circuit_type="code_and_algorithmic_synthesis",
        complexity_score=0.20,
        epistemic_uncertainty=0.25,
    )
    assert not miss_circuit.hit
    assert miss_circuit.rejection_reason == "circuit_or_complexity_mismatch"

    # Reject lookup when query is OOD or zero-norm
    assert not cache.lookup(vec_near, is_ood=True).hit
    assert not cache.lookup(np.zeros(4)).hit

    # 4. Hit when all three mechanistic gates agree
    hit_res = cache.lookup(
        vec_near,
        circuit_type="formal_math_and_logic",
        complexity_score=0.22,
        epistemic_uncertainty=0.25,
    )
    assert hit_res.hit
    assert hit_res.entry is not None
    assert hit_res.entry.response_content == "Cached answer 42"

    # Test LRU eviction when inserting beyond max_entries=2
    cache.put(np.array([0.0, 1.0, 0.0, 0.0]), "SLM-BERTau-Local", 0.0001)
    cache.put(np.array([0.0, 0.0, 1.0, 0.0]), "SLM-BERTau-Local", 0.0001)
    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["hits"] == 1
    assert stats["circuit_mismatch_rejections"] == 1


def test_h20_population_drift_monitor_mmd_and_auto_tighten() -> None:
    """Verify Kernel MMD^2 permutation test detects covariate shift and tightens conformal alpha."""
    rng = np.random.default_rng(77)
    ref_acts = rng.normal(loc=0.0, scale=1.0, size=(40, 16))
    in_dist_window = rng.normal(loc=0.0, scale=1.0, size=(30, 16))
    shifted_window = rng.normal(loc=2.5, scale=1.0, size=(30, 16))

    monitor = PopulationDriftMonitor(
        reference_activations=ref_acts,
        window_capacity=50,
        alpha_test=0.05,
        min_mmd_threshold=0.02,
        num_permutations=80,
        seed=42,
    )

    with pytest.raises(ValueError, match="At least 4"):
        monitor.evaluate_drift()

    # 1. Stationary window -> no drift detected
    for row in in_dist_window:
        monitor.record_activation(row)
    res_stationary = monitor.evaluate_drift(base_conformal_alpha=0.10)
    assert not res_stationary.is_drift_detected
    assert res_stationary.recommended_alpha == pytest.approx(0.10)

    # 2. Shifted window -> drift detected and controller tightened
    ctrl = ConformalRiskController(alpha=0.10)
    scores = np.linspace(0.0, 1.0, 50)
    losses = (scores > 0.5).astype(np.float64)
    ctrl.calibrate(scores, losses)
    initial_low = ctrl.low_threshold

    res_shifted = monitor.auto_tighten_conformal_controller(ctrl, window_activations=shifted_window)
    assert res_shifted.is_drift_detected
    assert res_shifted.mmd_squared > 0.10
    assert res_shifted.p_value <= 0.05
    assert ctrl.effective_alpha < 0.10
    assert ctrl.low_threshold < initial_low
    assert res_shifted.to_dict()["is_drift_detected"] is True
    monitor.clear_window()


@patch("litellm.acompletion", new_callable=AsyncMock)
def test_h19_h20_gateway_cache_and_counterfactual_endpoints(
    mock_acompletion: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify counterfactual, cache-stats, and X-Mechanistic-Cache header."""
    monkeypatch.setenv("ROUTER_API_KEY", "test-round6-key")
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-r6"
    mock_choice = MagicMock()
    mock_choice.message.content = "Computed deterministic response."
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=8, completion_tokens=8, total_tokens=16)
    mock_acompletion.return_value = mock_response

    test_app = create_app()
    test_app.state.activation_cache.clear()
    client = TestClient(test_app)
    headers = {"Authorization": "Bearer test-round6-key"}

    # 1. POST /v1/routing/counterfactual with prompt + multi-turn messages
    cf_resp = client.post(
        "/v1/routing/counterfactual",
        json={
            "prompt": "Prove that every finite division ring is a field (Wedderburn's theorem).",
            "messages": [
                {"role": "user", "content": "What is a ring in algebra?"},
                {
                    "role": "assistant",
                    "content": "A set equipped with addition and multiplication.",
                },
                {
                    "role": "user",
                    "content": (
                        "Now prove Wedderburn's little theorem using cyclotomic polynomials."
                    ),
                },
            ],
            "target_probability": 0.5,
        },
        headers=headers,
    )
    assert cf_resp.status_code == 200
    cf_data = cf_resp.json()
    assert cf_data["steered_probability"] == pytest.approx(0.5, abs=1e-4)
    assert isinstance(cf_data["minimum_flip_gamma"], float)
    assert len(cf_data["active_sae_features"]) > 0
    assert cf_data["trajectory_metrics"] is not None
    assert cf_data["trajectory_metrics"]["num_turns"] == 3

    # Missing prompt and messages returns 400
    bad_cf = client.post("/v1/routing/counterfactual", json={}, headers=headers)
    assert bad_cf.status_code == 400

    # 2. /v1/chat/completions with X-Mechanistic-Cache: true -> MISS then HIT
    chat_payload = {
        "model": "causal-probe-auto",
        "messages": [{"role": "user", "content": "What is my checking account balance?"}],
    }
    cache_headers = {**headers, "X-Mechanistic-Cache": "true"}

    r1 = client.post("/v1/chat/completions", json=chat_payload, headers=cache_headers)
    assert r1.status_code == 200
    assert r1.headers.get("x-mechanistic-cache") == "MISS"
    assert mock_acompletion.call_count == 1

    r2 = client.post("/v1/chat/completions", json=chat_payload, headers=cache_headers)
    assert r2.status_code == 200
    assert r2.headers.get("x-mechanistic-cache") == "HIT"
    assert "MechanisticCacheHit" in r2.json()["router_strategy"]
    # Upstream provider was NOT called a second time
    assert mock_acompletion.call_count == 1

    # 3. GET /v1/routing/cache-stats
    stats_resp = client.get("/v1/routing/cache-stats", headers=headers)
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert stats_data["hits"] >= 1
    assert stats_data["size"] >= 1
