"""Unit and Integration Tests for Mahalanobis OOD, Trajectory Drift, Steering & Audit Ledger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.evaluation.calibration_pack import (
    load_calibrated_causal_router,
    load_multidomain_calibration_dataset,
)
from mechanistic_router.gateway.server import create_app
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.observability.audit_log import (
    GENESIS_HASH,
    RoutingAuditEntry,
    RoutingAuditLedger,
)
from mechanistic_router.probing.activation_steering import ActivationSteeringEngine
from mechanistic_router.probing.ood_detector import MahalanobisOODDetector
from mechanistic_router.probing.sae_engine import SAEEngine
from mechanistic_router.probing.trajectory_probe import ConversationTrajectoryAnalyzer


@pytest.mark.asyncio
async def test_h13_mahalanobis_ood_detector_and_causal_router_gating(tmp_path: Path) -> None:
    """Verify Mahalanobis OOD detector flags out-of-distribution activations and escalates tier."""
    rng = np.random.default_rng(101)
    x_in_0 = rng.normal(loc=-1.0, scale=0.25, size=(40, 16))
    x_in_1 = rng.normal(loc=1.0, scale=0.25, size=(40, 16))
    x_cal = np.vstack([x_in_0, x_in_1])
    y_cal = np.array([0] * 40 + [1] * 40, dtype=int)

    detector = MahalanobisOODDetector(shrinkage=0.10, quantile_alpha=0.05)
    assert not detector.is_fitted
    with pytest.raises(ValueError, match="not been fitted"):
        _ = detector.threshold
    with pytest.raises(ValueError, match="not been fitted"):
        detector.score(x_cal)
    with pytest.raises(ValueError, match="unfitted"):
        detector.save(tmp_path / "unfitted.npz")

    detector.fit(x_cal, labels=y_cal)
    assert detector.is_fitted
    assert detector.threshold > 0.0

    in_sample = np.full(16, -1.0, dtype=np.float64)
    ood_sample = np.full(16, 12.0, dtype=np.float64)
    assert detector.is_ood(in_sample) is False
    assert detector.is_ood(ood_sample) is True
    batch_flags = detector.is_ood(np.vstack([in_sample, ood_sample]))
    assert isinstance(batch_flags, np.ndarray) and list(batch_flags) == [False, True]

    # Also test unlabeled fit and persistence
    unlabeled_det = MahalanobisOODDetector(shrinkage=0.2, quantile_alpha=0.10).fit(x_cal)
    ckpt = tmp_path / "mahalanobis_ood.npz"
    unlabeled_det.save(ckpt)
    loaded_det = MahalanobisOODDetector.load(ckpt)
    assert np.isclose(loaded_det.threshold, unlabeled_det.threshold)

    # Integration with CausalProbeRouter: OOD forces escalation to LLM-Frontier-Oracle
    router = load_calibrated_causal_router(model_pool=MODEL_POOL)
    cases = load_multidomain_calibration_dataset(n_samples=60, seed=42)
    router.fit_ood_detector([c.prompt for c in cases], quantile_alpha=0.05)
    assert router.ood_detector is not None and router.ood_detector.is_fitted

    # Artificially tighten OOD threshold to force OOD detection on routine prompt
    router.ood_detector._threshold = 1e-6
    dec_ood = await router.route("What is my current checking account balance?")
    assert dec_ood.selected_model == "LLM-Frontier-Oracle"
    meta = dec_ood.signals["LLM-Frontier-Oracle"].extra_metadata
    assert meta["is_ood"] is True
    assert meta["abstention_escalated"] is True
    assert meta["mahalanobis_distance"] is not None

    # Validation checks
    with pytest.raises(ValueError, match="shrinkage"):
        MahalanobisOODDetector(shrinkage=0.0)
    with pytest.raises(ValueError, match="quantile_alpha"):
        MahalanobisOODDetector(quantile_alpha=1.0)
    with pytest.raises(ValueError, match="2D array"):
        detector.fit(np.ones(5))
    with pytest.raises(ValueError, match="labels length"):
        detector.fit(x_cal, labels=[0, 1])
    with pytest.raises(ValueError, match="Expected feature_dim"):
        detector.score(np.ones(4))
    with pytest.raises(FileNotFoundError):
        MahalanobisOODDetector.load(tmp_path / "missing_ood.npz")


def test_h14_conversation_trajectory_analyzer() -> None:
    """Verify multi-turn trajectory drift and spectral effective dimensionality detection."""
    analyzer = ConversationTrajectoryAnalyzer(
        drift_threshold=0.35,
        d_eff_threshold=1.50,
        recency_decay=0.70,
    )

    single = analyzer.analyze_turns(["What is my current account balance?"])
    assert single.num_turns == 1
    assert single.mean_cosine_drift == 0.0
    assert single.saturation_escalated is False

    escalating_turns = [
        "What is 15% of 240?",
        "Solve the system of linear equations 3x + 2y = 16 and 5x - 3y = 14.",
        "Derive the closed-form solution for the stochastic differential "
        "equation using Ito's Lemma.",
    ]
    multi = analyzer.analyze_turns(escalating_turns)
    assert multi.num_turns == 3
    assert multi.max_cosine_drift > 0.0
    assert multi.trajectory_d_eff >= 1.0
    assert multi.saturation_escalated is True
    assert multi.pooled_activation.shape == (96,)

    # Direct 2D numpy array trajectory input
    orthogonal_turns = np.eye(3, 16, dtype=np.float32)
    ortho_metrics = analyzer.analyze_turns(orthogonal_turns)
    assert np.isclose(ortho_metrics.mean_cosine_drift, 1.0)
    assert ortho_metrics.saturation_escalated is True

    # Validation checks
    with pytest.raises(ValueError, match="drift_threshold"):
        ConversationTrajectoryAnalyzer(drift_threshold=0.0)
    with pytest.raises(ValueError, match="d_eff_threshold"):
        ConversationTrajectoryAnalyzer(d_eff_threshold=0.5)
    with pytest.raises(ValueError, match="recency_decay"):
        ConversationTrajectoryAnalyzer(recency_decay=0.0)
    with pytest.raises(ValueError, match="at least one non-empty"):
        analyzer.analyze_turns(["   "])
    with pytest.raises(ValueError, match="shape"):
        analyzer.analyze_turns(np.ones(8))


def test_h15_activation_steering_and_sae_clamping() -> None:
    """Verify Contrastive Activation Addition (CAA), min_flip_gamma, and SAE feature clamping."""
    rng = np.random.default_rng(77)
    neg_acts = rng.normal(loc=-1.5, scale=0.3, size=(30, 16)).astype(np.float32)
    pos_acts = rng.normal(loc=1.5, scale=0.3, size=(30, 16)).astype(np.float32)

    engine = ActivationSteeringEngine()
    with pytest.raises(ValueError, match="No steering_direction"):
        _ = engine.steering_direction

    direction = engine.fit_contrastive_direction(pos_acts, neg_acts)
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)

    router = load_calibrated_causal_router(model_pool=MODEL_POOL)
    routine_act = router.extractor.extract_one("What is my current checking account balance?")

    # Evaluate counterfactual sensitivity along the probe's own direction vector
    probe_Engine = ActivationSteeringEngine(steering_direction=router.probe.direction_vector)
    cf_res = probe_Engine.evaluate_counterfactual_sensitivity(
        activation=routine_act,
        probe=router.probe,
        gamma=8.0,
        target_threshold=0.5,
    )
    assert cf_res.steered_prob > cf_res.original_prob
    assert cf_res.decision_flipped is True

    # Verify exact boundary crossing at min_flip_gamma
    h_at_boundary = probe_Engine.steer_activation(routine_act, gamma=cf_res.min_flip_gamma)
    prob_at_boundary = float(router.probe.predict_proba(h_at_boundary[None, :])[0])
    assert np.isclose(prob_at_boundary, 0.5, atol=1e-4)

    # Error-preserving SAE feature clamping: clamping to original z yields exact original h
    sae = SAEEngine(d_in=len(routine_act), d_sae=128, top_k=8, seed=42)
    with torch.no_grad():
        _, z_orig = sae(torch.from_numpy(routine_act[None, :]).float())
    orig_val_0 = float(z_orig[0, 0].item())
    clamped_noop = probe_Engine.clamp_sae_features(routine_act, sae, {0: orig_val_0})
    np.testing.assert_allclose(clamped_noop, routine_act, atol=1e-5)

    clamped_boost = probe_Engine.clamp_sae_features(routine_act, sae, {0: orig_val_0 + 5.0})
    assert not np.allclose(clamped_boost, routine_act)

    # Validation checks
    with pytest.raises(ValueError, match="zero norm"):
        ActivationSteeringEngine(steering_direction=np.zeros(8))
    with pytest.raises(ValueError, match="2D arrays"):
        engine.fit_contrastive_direction(np.ones(8), np.ones(8))
    with pytest.raises(ValueError, match="non-empty"):
        engine.fit_contrastive_direction(np.empty((0, 8)), np.ones((2, 8)))
    with pytest.raises(ValueError, match="Activation dim"):
        probe_Engine.steer_activation(np.ones(4), gamma=1.0)
    with pytest.raises(ValueError, match="out of bounds"):
        probe_Engine.clamp_sae_features(routine_act, sae, {999: 1.0})
    with pytest.raises(ValueError, match="target_threshold"):
        probe_Engine.evaluate_counterfactual_sensitivity(
            routine_act, router.probe, gamma=1.0, target_threshold=1.0
        )


def test_h16_routing_audit_ledger_and_gateway_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SHA-256 hash chain integrity, tamper detection, and GET /v1/routing/audit-ledger."""
    ledger = RoutingAuditLedger()
    assert len(ledger) == 0
    assert ledger.head_hash == GENESIS_HASH
    assert ledger.export_entries(limit=0) == []

    e1 = ledger.append_decision(
        prompt="What is my checking balance?",
        selected_model="SLM-BERTau-Local",
        strategy_used="CausalProbeRouter",
        estimated_cost_usd=0.0002,
        cost_saved_usd=0.0298,
        conformal_alpha=0.10,
        sae_features=[1, 7],
    )
    e2 = ledger.append_decision(
        prompt="Derive stochastic PDE solution.",
        selected_model="LLM-Frontier-Oracle",
        strategy_used="CausalProbeRouter",
        estimated_cost_usd=0.0300,
        cost_saved_usd=0.0,
        conformal_alpha=0.10,
        sae_features=[4, 19],
    )
    assert e1.prev_hash == GENESIS_HASH
    assert e2.prev_hash == e1.entry_hash
    valid, count = ledger.verify_integrity()
    assert valid is True and count == 2

    # Tamper detection: corrupting an entry invalidates the chain
    tampered = RoutingAuditEntry(
        seq_id=e1.seq_id,
        timestamp_utc=e1.timestamp_utc,
        prompt_sha256=e1.prompt_sha256,
        selected_model="LLM-Frontier-Oracle",  # tampered!
        strategy_used=e1.strategy_used,
        estimated_cost_usd=e1.estimated_cost_usd,
        cost_saved_usd=e1.cost_saved_usd,
        conformal_alpha=e1.conformal_alpha,
        sae_features=e1.sae_features,
        prev_hash=e1.prev_hash,
        entry_hash=e1.entry_hash,
    )
    ledger._entries[0] = tampered
    valid_after, failed_idx = ledger.verify_integrity()
    assert valid_after is False and failed_idx == 0

    # Gateway endpoint verification
    monkeypatch.setenv("ROUTER_API_KEY", "audit-test-key")
    app = create_app(DEFAULT_CONFIG.model_copy())
    client = TestClient(app)
    headers = {"Authorization": "Bearer audit-test-key"}

    client.post(
        "/v1/routing/inspect",
        headers=headers,
        json={"prompt": "Audit ledger test prompt", "strategy": "causal-probe-auto"},
    )
    resp_audit = client.get("/v1/routing/audit-ledger?limit=10", headers=headers)
    assert resp_audit.status_code == 200
    audit_body = resp_audit.json()
    assert audit_body["chain_valid"] is True
    assert audit_body["verified_entries"] == 1
    assert len(audit_body["entries"]) == 1
    assert audit_body["entries"][0]["prev_hash"] == GENESIS_HASH
