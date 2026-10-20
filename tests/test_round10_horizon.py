"""Verification suite for Multi-Modal Probing, Doubly Robust Off-Policy Evaluation,
Mechanistic Hallucination Detection, and Bures-Wasserstein Subspace Alignment.

Covers:
- Multi-Modal Vision-Language Cross-Attention & Patch-Token Residual Probing
  (MultiModalActivationProbe, MultiModalProbeTelemetry)
- Inverse Propensity Scoring (IPS) & Doubly Robust Counterfactual Policy Evaluator
  (DoublyRobustPolicyEvaluator, LoggedRoutingSample, OffPolicyEvaluationResult)
- Mechanistic Token-Level Hallucination & Epistemic Divergence Detector
  (MechanisticHallucinationDetector, HallucinationRiskReport)
- Cross-Lingual & Cross-Family Bures-Wasserstein Optimal Transport Subspace Alignment
  (BuresWassersteinAligner, WassersteinAlignmentTelemetry)
- Gateway Endpoints (/v1/routing/off-policy-evaluate, /v1/routing/hallucination-check,
  /v1/routing/multimodal-inspect, /v1/routing/wasserstein-align)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mechanistic_router
from mechanistic_router.config import RouterConfig
from mechanistic_router.evaluation.off_policy_evaluator import (
    DoublyRobustPolicyEvaluator,
    LoggedRoutingSample,
    OffPolicyEvaluationResult,
)
from mechanistic_router.gateway.server import create_app
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.hallucination_detector import (
    HallucinationRiskReport,
    MechanisticHallucinationDetector,
)
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.probing.multimodal_probe import (
    MultiModalActivationProbe,
    MultiModalProbeTelemetry,
)
from mechanistic_router.probing.wasserstein_alignment import (
    BuresWassersteinAligner,
    WassersteinAlignmentTelemetry,
)


# ---------------------------------------------------------------------------
# Multi-Modal Vision-Language Cross-Attention & Patch-Token Probing
# ---------------------------------------------------------------------------
def test_multimodal_activation_probe_fastv_and_prumerge() -> None:
    """Verifies cross-modal attention, FastV/PruMerge patch pruning, and d_eff^{vision}."""
    with pytest.raises(ValueError, match="patch_retention_ratio"):
        MultiModalActivationProbe(patch_retention_ratio=0.0)
    with pytest.raises(ValueError, match="vision_weight"):
        MultiModalActivationProbe(vision_weight=-0.1, text_weight=0.5)

    probe = MultiModalActivationProbe(
        patch_retention_ratio=0.50,
        vision_weight=0.50,
        text_weight=0.50,
    )

    rng = np.random.default_rng(42)
    # 12 vision patches, 4 text tokens, hidden_dim = 16
    vision_patches = rng.normal(0.0, 0.2, size=(12, 16))
    text_tokens = rng.normal(0.0, 0.5, size=(4, 16))
    # Plant strong alignment on patches 2 and 7 with text queries
    mean_query = np.mean(text_tokens, axis=0)
    vision_patches[2] = mean_query * 4.0
    vision_patches[7] = mean_query * 3.5

    cross_attn = probe.compute_cross_modal_attention(vision_patches, text_tokens)
    assert cross_attn.shape == (4, 12)
    assert np.allclose(np.sum(cross_attn, axis=1), 1.0, atol=1e-6)

    merged, retained, pruned, recon_err = probe.prune_and_merge_patches(
        vision_patches, text_tokens, retention_ratio=0.50
    )
    assert merged.shape == (6, 16)
    assert len(retained) == 6
    assert len(pruned) == 6
    assert 2 in retained
    assert 7 in retained
    assert 0.0 <= recon_err <= 1.0

    # Fit a simple linear text probe to test integration
    X_train = rng.normal(size=(30, 16))
    y_train = (X_train[:, 0] > 0).astype(int)
    lin_probe = LinearActivationProbe().fit(X_train, y_train)

    telemetry = probe.evaluate(vision_patches, text_tokens, text_probe=lin_probe)
    assert isinstance(telemetry, MultiModalProbeTelemetry)
    assert telemetry.num_vision_patches == 12
    assert telemetry.num_text_tokens == 4
    assert telemetry.vision_effective_dim >= 1.0
    assert 0.0 <= telemetry.cross_modal_alignment <= 1.0
    assert 0.0 <= telemetry.cross_modal_entropy <= 1.0
    assert 0.0 <= telemetry.prob_complex <= 1.0
    assert "vision_effective_dim" in telemetry.to_dict()

    # Test deterministic synthesis helper and error cases
    v_syn, t_syn = MultiModalActivationProbe.synthesize_vision_patches_from_text_activation(
        rng.normal(size=(64,)), num_patches=10, image_complexity_hint=0.7
    )
    assert v_syn.shape[0] == 10
    assert v_syn.shape[1] == t_syn.shape[1]

    with pytest.raises(ValueError, match="num_patches"):
        MultiModalActivationProbe.synthesize_vision_patches_from_text_activation(
            rng.normal(size=(32,)), num_patches=1
        )
    with pytest.raises(ValueError, match="Hidden dimension mismatch"):
        probe.compute_cross_modal_attention(np.zeros((4, 8)), np.zeros((2, 16)))


# ---------------------------------------------------------------------------
# Doubly Robust & SNIPS Counterfactual Policy Evaluator
# ---------------------------------------------------------------------------
def test_doubly_robust_policy_evaluator_and_promotion_gate() -> None:
    """Verifies DM, IPS, SNIPS, DR risk estimation, Kish ESS, and safe promotion gate."""
    with pytest.raises(ValueError, match="tier_costs_usd"):
        DoublyRobustPolicyEvaluator(tier_costs_usd=(0.02, -0.1, 1.50))
    with pytest.raises(ValueError, match="max_importance_weight"):
        DoublyRobustPolicyEvaluator(max_importance_weight=0.0)
    with pytest.raises(ValueError, match="min_ess_ratio"):
        DoublyRobustPolicyEvaluator(min_ess_ratio=0.0)
    with pytest.raises(ValueError, match="confidence_level"):
        DoublyRobustPolicyEvaluator(confidence_level=0.2)

    evaluator = DoublyRobustPolicyEvaluator(
        tier_costs_usd=(0.02, 0.25, 1.50),
        max_importance_weight=15.0,
        min_ess_ratio=0.15,
        confidence_level=0.95,
    )

    rng = np.random.default_rng(99)
    samples: list[LoggedRoutingSample] = []
    for _ in range(80):
        p = float(rng.uniform(0.0, 1.0))
        tier = int(rng.choice([0, 1, 2], p=[0.4, 0.3, 0.3]))
        if tier == 0:
            loss = 0.02 if p < 0.35 else 0.75
        elif tier == 1:
            loss = 0.03 if p < 0.70 else 0.45
        else:
            loss = 0.01
        samples.append(
            LoggedRoutingSample(
                prob_complex=p,
                logged_tier_index=tier,
                logging_propensity=0.33,
                observed_loss=loss,
                observed_cost_usd=(0.02, 0.25, 1.50)[tier],
            )
        )

    # 1. Conservative candidate policy (low=0.20, high=0.45) achieves low DR risk
    safe_res = evaluator.evaluate_thresholds(
        samples=samples,
        candidate_low_threshold=0.20,
        candidate_high_threshold=0.45,
        target_sla_alpha=0.20,
        temperature=0.08,
    )
    assert isinstance(safe_res, OffPolicyEvaluationResult)
    assert safe_res.num_samples == 80
    assert 0.0 <= safe_res.doubly_robust_risk <= 0.20
    assert safe_res.dr_risk_ci_lower <= safe_res.doubly_robust_risk <= safe_res.dr_risk_ci_upper
    assert safe_res.effective_sample_size > 15.0
    assert safe_res.safe_to_promote

    # 2. Reckless candidate policy (low=0.85, high=0.95) delegates to cheap tier -> high risk
    risky_res = evaluator.evaluate_thresholds(
        samples=samples,
        candidate_low_threshold=0.85,
        candidate_high_threshold=0.95,
        target_sla_alpha=0.10,
        temperature=0.05,
    )
    assert risky_res.doubly_robust_risk > safe_res.doubly_robust_risk
    assert risky_res.expected_cost_usd < safe_res.expected_cost_usd
    assert not risky_res.safe_to_promote
    assert "doubly_robust_risk" in risky_res.to_dict()

    with pytest.raises(ValueError, match="non-empty"):
        evaluator.evaluate_thresholds([], 0.3, 0.7)
    with pytest.raises(ValueError, match="Thresholds"):
        evaluator.evaluate_thresholds(samples, 0.8, 0.3)


# ---------------------------------------------------------------------------
# Mechanistic Token-Level Hallucination & Epistemic Divergence Detector
# ---------------------------------------------------------------------------
def test_mechanistic_hallucination_detector_dola_and_ccs() -> None:
    """Verifies DoLa premature-vs-mature layer divergence and conformal hallucination gate."""
    with pytest.raises(ValueError, match="num_layers"):
        MechanisticHallucinationDetector(num_layers=1)
    with pytest.raises(ValueError, match="false_alarm_alpha"):
        MechanisticHallucinationDetector(false_alarm_alpha=1.0)
    with pytest.raises(ValueError, match="divergence_weight"):
        MechanisticHallucinationDetector(divergence_weight=0.0, truthfulness_weight=0.0)

    detector = MechanisticHallucinationDetector(
        num_layers=3,
        false_alarm_alpha=0.10,
    )
    assert not detector.is_fitted

    rng = np.random.default_rng(77)
    # Factual activations: all 3 layer blocks (each of dim 8, total dim 24) are consistent
    base_layer = rng.normal(1.0, 0.2, size=(30, 8))
    X_factual = np.hstack(
        [
            base_layer + rng.normal(0.0, 0.03, size=(30, 8)),
            base_layer + rng.normal(0.0, 0.02, size=(30, 8)),
            base_layer,
        ]
    )

    # Confabulation activations: premature layers strongly oppose final layer + shifted mean
    conf_dir = np.ones((1, 24), dtype=np.float64) * 2.5
    X_confab = (
        np.hstack(
            [
                -base_layer[:15],
                -0.8 * base_layer[:15],
                base_layer[:15],
            ]
        )
        + conf_dir
    )

    detector.fit(X_factual, X_confab)
    assert detector.is_fitted
    assert 0.05 <= detector.conformal_threshold <= 0.99

    rep_factual = detector.evaluate(X_factual[0], prob_complex=0.15)
    rep_confab = detector.evaluate(X_confab[0], prob_complex=0.50)

    assert isinstance(rep_factual, HallucinationRiskReport)
    assert rep_confab.dola_layer_divergence > rep_factual.dola_layer_divergence
    assert rep_confab.truthfulness_projection > rep_factual.truthfulness_projection
    assert rep_confab.hallucination_risk_score > rep_factual.hallucination_risk_score
    assert rep_confab.should_escalate_to_oracle
    assert not rep_factual.should_escalate_to_oracle
    assert "dola_layer_divergence" in rep_confab.to_dict()


# ---------------------------------------------------------------------------
# Cross-Lingual & Cross-Family Bures-Wasserstein Optimal Transport Alignment
# ---------------------------------------------------------------------------
def test_bures_wasserstein_aligner_pullback_and_barycenter(tmp_path: Path) -> None:
    """Verifies Monge covariance transport, zero-shot probe pullback, and Wasserstein barycenter."""
    assert mechanistic_router.__author__ == "Vinicius Caridá"

    with pytest.raises(ValueError, match="reg_lambda"):
        BuresWassersteinAligner(reg_lambda=-0.1)
    with pytest.raises(ValueError, match="eig_floor"):
        BuresWassersteinAligner(eig_floor=0.0)

    rng = np.random.default_rng(123)
    # Target space: N(mu_t, Sigma_t) with anisotropic covariance (60 samples, D=8)
    X_target = rng.normal(0.0, 1.0, size=(60, 8)) @ np.diag(
        [2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4]
    )
    X_target += np.array([1.5, -1.0, 0.5, 0.0, -0.5, 1.0, 0.2, -0.3])

    # Source space: unpaired size (45 samples, D=8) with rotated/scaled covariance + shifted mean
    rot, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    X_source = (
        rng.normal(0.0, 1.0, size=(45, 8)) @ np.diag([0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 0.9, 1.1]) @ rot
    )
    X_source += np.array([-2.0, 2.5, -1.0, 0.8, 1.2, -0.7, -0.4, 0.9])

    aligner = BuresWassersteinAligner(reg_lambda=1e-4)
    assert not aligner.is_fitted
    with pytest.raises(RuntimeError, match="must be fitted"):
        aligner.transform(X_source)

    telemetry = aligner.fit(X_source=X_source, X_target=X_target)
    assert aligner.is_fitted
    assert isinstance(telemetry, WassersteinAlignmentTelemetry)
    assert telemetry.wasserstein_distance > 1.0
    assert telemetry.mean_shift_l2 > 1.0
    assert telemetry.bures_covariance_distance > 0.1
    assert telemetry.covariance_frobenius_residual < 1e-6
    assert telemetry.n_source_samples == 45
    assert telemetry.n_target_samples == 60
    assert "wasserstein_distance" in telemetry.to_dict()

    # Identity alignment check: W_2(X_target, X_target) == 0
    id_aligner = BuresWassersteinAligner(reg_lambda=1e-4)
    id_tel = id_aligner.fit(X_target, X_target)
    assert id_tel.wasserstein_distance < 1e-6

    # Verify 1D and 2D transform + inverse_transform round-trip
    X_aligned = aligner.transform(X_source)
    assert X_aligned.shape == X_source.shape
    X_recovered = aligner.inverse_transform(X_aligned)
    assert np.allclose(X_recovered, X_source, atol=1e-5)

    vec_aligned = aligner.transform(X_source[0])
    assert vec_aligned.shape == (8,)
    assert np.allclose(aligner.inverse_transform(vec_aligned), X_source[0], atol=1e-5)

    # Zero-shot LinearActivationProbe pullback invariance:
    # target_probe.predict_proba(aligner.transform(X)) == pulled_probe.predict_proba(X)
    y_target = (X_target[:, 0] > float(np.median(X_target[:, 0]))).astype(int)
    target_probe = LinearActivationProbe(random_state=42).fit(X_target, y_target)
    pulled_probe = aligner.pullback_probe(target_probe)

    probs_via_transform = target_probe.predict_proba(X_aligned)
    probs_via_pullback = pulled_probe.predict_proba(X_source)
    assert np.allclose(probs_via_transform, probs_via_pullback, atol=1e-5)

    # Persistence round-trip (.npz)
    ckpt = tmp_path / "wasserstein_aligner.npz"
    aligner.save(ckpt)
    loaded = BuresWassersteinAligner.load(ckpt)
    assert loaded.is_fitted
    assert np.allclose(loaded.transform(X_source), X_aligned, atol=1e-6)

    # Álvarez-Esteban Wasserstein Barycenter across 3 language/domain distributions
    X_mid = 0.5 * X_source[:40] + 0.5 * X_target[:40]
    mu_bar, sigma_bar = aligner.compute_barycenter(
        distributions=[X_source, X_target, X_mid],
        weights=[0.3, 0.5, 0.2],
    )
    assert mu_bar.shape == (8,)
    assert sigma_bar.shape == (8, 8)
    assert np.allclose(sigma_bar, sigma_bar.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(sigma_bar)
    assert np.all(eigvals > 0.0)

    # Verify deprecated mock_dataset helper for 100% module coverage with zero leaked warnings
    with pytest.warns(DeprecationWarning, match="mechanistic_router.data.mock_dataset"):
        from mechanistic_router.data.mock_dataset import create_financial_dataset

        legacy_cases = create_financial_dataset(n_samples=10, seed=42)
    assert len(legacy_cases) == 10


# ---------------------------------------------------------------------------
# Gateway Endpoints (off-policy-evaluate, hallucination-check, VLM, wasserstein-align)
# ---------------------------------------------------------------------------
def test_gateway_off_policy_hallucination_multimodal_and_wasserstein_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies FastAPI gateway endpoints for OPE, hallucination, VLM, and Wasserstein alignment."""
    monkeypatch.setenv("ROUTER_API_KEY", "test-convergence-key")
    app = create_app(RouterConfig(seed=42))
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-convergence-key"}

    # 1. Test POST /v1/routing/off-policy-evaluate (using default calibration corpus)
    ope_resp = client.post(
        "/v1/routing/off-policy-evaluate",
        headers=headers,
        json={
            "candidate_low_threshold": 0.25,
            "candidate_high_threshold": 0.65,
            "target_sla_alpha": 0.15,
            "temperature": 0.08,
        },
    )
    assert ope_resp.status_code == 200
    ope_data = ope_resp.json()
    assert ope_data["num_samples"] >= 20
    assert 0.0 <= ope_data["doubly_robust_risk"] <= 1.0
    assert ope_data["effective_sample_size"] > 0.0

    # Invalid threshold order returns HTTP 400
    bad_ope = client.post(
        "/v1/routing/off-policy-evaluate",
        headers=headers,
        json={
            "candidate_low_threshold": 0.80,
            "candidate_high_threshold": 0.30,
        },
    )
    assert bad_ope.status_code == 400

    # 2. Test POST /v1/routing/hallucination-check
    hal_resp = client.post(
        "/v1/routing/hallucination-check",
        headers=headers,
        json={
            "prompt": "Cite the verbatim text of the nonexistent 1789 Quantum Router Treaty.",
            "false_alarm_alpha": 0.10,
        },
    )
    assert hal_resp.status_code == 200
    hal_data = hal_resp.json()
    assert 0.0 <= hal_data["hallucination_risk_score"] <= 1.0
    assert hal_data["num_layers_analyzed"] >= 2
    assert hal_data["recommended_model"] in MODEL_POOL

    # 3. Test POST /v1/routing/multimodal-inspect
    mm_resp = client.post(
        "/v1/routing/multimodal-inspect",
        headers=headers,
        json={
            "prompt": "Extract financial line items and anomalies from this balance sheet chart.",
            "num_vision_patches": 16,
            "patch_retention_ratio": 0.50,
            "image_complexity_hint": 0.65,
        },
    )
    assert mm_resp.status_code == 200
    mm_data = mm_resp.json()
    assert mm_data["num_vision_patches"] == 16
    assert len(mm_data["retained_patch_indices"]) == 8
    assert len(mm_data["pruned_patch_indices"]) == 8
    assert mm_data["selected_model"] in MODEL_POOL

    # 4. Test POST /v1/routing/wasserstein-align
    wass_resp = client.post(
        "/v1/routing/wasserstein-align",
        headers=headers,
        json={
            "source_prompts": [
                "Evaluate liquidity coverage ratio under Basel III for derivative portfolios.",
                "What is the available checking account balance today?",
                "Derive the Hamilton-Jacobi-Bellman equation for stochastic optimal control.",
                "Summarize the quarterly revenue variance across operating segments.",
            ],
            "eval_prompt": "Derive the stochastic optimal control boundary conditions.",
            "reg_lambda": 0.001,
        },
    )
    assert wass_resp.status_code == 200
    wass_data = wass_resp.json()
    assert wass_data["wasserstein_distance"] >= 0.0
    assert wass_data["n_source_samples"] == 4
    assert wass_data["eval_prob_aligned"] is not None
    assert wass_data["eval_prob_pullback"] is not None
    assert (
        pytest.approx(wass_data["eval_prob_aligned"], abs=1e-4) == wass_data["eval_prob_pullback"]
    )
    assert wass_data["selected_model_aligned"] in MODEL_POOL
