"""Comprehensive Tests for Procrustes Alignment, Speculative Cascade, Multi-Objective CRC & SAE."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mechanistic_router.evaluation.baselines import (
    CausalProbePolicy,
    CostPerformancePolicy,
    MechanisticPolicy,
    SemanticPolicy,
)
from mechanistic_router.evaluation.calibration_pack import load_calibrated_causal_router
from mechanistic_router.evaluation.metrics import (
    compute_matched_accuracy_cost,
    compute_matched_cost_accuracy,
    compute_paired_bootstrap_ci,
    compute_pareto_auc,
    compute_pareto_front,
)
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.activation_extractor import PrefillActivationExtractor
from mechanistic_router.probing.alignment import OrthogonalProcrustesAligner
from mechanistic_router.probing.conformal import ConformalRiskController
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.routers.speculative_cascade import (
    SpeculativeCascadeMonitor,
)


class _DummyBatchEncoding(dict[str, torch.Tensor]):
    def to(self, device: str) -> _DummyBatchEncoding:
        return self


class _DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ) -> _DummyBatchEncoding:
        batch_size = len(prompts)
        seq_len = 5
        return _DummyBatchEncoding(
            {
                "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }
        )


class _DummyModel:
    def eval(self) -> _DummyModel:
        return self

    def to(self, device: str) -> _DummyModel:
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        d_model = 8
        hidden_states = tuple(
            torch.arange(batch_size * seq_len * d_model, dtype=torch.float32).reshape(
                batch_size, seq_len, d_model
            )
            + float(layer_i)
            for layer_i in range(6)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def test_h5_orthogonal_procrustes_aligner_and_zero_label_probe_transfer(tmp_path: Path) -> None:
    """Verifies Cross-Family Procrustes Alignment and zero-label probe transfer."""
    rng = np.random.default_rng(123)
    n_anchors = 120
    dim = 16

    X_source = rng.normal(0.0, 1.0, size=(n_anchors, dim)).astype(np.float64)
    # Create random orthogonal matrix Q via QR decomposition
    random_mat = rng.normal(0.0, 1.0, size=(dim, dim))
    q_orth, _ = np.linalg.qr(random_mat)
    shift = rng.normal(0.0, 0.5, size=(dim,))

    X_target = (X_source @ q_orth) * 1.8 + shift

    aligner = OrthogonalProcrustesAligner(center=True, scale=True)
    with pytest.raises(ValueError, match="has not been fitted"):
        _ = aligner.rotation_matrix
    with pytest.raises(ValueError, match="has not been fitted"):
        aligner.transform(X_source)
    with pytest.raises(ValueError, match="Cannot save an unfitted"):
        aligner.save(tmp_path / "unfitted.npz")

    aligner.fit(X_source, X_target)
    assert aligner.is_fitted
    assert aligner.rotation_matrix.shape == (dim, dim)

    cos_sim = aligner.alignment_cosine_similarity(X_source, X_target)
    assert cos_sim > 0.999

    # Verify 1D vector transform
    single_aligned = aligner.transform(X_source[0])
    assert single_aligned.shape == (dim,)
    assert np.allclose(single_aligned, X_target[0], atol=1e-4)

    # Fit a LinearActivationProbe in target space and pull it back to source space without labels
    w_true = rng.normal(0.0, 1.0, size=(dim,))
    logits = X_target @ w_true
    y_target = (logits > np.median(logits)).astype(int)

    target_probe = LinearActivationProbe(C=2.0, random_state=42)
    target_probe.fit(X_target.astype(np.float32), y_target)

    transferred_probe = aligner.align_probe(target_probe)
    probs_target = target_probe.predict_proba(X_target.astype(np.float32))
    probs_source_transferred = transferred_probe.predict_proba(X_source.astype(np.float32))
    assert np.allclose(probs_target, probs_source_transferred, atol=1e-4)

    # Persistence round-trip
    save_path = tmp_path / "procrustes.npz"
    aligner.save(save_path)
    loaded = OrthogonalProcrustesAligner.load(save_path)
    assert loaded.is_fitted
    assert np.allclose(loaded.transform(X_source), aligner.transform(X_source), atol=1e-5)

    with pytest.raises(FileNotFoundError):
        OrthogonalProcrustesAligner.load(tmp_path / "missing.npz")


@pytest.mark.asyncio
async def test_h6_speculative_cascade_monitor_handoff_and_confident_stream() -> None:
    """Verifies mid-generation token uncertainty monitoring and speculative hand-off."""
    monitor = SpeculativeCascadeMonitor(
        handoff_threshold=0.60,
        max_prefix_chunks=3,
        nll_scale=2.0,
    )

    with pytest.raises(ValueError, match="handoff_threshold"):
        SpeculativeCascadeMonitor(handoff_threshold=0.0)
    with pytest.raises(ValueError, match="max_prefix_chunks"):
        SpeculativeCascadeMonitor(max_prefix_chunks=0)
    with pytest.raises(ValueError, match="nll_scale"):
        SpeculativeCascadeMonitor(nll_scale=-1.0)

    # Check repetition detection
    rep_u = monitor.evaluate_chunk_uncertainty("loop loop loop loop loop loop loop")
    assert rep_u >= 0.80

    # Case 1: Cheap model streams confident tokens -> no handoff triggered
    async def confident_stream(
        model_name: str, prompt: str
    ) -> AsyncGenerator[tuple[str, list[float] | None], None]:
        _ = (model_name, prompt)
        yield ("The balance ", [-0.05, -0.02])
        yield ("is $250.00 ", [-0.03, -0.01])
        yield ("today.", [-0.01])

    res_ok = await monitor.execute_with_handoff(
        prompt="What is my balance?",
        initial_model="SLM-BERTau-Local",
        fallback_model="LLM-Frontier-Oracle",
        stream_provider=confident_stream,
    )
    assert not res_ok.handoff_triggered
    assert res_ok.final_model == "SLM-BERTau-Local"
    assert res_ok.generated_text == "The balance is $250.00 today."

    # Case 2: Cheap model spikes uncertainty on chunk 2 -> triggers handoff to Oracle
    async def uncertain_stream(
        model_name: str, prompt: str
    ) -> AsyncGenerator[tuple[str, list[float] | None], None]:
        _ = prompt
        if model_name == "SLM-BERTau-Local":
            yield ("Step 1: Let X_t be a process. ", [-0.10, -0.08])
            yield ("I am not sure how to derive the martingale bound.", [-2.4, -2.8])
            yield ("Unreachable chunk", [-0.1])
        else:
            yield ("By Ito's Lemma, d(X_t) converges almost surely.", [-0.04, -0.02])

    res_handoff = await monitor.execute_with_handoff(
        prompt="Prove martingale convergence.",
        initial_model="SLM-BERTau-Local",
        fallback_model="LLM-Frontier-Oracle",
        stream_provider=uncertain_stream,
    )
    assert res_handoff.handoff_triggered
    assert res_handoff.handoff_at_chunk == 2
    assert res_handoff.final_model == "LLM-Frontier-Oracle"
    assert "Step 1: Let X_t be a process." in res_handoff.generated_text
    assert "By Ito's Lemma" in res_handoff.generated_text


def test_h7_multi_objective_conformal_risk_control() -> None:
    """Verifies joint quality + SLA latency multi-objective conformal calibration."""
    rng = np.random.default_rng(99)
    probs = np.linspace(0.0, 1.0, 150)
    quality_losses_cheap = np.where(probs > 0.45, 0.40, 0.0)
    sla_losses_cheap = np.where(probs > 0.30, 0.50, 0.0)
    quality_losses_mid = np.where(probs > 0.70, 0.35, 0.0)
    sla_losses_mid = np.where(probs > 0.60, 0.40, 0.0)
    _ = rng

    controller = ConformalRiskController(alpha=0.10)
    summary = controller.calibrate_multi_objective(
        probe_probs=probs,
        quality_losses_cheap=quality_losses_cheap,
        sla_losses_cheap=sla_losses_cheap,
        alpha_quality=0.12,
        alpha_sla=0.08,
        quality_losses_mid=quality_losses_mid,
        sla_losses_mid=sla_losses_mid,
    )

    assert controller.is_calibrated
    assert summary.low_threshold <= summary.high_threshold
    assert summary.upper_risk_bound_low <= 0.08 + 1e-9
    assert summary.upper_risk_bound_high <= 0.08 + 1e-9

    # Also verify binary multi-objective calibration (without mid tier)
    summary_bin = controller.calibrate_multi_objective(
        probe_probs=probs,
        quality_losses_cheap=quality_losses_cheap,
        sla_losses_cheap=sla_losses_cheap,
        alpha_quality=0.10,
        alpha_sla=0.10,
    )
    assert summary_bin.low_threshold == summary_bin.high_threshold


@pytest.mark.asyncio
async def test_h8_causal_probe_router_sae_concept_bottleneck_circuits() -> None:
    """Verifies SAEEngine active circuit extraction inside CausalProbeRouter."""
    router = load_calibrated_causal_router(model_pool=MODEL_POOL)
    decision = await router.route(
        "Derive the closed-form solution for the stochastic differential "
        "equation using Ito's Lemma."
    )
    sig = decision.signals[decision.selected_model]
    assert isinstance(sig.sae_features, list)
    assert len(sig.sae_features) > 0
    assert "circuit_type" in sig.extra_metadata
    assert sig.extra_metadata["circuit_type"] in (
        "factual_retrieval",
        "mathematical_computation",
        "polysemantic_reasoning",
    )


@pytest.mark.asyncio
async def test_coverage_elevation_extractor_cache_and_baselines(tmp_path: Path) -> None:
    """Exercises last-token pooling, disk cache persistence, and baseline policy adapters."""
    cache_file = tmp_path / "acts_cache.npz"
    extractor = PrefillActivationExtractor(
        pooling="last",
        cache_file=cache_file,
        device="cpu",
    )
    extractor._tokenizer = _DummyTokenizer()
    extractor._model = _DummyModel()

    assert extractor.num_extracted_layers == 1
    vec1 = extractor.extract_one("Prompt one")
    assert vec1.shape == (8,)
    assert cache_file.is_file()

    # Second call hits in-memory cache directly
    vec1_cached = extractor.extract_one("Prompt one")
    assert np.allclose(vec1, vec1_cached)
    batch_cached = extractor.extract_batch(["Prompt one"])
    assert batch_cached.shape == (1, 8)

    # Reload new extractor from persisted cache_file
    extractor_reloaded = PrefillActivationExtractor(
        pooling="last",
        cache_file=cache_file,
        device="cpu",
    )
    assert len(extractor_reloaded._cache) == 1

    # Exercise baseline policy adapters both synchronously and inside running event loop
    causal_router = load_calibrated_causal_router(model_pool=MODEL_POOL)
    pol_causal = CausalProbePolicy(router=causal_router, model_pool=MODEL_POOL)
    pol_causal_empty = CausalProbePolicy(router=None, model_pool=MODEL_POOL)
    pol_cp = CostPerformancePolicy(model_pool=MODEL_POOL)
    pol_sem = SemanticPolicy(model_pool=MODEL_POOL)
    pol_mech = MechanisticPolicy(model_pool=MODEL_POOL)

    assert pol_causal.select_model("Simple balance check") in MODEL_POOL
    assert (await pol_causal.route("Simple balance check")) in MODEL_POOL
    assert pol_causal_empty.select_model("Simple balance check") == "SLM-BERTau-Local"
    assert (await pol_causal_empty.route("Simple balance check")) == "SLM-BERTau-Local"

    assert pol_cp.select_model("Simple balance check") in MODEL_POOL
    assert (await pol_cp.route("Simple balance check")) in MODEL_POOL
    assert pol_sem.select_model("Simple balance check") in MODEL_POOL
    assert (await pol_sem.route("Simple balance check")) in MODEL_POOL
    assert pol_mech.select_model("Simple balance check") in MODEL_POOL
    assert (await pol_mech.route("Simple balance check")) in MODEL_POOL

    # Exercise evaluation metrics edge cases
    assert compute_pareto_front([]) == []
    assert compute_matched_cost_accuracy([], 0.5) == 0.0
    assert compute_matched_accuracy_cost([], 0.8) == 0.0
    assert compute_pareto_auc([]) == 0.0
    assert compute_paired_bootstrap_ci([], [], [], []) == {}
    ci_res = compute_paired_bootstrap_ci(
        [0.1, 0.2], [0.8, 0.9], [0.2, 0.3], [0.7, 0.8], n_bootstraps=50, seed=42
    )
    assert "candidate_acc_ci" in ci_res

    # Exercise OrthogonalProcrustesAligner uncentered/unscaled and validation guards
    aligner_raw = OrthogonalProcrustesAligner(center=False, scale=False)
    X_a = np.eye(4, dtype=np.float64)
    X_b = np.eye(4, dtype=np.float64)
    aligner_raw.fit(X_a, X_b)
    assert aligner_raw.alignment_cosine_similarity(np.zeros((4, 4)), np.zeros((4, 4))) == 0.0
    with pytest.raises(ValueError, match="2D matrices"):
        aligner_raw.fit(np.ones(4), X_b)
    with pytest.raises(ValueError, match="same number of anchor samples"):
        aligner_raw.fit(np.ones((1, 4)), np.ones((1, 4)))
    with pytest.raises(ValueError, match="Expected source dimension"):
        aligner_raw.transform(np.ones((2, 9)))
    with pytest.raises(ValueError, match="shapes do not match"):
        aligner_raw.alignment_cosine_similarity(X_a, np.ones((4, 2)))

    # Exercise LinearActivationProbe group-lasso validation guards
    with pytest.raises(ValueError, match="num_groups must be >= 1"):
        LinearActivationProbe(num_groups=0)
    with pytest.raises(ValueError, match="group_lasso_lambda must be >= 0.0"):
        LinearActivationProbe(group_lasso_lambda=-0.1)
    unfitted_probe = LinearActivationProbe()
    with pytest.raises(ValueError, match="Probe has not been fitted"):
        unfitted_probe.get_group_importances()
    assert LinearActivationProbe.apply_group_lasso_shrinkage(
        np.ones(4, dtype=np.float32), num_groups=1, group_lasso_lambda=0.1
    ).shape == (4,)
