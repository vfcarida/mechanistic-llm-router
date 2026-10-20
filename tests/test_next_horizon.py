"""Comprehensive Tests for Multi-Layer Residual Stream, Group-Lasso, Calibration Pack & ACI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from mechanistic_router.evaluation.calibration_pack import (
    DEFAULT_CONFORMAL_CHECKPOINT,
    DEFAULT_PROBE_CHECKPOINT,
    MULTIDOMAIN_BENCHMARK_CORPUS,
    DeterministicMultiLayerExtractor,
    build_and_save_calibration_pack,
    load_calibrated_causal_router,
    load_multidomain_calibration_dataset,
)
from mechanistic_router.gateway.server import create_app
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.observability.metrics import RouterMetrics
from mechanistic_router.probing.activation_extractor import PrefillActivationExtractor
from mechanistic_router.probing.conformal import ConformalRiskController
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.schemas.routing import ProbingSignals


class _DummyBatchEncoding(dict[str, torch.Tensor]):
    def to(self, device: str) -> _DummyBatchEncoding:
        return self


class _DummyMultiLayerTokenizer:
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
        seq_len = 6
        return _DummyBatchEncoding(
            {
                "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            }
        )


class _DummyMultiLayerModel:
    def eval(self) -> _DummyMultiLayerModel:
        return self

    def to(self, device: str) -> _DummyMultiLayerModel:
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
            torch.full((batch_size, seq_len, d_model), fill_value=float(layer_i + 1))
            for layer_i in range(12)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def test_h1_multilayer_extractor_concat_and_mean() -> None:
    """Verifies multi-layer residual stream extraction (concat & mean fusion)."""
    extractor_concat = PrefillActivationExtractor(
        layer_indices=[-9, -5, -1],
        layer_fusion="concat",
        device="cpu",
    )
    extractor_concat._tokenizer = _DummyMultiLayerTokenizer()
    extractor_concat._model = _DummyMultiLayerModel()

    assert extractor_concat.num_extracted_layers == 3
    vecs_concat = extractor_concat.extract_batch(["Prompt A", "Prompt B"])
    assert vecs_concat.shape == (2, 24)  # 3 layers * 8 dim

    extractor_mean = PrefillActivationExtractor(
        layer_indices=[-9, -5, -1],
        layer_fusion="mean",
        device="cpu",
    )
    extractor_mean._tokenizer = _DummyMultiLayerTokenizer()
    extractor_mean._model = _DummyMultiLayerModel()
    vecs_mean = extractor_mean.extract_batch(["Prompt A", "Prompt B"])
    assert vecs_mean.shape == (2, 8)

    with pytest.raises(ValueError, match="layer_fusion must be"):
        PrefillActivationExtractor(layer_fusion="invalid")  # type: ignore[arg-type]


def test_h1_group_lasso_sparsity_prunes_noisy_layer_groups(tmp_path: Path) -> None:
    """Verifies Group-Lasso (L2,1) block coordinate shrinkage and group importances."""
    rng = np.random.default_rng(42)
    n_samples = 120
    d_group = 10

    # Group 0: pure noise; Group 1: pure noise; Group 2: strong linear signal
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=int)
    g0 = rng.normal(0.0, 0.05, size=(n_samples, d_group))
    g1 = rng.normal(0.0, 0.05, size=(n_samples, d_group))
    signal = (2 * y - 1)[:, None] * 2.5
    g2 = signal + rng.normal(0.0, 0.1, size=(n_samples, d_group))
    X = np.hstack([g0, g1, g2]).astype(np.float32)

    probe = LinearActivationProbe(
        C=1.0,
        num_groups=3,
        group_lasso_lambda=0.08,
        random_state=42,
    )
    probe.fit(X, y)

    importances = probe.get_group_importances()
    assert len(importances) == 3
    assert pytest.approx(sum(importances), abs=1e-5) == 1.0
    # Group 2 carries the true causal signal and must dominate group importance
    assert importances[2] > 0.85
    assert importances[0] < 0.10
    assert importances[1] < 0.10

    # Verify persistence of group-lasso metadata
    save_file = tmp_path / "group_probe.npz"
    probe.save(save_file)
    loaded = LinearActivationProbe.load(save_file)
    assert loaded.num_groups == 3
    assert pytest.approx(loaded.group_lasso_lambda) == 0.08
    assert np.allclose(loaded.get_group_importances(), importances)


@pytest.mark.asyncio
async def test_h2_multidomain_calibration_pack_and_default_checkpoints(tmp_path: Path) -> None:
    """Verifies 5-domain calibration dataset and pre-calibrated checkpoint pack."""
    assert len(MULTIDOMAIN_BENCHMARK_CORPUS) == 5
    cases = load_multidomain_calibration_dataset(n_samples=100, seed=42)
    assert len(cases) == 100

    probe_file = tmp_path / "probe.npz"
    conformal_file = tmp_path / "conformal.npz"
    probe, controller, summary = build_and_save_calibration_pack(
        probe_path=probe_file,
        conformal_path=conformal_file,
        alpha=0.10,
        n_samples=150,
        seed=42,
    )
    assert probe_file.is_file()
    assert conformal_file.is_file()
    assert controller.is_calibrated
    assert summary.upper_risk_bound_low <= 0.10 + 1e-9
    assert len(probe.get_group_importances()) == 3

    # Ensure repository default checkpoints are also generated and loadable
    if not DEFAULT_PROBE_CHECKPOINT.is_file() or not DEFAULT_CONFORMAL_CHECKPOINT.is_file():
        build_and_save_calibration_pack()
    assert DEFAULT_PROBE_CHECKPOINT.is_file()
    assert DEFAULT_CONFORMAL_CHECKPOINT.is_file()

    router = load_calibrated_causal_router(
        model_pool=MODEL_POOL,
        probe_path=probe_file,
        conformal_path=conformal_file,
    )
    easy_decision = await router.route("What is my current checking account balance?")
    hard_decision = await router.route(
        "Derive the closed-form solution for the stochastic differential equation "
        "dX_t = mu * X_t * dt + sigma * X_t * dW_t using Ito's Lemma and prove "
        "martingale convergence conditions for the discounted process."
    )
    assert easy_decision.selected_model in ("SLM-BERTau-Local", "LLM-Mid-Tier")
    assert hard_decision.selected_model == "LLM-Frontier-Oracle"


def test_h2_deterministic_multilayer_extractor_modes() -> None:
    """Verifies DeterministicMultiLayerExtractor in both concat and mean fusion modes."""
    ext_concat = DeterministicMultiLayerExtractor(
        d_layer=16, layer_indices=[-9, -5, -1], layer_fusion="concat"
    )
    assert ext_concat.extract_batch([]).shape == (0, 48)
    out_concat = ext_concat.extract_batch(["Hello world"])
    assert out_concat.shape == (1, 48)

    ext_mean = DeterministicMultiLayerExtractor(
        d_layer=16, layer_indices=[-9, -5, -1], layer_fusion="mean"
    )
    assert ext_mean.extract_batch([]).shape == (0, 16)
    out_mean = ext_mean.extract_batch(["Hello world"])
    assert out_mean.shape == (1, 16)


def test_h3_online_conformal_adaptation_under_distribution_shift(tmp_path: Path) -> None:
    """Verifies Adaptive Conformal Inference (Gibbs & Candes ACI) online updates."""
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.0, 1.0, size=200)
    cheap_losses = np.where(probs > 0.35, 0.5, 0.0)
    mid_losses = np.where(probs > 0.65, 0.4, 0.0)

    controller = ConformalRiskController(alpha=0.10, learning_rate=0.10)
    controller.calibrate(probs, cheap_losses, mid_losses)

    initial_low = controller.low_threshold
    initial_high = controller.high_threshold
    assert initial_low > 0.0

    # Simulate distribution shift with high quality loss (0.90 > alpha=0.10) -> thresholds tighten
    for _ in range(5):
        controller.update_online(incurred_loss_low=0.90, incurred_loss_high=0.80)

    tightened_low = controller.low_threshold
    tightened_high = controller.high_threshold
    assert tightened_low < initial_low
    assert tightened_high < initial_high
    assert controller.online_steps == 5
    emp_low, emp_high = controller.online_empirical_risk
    assert pytest.approx(emp_low) == 0.90
    assert pytest.approx(emp_high) == 0.80

    # Simulate benign regime with 0.0 loss (< alpha=0.10) -> thresholds relax upward
    for _ in range(5):
        controller.update_online(incurred_loss_low=0.0, incurred_loss_high=0.0)

    assert controller.low_threshold > tightened_low
    assert controller.high_threshold > tightened_high
    assert controller.online_steps == 10

    # Persistence round-trip preserves online state
    ckpt = tmp_path / "online_conformal.npz"
    controller.save(ckpt)
    loaded = ConformalRiskController.load(ckpt)
    assert loaded.online_steps == 10
    assert pytest.approx(loaded.low_threshold) == controller.low_threshold
    assert pytest.approx(loaded.high_threshold) == controller.high_threshold
    assert pytest.approx(loaded.effective_alpha) == controller.effective_alpha


def test_h4_gateway_inspect_feedback_and_mechanistic_otel_metrics() -> None:
    """Verifies /v1/routing/inspect, /v1/routing/feedback, and OTel metrics."""
    app = create_app()
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-router-key"}

    # 1. White-box inspection endpoint (/v1/routing/inspect)
    resp_inspect = client.post(
        "/v1/routing/inspect",
        headers=headers,
        json={
            "prompt": (
                "Derive the closed-form solution for the stochastic differential equation "
                "using Ito's Lemma and prove martingale convergence."
            ),
            "strategy": "causal-probe-auto",
        },
    )
    assert resp_inspect.status_code == 200
    body = resp_inspect.json()
    assert body["strategy_used"] == "CausalProbeRouter"
    assert body["selected_model"] in MODEL_POOL
    assert body["conformal_state"] is not None
    assert "low_threshold" in body["conformal_state"]
    assert "high_threshold" in body["conformal_state"]
    selected_signals = body["signals"][body["selected_model"]]
    assert "group_importances" in selected_signals["extra_metadata"]
    assert len(selected_signals["extra_metadata"]["group_importances"]) == 3

    prev_low = body["conformal_state"]["low_threshold"]

    # 2. Online conformal feedback endpoint (/v1/routing/feedback)
    resp_fb = client.post(
        "/v1/routing/feedback",
        headers=headers,
        json={
            "incurred_loss_low": 0.85,
            "incurred_loss_high": 0.75,
            "step_size": 0.10,
        },
    )
    assert resp_fb.status_code == 200
    fb_body = resp_fb.json()
    assert fb_body["online_steps"] >= 1
    assert fb_body["low_threshold"] < prev_low

    # 3. Direct verification of RouterMetrics mechanistic instruments
    metrics_inst = RouterMetrics(service_name="test-mechanistic-otel")
    metrics_inst.record_route(
        route_name="llama-3-8b",
        strategy="CausalProbeRouter",
        router_latency_ms=1.2,
        endpoint_latency_ms=15.0,
        cost_saved_usd=0.0148,
        signals={
            "llama-3-8b": ProbingSignals(
                d_eff_mean=24.5,
                fisher_j=0.42,
                fisher_j_norm=0.42,
                is_competent=True,
                sae_features=[3, 12],
                extra_metadata={
                    "epistemic_uncertainty": 0.88,
                    "abstention_escalated": True,
                },
            )
        },
    )
