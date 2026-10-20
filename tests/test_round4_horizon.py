"""Unit and Integration Tests for CRDT Conformal Sync, Speculative Gateway, LinUCB & CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from mechanistic_router.cli import main as cli_main
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.gateway.dispatcher import LiteLLMDispatcher, ProviderDispatchError
from mechanistic_router.gateway.server import create_app, get_max_prompt_chars
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.activation_extractor import PrefillActivationExtractor
from mechanistic_router.probing.base import TransformerActivationEncoder
from mechanistic_router.probing.conformal import ConformalRiskController
from mechanistic_router.probing.distributed_conformal import (
    AtomicFileConformalBackend,
    ConformalStateSnapshot,
    DistributedConformalCoordinator,
    InMemorySharedConformalBackend,
    merge_conformal_snapshots,
)
from mechanistic_router.routers.linucb_probe import LinUCBActivationRouter
from mechanistic_router.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)
from mechanistic_router.schemas.routing import RoutingRequest


def _make_calibrated_controller(alpha: float = 0.12) -> ConformalRiskController:
    rng = np.random.default_rng(42)
    probs = np.linspace(0.01, 0.99, 80)
    cheap_losses = np.where(probs < 0.4, 0.02, 0.45) + rng.uniform(0.0, 0.01, size=80)
    mid_losses = np.where(probs < 0.75, 0.01, 0.35) + rng.uniform(0.0, 0.01, size=80)
    ctrl = ConformalRiskController(alpha=alpha, learning_rate=0.05)
    ctrl.calibrate(probs, cheap_losses, mid_losses)
    return ctrl


def test_h9_conformal_snapshot_commutativity_and_associativity() -> None:
    """Verify CRDT merge of ConformalStateSnapshots is strictly commutative and associative."""
    c1 = _make_calibrated_controller()
    c2 = _make_calibrated_controller()
    c3 = _make_calibrated_controller()

    c1.update_online(0.35, 0.20)
    c1.update_online(0.40, 0.25)

    c2.update_online(0.02, 0.01)

    c3.update_online(0.50, 0.30)
    c3.update_online(0.10, 0.05)
    c3.update_online(0.25, 0.15)

    s1 = ConformalStateSnapshot.from_controller(c1, replica_id="r1")
    s2 = ConformalStateSnapshot.from_controller(c2, replica_id="r2")
    s3 = ConformalStateSnapshot.from_controller(c3, replica_id="r3")

    m12 = merge_conformal_snapshots(s1, s2)
    m21 = merge_conformal_snapshots(s2, s1)
    assert m12 == m21
    assert m12.online_steps == 3
    assert np.isclose(m12.cumulative_loss_low, 0.77)

    m_left = merge_conformal_snapshots(merge_conformal_snapshots(s1, s2), s3)
    m_right = merge_conformal_snapshots(s1, merge_conformal_snapshots(s2, s3))
    assert m_left == m_right
    assert m_left.online_steps == 6


def test_h9_distributed_conformal_coordinator_and_file_backend(tmp_path: Path) -> None:
    """Verify multi-replica synchronization over InMemory and AtomicFile backends."""
    state_file = tmp_path / "conformal_shared.json"
    backend_r1 = AtomicFileConformalBackend(state_file)
    backend_r2 = AtomicFileConformalBackend(state_file)

    coord1 = DistributedConformalCoordinator(backend=backend_r1, replica_id="worker-1")
    coord2 = DistributedConformalCoordinator(backend=backend_r2, replica_id="worker-2")

    ctrl1 = _make_calibrated_controller()
    ctrl2 = _make_calibrated_controller()

    ctrl1.update_online(0.45, 0.30)
    snap1 = coord1.sync_controller(ctrl1)
    assert snap1.online_steps == 1

    ctrl2.update_online(0.35, 0.20)
    ctrl2.update_online(0.25, 0.10)
    snap2 = coord2.sync_controller(ctrl2)
    assert snap2.online_steps == 3

    # Re-sync worker-1 with no new local steps -> pulls global 3 steps
    snap1_synced = coord1.sync_controller(ctrl1)
    assert snap1_synced.online_steps == 3
    assert np.isclose(ctrl1.low_threshold, ctrl2.low_threshold)
    assert np.isclose(ctrl1.high_threshold, ctrl2.high_threshold)

    # Validation errors on uncalibrated controller & negative delta
    uncal = ConformalRiskController(alpha=0.10)
    with pytest.raises(ValueError, match="uncalibrated"):
        ConformalStateSnapshot.from_controller(uncal)
    with pytest.raises(ValueError, match="calibrated"):
        snap1.apply_to_controller(uncal)
    with pytest.raises(ValueError, match="non-negative"):
        coord1.merge_remote_delta(ctrl1, delta_steps=-1, delta_loss_low=0.0, delta_loss_high=0.0)


def test_h9_gateway_conformal_sync_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GET /v1/routing/conformal-state and POST /v1/routing/conformal-sync."""
    monkeypatch.setenv("ROUTER_API_KEY", "test-round4-key")
    app = create_app(DEFAULT_CONFIG.model_copy())
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-round4-key"}

    resp_state = client.get("/v1/routing/conformal-state", headers=headers)
    assert resp_state.status_code == 200
    initial_data = resp_state.json()
    assert initial_data["online_steps"] == 0

    resp_sync = client.post(
        "/v1/routing/conformal-sync",
        headers=headers,
        json={
            "delta_steps": 4,
            "delta_loss_low": 1.6,
            "delta_loss_high": 0.8,
            "replica_id": "edge-replica-7",
        },
    )
    assert resp_sync.status_code == 200
    synced_data = resp_sync.json()
    assert synced_data["online_steps"] == 4
    assert synced_data["low_threshold"] <= initial_data["low_threshold"]

    # Also test HTTP 400 when conformal_controller is None
    app.state.causal_probe_router.conformal_controller = None
    assert client.get("/v1/routing/conformal-state", headers=headers).status_code == 400
    assert (
        client.post(
            "/v1/routing/conformal-sync",
            headers=headers,
            json={"delta_steps": 1, "delta_loss_low": 0.1, "delta_loss_high": 0.1},
        ).status_code
        == 400
    )


def test_h10_gateway_speculative_cascade_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /v1/chat/completions triggers speculative hand-off when cheap tier hedges."""
    monkeypatch.setenv("ROUTER_API_KEY", "test-round4-key")
    app = create_app(DEFAULT_CONFIG.model_copy())
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-round4-key"}

    async def mock_dispatch(
        target_model: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        if target_model != "LLM-Frontier-Oracle":
            text = "I am not sure and cannot derive the solution, it is beyond my capability."
        else:
            text = "Complete formal derivation resolved by the frontier oracle."
        return ChatCompletionResponse(
            id="chatcmpl-spec",
            model=target_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=8, completion_tokens=12, total_tokens=20),
            router_strategy="LiteLLMDispatcher",
        )

    monkeypatch.setattr(app.state.dispatcher, "dispatch", AsyncMock(side_effect=mock_dispatch))

    resp = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "speculative-cascade-auto",
            "messages": [{"role": "user", "content": "What is my checking account balance?"}],
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-speculative-handoff") == "true"
    assert resp.headers.get("x-initial-model") in ("SLM-BERTau-Local", "LLM-Mid-Tier")
    body = resp.json()
    assert body["model"] == "LLM-Frontier-Oracle"
    assert "[SpeculativeHandoff] -> LLM-Frontier-Oracle" in body["router_strategy"]


@pytest.mark.asyncio
async def test_h11_linucb_sherman_morrison_exact_inverse_and_persistence(
    tmp_path: Path,
) -> None:
    """Verify LinUCB Sherman-Morrison O(D^2) update matches exact matrix inversion."""
    router = LinUCBActivationRouter(
        model_pool=MODEL_POOL,
        feature_dim=16,
        alpha_ucb=0.4,
        ridge_lambda=1.5,
    )

    arm = "SLM-BERTau-Local"
    a_exact = 1.5 * np.eye(16, dtype=np.float64)
    prompts = [
        "Check my current checking balance",
        "Transfer 50 dollars to savings",
        "Show recent card transactions",
    ]

    for idx, p in enumerate(prompts):
        x = router._extract_context(p)
        a_exact += np.outer(x, x)
        stats = router.update_feedback(arm, p, reward=0.9 - 0.1 * idx)
        assert float(stats["uncertainty_after"]) < float(stats["uncertainty_before"])

    expected_a_inv = np.linalg.inv(a_exact)
    np.testing.assert_allclose(router._a_inv[arm], expected_a_inv, atol=1e-10)

    # Test _extract_context with raw numpy vectors (shorter, longer, and all-zeros)
    short_ctx = router._extract_context(np.ones(8, dtype=np.float64))
    long_ctx = router._extract_context(np.ones(32, dtype=np.float64))
    zero_ctx = router._extract_context(np.zeros(16, dtype=np.float64))
    assert short_ctx.shape == (16,) and long_ctx.shape == (16,) and zero_ctx.shape == (16,)

    decision = await router.route(
        RoutingRequest(prompt="Check my current checking balance", max_budget_usd=0.002)
    )
    assert decision.selected_model in ("SLM-BERTau-Local", "LLM-Mid-Tier")
    assert decision.strategy_used == "LinUCBActivationRouter"

    # Also test tiny max_budget_usd fallback
    decision_tiny = await router.route(RoutingRequest(prompt="Check balance", max_budget_usd=1e-9))
    assert decision_tiny.selected_model in MODEL_POOL

    # Persistence round-trip
    ckpt = tmp_path / "linucb_state.npz"
    router.save(ckpt)
    loaded = LinUCBActivationRouter.load(ckpt, model_pool=MODEL_POOL)
    np.testing.assert_allclose(loaded._a_inv[arm], router._a_inv[arm], atol=1e-10)
    assert loaded._pull_counts[arm] == 3

    # Validation checks
    with pytest.raises(ValueError, match="feature_dim"):
        LinUCBActivationRouter(MODEL_POOL, feature_dim=1)
    with pytest.raises(ValueError, match="alpha_ucb"):
        LinUCBActivationRouter(MODEL_POOL, alpha_ucb=-0.1)
    with pytest.raises(ValueError, match="ridge_lambda"):
        LinUCBActivationRouter(MODEL_POOL, ridge_lambda=0.0)
    with pytest.raises(ValueError, match="Unknown model_name"):
        router.update_feedback("NonExistentModel", "hello", 0.5)
    with pytest.raises(TypeError, match="RoutingRequest or str"):
        await router.route(123)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        LinUCBActivationRouter.load(tmp_path / "missing.npz", model_pool=MODEL_POOL)


def test_h12_cli_subcommands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI subcommands `calibrate`, `inspect`, and `benchmark` return valid JSON."""
    rc_cal = cli_main(["calibrate", "--output-dir", str(tmp_path), "--alpha", "0.12"])
    assert rc_cal == 0
    out_cal = json.loads(capsys.readouterr().out)
    assert out_cal["status"] == "calibrated"
    assert (tmp_path / "multilayer_group_lasso_probe.npz").is_file()

    for strat in (
        "causal-probe-auto",
        "linucb-auto",
        "mechanistic-auto",
        "semantic-auto",
        "cost-performance-auto",
    ):
        rc_insp = cli_main(
            ["inspect", "--prompt", "Provide formal PDE proof.", "--strategy", strat]
        )
        assert rc_insp == 0
        out_insp = json.loads(capsys.readouterr().out)
        assert out_insp["selected_model"] in MODEL_POOL

    rc_bench = cli_main(["benchmark", "--samples", "4"])
    assert rc_bench == 0
    out_bench = json.loads(capsys.readouterr().out)
    assert out_bench["samples_evaluated"] == 4
    assert "causal-probe-auto" in out_bench["strategies"]
    assert "linucb-auto" in out_bench["strategies"]


@pytest.mark.asyncio
async def test_coverage_elevation_base_dispatcher_extractor_and_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover edge branches in probing/base.py, dispatcher.py, and activation_extractor.py."""

    # 1. TransformerActivationEncoder branches in probing/base.py
    class _LastHiddenOnlyModel(nn.Module):
        def forward(
            self, input_ids: torch.Tensor, output_hidden_states: bool = True, **kwargs: Any
        ) -> Any:
            h = torch.ones((input_ids.size(0), input_ids.size(1), 8))
            return SimpleNamespace(last_hidden_state=h, hidden_states=None)

    class _HiddenStatesOnlyModel(nn.Module):
        def forward(
            self, input_ids: torch.Tensor, output_hidden_states: bool = True, **kwargs: Any
        ) -> Any:
            h = torch.ones((input_ids.size(0), input_ids.size(1), 8))
            return SimpleNamespace(last_hidden_state=None, hidden_states=[h, h * 2])

    class _EmptyOutputsModel(nn.Module):
        def forward(
            self, input_ids: torch.Tensor, output_hidden_states: bool = True, **kwargs: Any
        ) -> Any:
            return SimpleNamespace(last_hidden_state=None, hidden_states=None)

    enc_last = TransformerActivationEncoder(model=_LastHiddenOnlyModel(), tokenizer=None)
    last_h, inter_h = enc_last(torch.tensor([[1, 2, 3]]))
    assert last_h.shape == (1, 3, 8) and len(inter_h) == 1
    with pytest.raises(RuntimeError, match="Tokenizer is not configured"):
        enc_last.encode_text("hello")

    enc_hs = TransformerActivationEncoder(model=_HiddenStatesOnlyModel(), tokenizer=None)
    last_h2, inter_h2 = enc_hs(torch.tensor([[1, 2]]))
    assert last_h2.shape == (1, 2, 8) and len(inter_h2) == 2

    enc_empty = TransformerActivationEncoder(model=_EmptyOutputsModel(), tokenizer=None)
    with pytest.raises(RuntimeError, match="did not provide last_hidden_state"):
        enc_empty(torch.tensor([[1, 2]]))

    # 2. PrefillActivationExtractor mean pooling, mean layer fusion, and corrupt cache
    corrupt_cache = tmp_path / "corrupt.npz"
    corrupt_cache.write_bytes(b"not-a-valid-zip")

    class _DummyTokenizer:
        vocab_size = 100

        def __call__(self, texts: list[str], **kwargs: Any) -> Any:
            batch = len(texts)
            res = {
                "input_ids": torch.ones((batch, 4), dtype=torch.long),
                "attention_mask": torch.ones((batch, 4), dtype=torch.long),
            }
            return SimpleNamespace(
                to=lambda dev: res,
                __getitem__=lambda self_obj, k: res[k],
            )

    dummy_enc = TransformerActivationEncoder(
        model=_HiddenStatesOnlyModel(), tokenizer=_DummyTokenizer()
    )
    extractor = PrefillActivationExtractor(
        encoder=dummy_enc,
        pooling="mean",
        layer_indices=[0, 1],
        layer_fusion="mean",
        cache_file=str(corrupt_cache),
    )
    vec = extractor.extract_one("mean pooled multi-layer prompt")
    assert vec.shape == (8,)

    # InMemorySharedConformalBackend read/write
    mem_backend = InMemorySharedConformalBackend()
    assert mem_backend.read() is None
    snap_ctrl = ConformalStateSnapshot.from_controller(_make_calibrated_controller())
    mem_backend.write(snap_ctrl)
    assert mem_backend.read() == snap_ctrl

    # 3. LiteLLMDispatcher drop_params=True and synchronous stream iterator + retry failure
    disp = LiteLLMDispatcher(max_retries=2, cooldown_seconds=0.01, drop_params=True)
    req = ChatCompletionRequest(
        model="mechanistic-auto",
        messages=[ChatCompletionMessage(role="user", content="stream test")],
        stream=True,
    )

    def _sync_chunk_iter() -> Any:
        yield SimpleNamespace(
            id="chunk-1",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))],
        )
        yield SimpleNamespace(
            id="chunk-2",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))],
        )

    with patch("litellm.acompletion", new=AsyncMock(return_value=_sync_chunk_iter())):
        chunks = [c async for c in disp.dispatch_stream("SLM-BERTau-Local", req)]
        assert len(chunks) == 3
        assert "[DONE]" in chunks[-1]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("stream down"))):
        with pytest.raises(ProviderDispatchError, match="stream down"):
            _ = [c async for c in disp.dispatch_stream("SLM-BERTau-Local", req)]

    # 4. get_max_prompt_chars invalid env vars and FastAPI lifespan context
    monkeypatch.setenv("ROUTER_MAX_PROMPT_CHARS", "-5")
    assert get_max_prompt_chars() == 10_000
    monkeypatch.setenv("ROUTER_MAX_PROMPT_CHARS", "not-an-int")
    assert get_max_prompt_chars() == 10_000
    monkeypatch.setenv("ROUTER_MAX_PROMPT_CHARS", "4096")
    assert get_max_prompt_chars() == 4096

    monkeypatch.setenv("ROUTER_API_KEY", "lifespan-key")
    custom_cfg = DEFAULT_CONFIG.model_copy(update={"lambda_budget": 0.4})
    with TestClient(create_app(custom_cfg)) as client:
        r_models = client.get("/v1/models")
        assert r_models.status_code == 200
        model_ids = [m["id"] for m in r_models.json()["data"]]
        assert "linucb-auto" in model_ids
        assert "speculative-cascade-auto" in model_ids

        for strat in ("linucb-auto", "semantic-auto", "cost-performance-auto", "mechanistic-auto"):
            r_insp = client.post(
                "/v1/routing/inspect",
                headers={"X-API-Key": "lifespan-key"},
                json={"prompt": "Evaluate routing strategy", "strategy": strat},
            )
            assert r_insp.status_code == 200
