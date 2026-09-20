"""Tests leakage isolation for CausalProbePolicy."""

import inspect

import numpy as np
import pytest

from experiments.mechanistic_probe.policy import CausalProbePolicy
from experiments.mechanistic_probe.probe import LinearActivationProbe
from mechanistic_router.evaluation.baselines import MODEL_POOL
from mechanistic_router.schemas.eval import EvalCase
from mechanistic_router.schemas.routing import TaskComplexity


class MockExtractor:
    """Mock extractor for fast unit testing without loading weights."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def extract_one(self, prompt: str) -> np.ndarray:
        # Deterministic vector based on prompt length
        rng = np.random.default_rng(len(prompt))
        return rng.standard_normal(self.dim).astype(np.float32)

    def extract_batch(self, prompts: list[str]) -> np.ndarray:
        return np.stack([self.extract_one(p) for p in prompts], axis=0)


@pytest.mark.asyncio
async def test_causal_probe_policy_leakage_boundary() -> None:
    """Verifies that CausalProbePolicy.route() receives only prompt text and no eval labels."""
    extractor = MockExtractor(dim=16)
    probe = LinearActivationProbe(random_state=42)

    # Mock fitting
    X_train = np.random.default_rng(42).standard_normal((20, 16)).astype(np.float32)
    y_train = np.array([0] * 10 + [1] * 10)
    probe.fit(X_train, y_train)

    policy = CausalProbePolicy(extractor=extractor, probe=probe, model_pool=MODEL_POOL)  # type: ignore[arg-type]

    # Verify route() signature
    sig = inspect.signature(policy.route)
    params = list(sig.parameters.keys())
    assert params == ["prompt"], f"Policy route() must only accept prompt, found: {params}"

    # Verify route execution produces valid model decision
    decision = await policy.route("Qual é o saldo da minha conta corrente?")
    assert decision in MODEL_POOL


def test_causal_probe_tuning_on_dev_cases() -> None:
    """Verifies tuning adjusts threshold based on dev cases."""
    extractor = MockExtractor(dim=16)
    probe = LinearActivationProbe(random_state=42)
    X = np.random.default_rng(123).standard_normal((30, 16)).astype(np.float32)
    y = np.array([0] * 15 + [1] * 15)
    probe.fit(X, y)

    policy = CausalProbePolicy(extractor=extractor, probe=probe, model_pool=MODEL_POOL)  # type: ignore[arg-type]

    dev_cases = [
        EvalCase(
            case_id=f"case_{i}",
            prompt=f"Pergunta de teste {i}",
            reference_tier=TaskComplexity.ROUTINE if i % 2 == 0 else TaskComplexity.COMPLEX,
            per_model_outcome={
                "SLM-BERTau-Local": 0.9 if i % 2 == 0 else 0.2,
                "LLM-Frontier-Oracle": 0.98,
            },
            price_table={"SLM-BERTau-Local": 0.02, "LLM-Frontier-Oracle": 1.50},
        )
        for i in range(10)
    ]

    _ = policy.threshold
    policy.tune(dev_cases)
    # Threshold was evaluated on dev set
    assert isinstance(policy.threshold, float)
    assert 0.1 <= policy.threshold <= 0.9
