"""Causal Probe Routing Demonstration Script.

Demonstrates end-to-end training of a LinearActivationProbe on residual-stream
representations, secure serialization to compressed NumPy (.npz), checkpoint reloading,
and asynchronous query routing with CausalProbeRouter.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.routers.causal_probe import CausalProbeRouter
from mechanistic_router.schemas.routing import RoutingRequest


def train_and_persist_probe(probe_path: Path, hidden_dim: int = 128) -> None:
    """Simulates residual-stream latent vectors and trains a LinearActivationProbe."""
    print("1. Generating synthetic activation dataset for training...")
    rng = np.random.RandomState(42)
    n_samples = 100

    # Synthetic latent representations: routine (class 0) vs complex (class 1)
    X = rng.randn(n_samples, hidden_dim).astype(np.float32)
    # Define an activation direction: positive on first 4 features indicates complex reasoning
    y = (np.sum(X[:, :4], axis=1) > 0.0).astype(int)

    probe = LinearActivationProbe(C=1.0, random_state=42)
    probe.fit(X, y)

    print(f"   Fitted probe with weight vector shape: {probe.weights.shape}")
    print(f"   Hyperplane normal vector norm (L2): {probe.direction_norm:.4f}")
    print(f"   Hyperplane intercept: {probe.intercept:.4f}")

    print(f"2. Saving probe checkpoint securely via compressed NumPy to: {probe_path}")
    probe.save(probe_path)
    print(f"   Checkpoint saved successfully ({probe_path.stat().st_size} bytes).")


async def main() -> None:
    print("=== Mechanistic Causal Probe Routing Demonstration ===\n")

    checkpoint_dir = Path(".cache/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probe_checkpoint = checkpoint_dir / "causal_probe_demo.npz"

    # Step 1: Train & Save
    train_and_persist_probe(probe_checkpoint, hidden_dim=128)

    # Step 2: Load into CausalProbeRouter
    print("\n3. Initializing CausalProbeRouter from saved checkpoint...")
    mock_extractor = MagicMock()

    # Define mock activations for representative queries
    routine_activation = np.random.RandomState(10).randn(128).astype(np.float32)
    # Force negative projection on first 4 features (routine)
    routine_activation[:4] = -2.0

    complex_activation = np.random.RandomState(20).randn(128).astype(np.float32)
    # Force positive projection on first 4 features (complex)
    complex_activation[:4] = 3.0

    def mock_extract(prompt: str) -> np.ndarray:
        if "risk" in prompt.lower() or "dti" in prompt.lower() or "complex" in prompt.lower():
            return complex_activation
        return routine_activation

    mock_extractor.extract_one.side_effect = mock_extract

    router = CausalProbeRouter.from_saved_probe(
        probe_path=probe_checkpoint,
        extractor=mock_extractor,
        model_pool=MODEL_POOL,
        config=DEFAULT_CONFIG,
        threshold=0.5,
    )

    sample_queries = [
        "What is my current credit card statement balance?",
        "Perform a complete risk analysis on my DTI ratio, LTV, and credit history.",
    ]

    print("\n4. Evaluating Routing Decisions across latent decision boundary:")
    for query in sample_queries:
        req = RoutingRequest(prompt=query)
        decision = await router.route(req)
        signals = decision.signals[decision.selected_model]
        metadata = signals.extra_metadata

        print(f"\n   Query: '{query}'")
        print(f"   Selected Model:     {decision.selected_model}")
        print(f"   Estimated Cost:     ${decision.estimated_cost_usd:.4f}")
        print(f"   Latency:            {decision.latency_ms:.2f} ms")
        print(f"   Prob(Strong Model): {metadata.get('prob_strong', 0.0):.4f}")
        print(f"   Separation Margin:  {metadata.get('margin', 0.0):.4f}")
        print(f"   Direction Norm:     {metadata.get('direction_norm', 0.0):.4f}")

    print("\n=== Causal Probe Routing Demonstration Completed Successfully ===")


if __name__ == "__main__":
    asyncio.run(main())
