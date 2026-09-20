"""Quickstart Example Script for Mechanistic LLM Router."""

import asyncio
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TaskComplexity
from mechanistic_router.routers.mechanistic import MechanisticRouter
from mechanistic_router.schemas.routing import RoutingRequest


async def main() -> None:
    print("Initializing Mechanistic LLM Router...")

    # Initialize encoder prefill stage simulator and router instance
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)

    # Sample queries across complexity spectrum
    sample_queries = [
        ("What is my current credit card statement balance?", TaskComplexity.ROUTINE),
        ("How can I renegociate a $5,000 debt considering installments?", TaskComplexity.MODERATE),
        (
            "Perform a complete risk analysis on my DTI ratio, LTV, and credit history projection over 12 months.",
            TaskComplexity.COMPLEX,
        ),
    ]

    print("\n--- Routing Decision Evaluation ---")
    for prompt, expected_tier in sample_queries:
        req = RoutingRequest(prompt=prompt)
        decision = await router.route(req)

        print(f"\nPrompt: '{prompt}'")
        print(f"Reference Benchmark Tier: {expected_tier.value}")
        print(f"Selected Route: {decision.selected_model}")
        print(f"Estimated Cost: ${decision.estimated_cost_usd:.4f}")
        print(f"Decision Latency: {decision.latency_ms:.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
