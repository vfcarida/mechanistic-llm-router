"""LLMRouterBench Evaluation & Pareto Optimization Script."""

import asyncio
from typing import Any

import pandas as pd

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.data.mock_dataset import create_financial_dataset
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.routers.cost_performance import CostPerformanceRouter
from mechanistic_router.routers.mechanistic import MechanisticRouter
from mechanistic_router.routers.semantic import SemanticRouter
from mechanistic_router.schemas.routing import RoutingRequest
from mechanistic_router.signals.math_utils import compute_convex_hull


async def evaluate_benchmark(n_samples: int = 150) -> None:
    print(f"Loading LLMRouterBench benchmark dataset ({n_samples} samples)...")
    dataset = create_financial_dataset(n_samples=n_samples, seed=DEFAULT_CONFIG.seed)

    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    routers = {
        "MechanisticRouter": MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG),
        "CostPerformanceRouter": CostPerformanceRouter(MODEL_POOL, DEFAULT_CONFIG),
        "SemanticRouter": SemanticRouter(MODEL_POOL, DEFAULT_CONFIG),
    }

    results: list[dict[str, Any]] = []

    # Evaluate routers without label leakage
    for router_name, router_inst in routers.items():
        total_cost = 0.0
        total_score = 0.0

        for case in dataset:
            req = RoutingRequest(prompt=case.prompt)
            decision = await router_inst.route(req)

            cost = case.price_table.get(decision.selected_model, decision.estimated_cost_usd)
            total_cost += cost

            # Grade using eval-only ground-truth outcomes, not router-internal tiers
            total_score += case.per_model_outcome.get(decision.selected_model, 0.0)

        avg_cost = total_cost / n_samples
        accuracy = total_score / n_samples
        results.append(
            {
                "Strategy": router_name,
                "Avg Cost ($)": round(avg_cost, 4),
                "Accuracy": round(accuracy, 4),
            }
        )

    # Add Oracle Baseline (always routes to frontier model)
    oracle = MODEL_POOL["LLM-Frontier-Oracle"]
    oracle_score = (
        sum(case.per_model_outcome["LLM-Frontier-Oracle"] for case in dataset) / n_samples
    )
    results.append(
        {
            "Strategy": "Oracle (Frontier Only)",
            "Avg Cost ($)": oracle.cost,
            "Accuracy": round(oracle_score, 4),
        }
    )

    # Add Zero-Router Baseline (always routes to local SLM)
    slm = MODEL_POOL["SLM-BERTau-Local"]
    slm_score = sum(case.per_model_outcome["SLM-BERTau-Local"] for case in dataset) / n_samples
    results.append(
        {
            "Strategy": "Zero-Router (SLM Only)",
            "Avg Cost ($)": slm.cost,
            "Accuracy": round(slm_score, 4),
        }
    )

    df = pd.DataFrame(results)
    print("\n=== Benchmark Evaluation Results ===")
    print(df.to_string(index=False))

    # Compute Pareto Non-Decreasing Convex Hull
    points = [(row["Avg Cost ($)"], row["Accuracy"]) for _, row in df.iterrows()]
    pareto_hull = compute_convex_hull(points)

    print("\n=== Mathematical Pareto Convex Hull Boundary Points (Cost $, Accuracy) ===")
    for pt in pareto_hull:
        print(f"Cost: ${pt[0]:.4f} | Quality/Accuracy: {pt[1]:.4f}")


if __name__ == "__main__":
    asyncio.run(evaluate_benchmark())
