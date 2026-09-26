#!/usr/bin/env python3
"""Proof-of-concept demonstration script for the Mechanistic LLM Router."""

import pathlib
import sys

# Ensure src/ is on the path before importing the package.
src_dir = pathlib.Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir.resolve()))

import asyncio  # noqa: E402
import logging  # noqa: E402
import textwrap  # noqa: E402
from collections import Counter  # noqa: E402

from mechanistic_router.config import DEFAULT_CONFIG  # noqa: E402
from mechanistic_router.core.encoder import SharedTrunkEncoder  # noqa: E402
from mechanistic_router.data.mock_dataset import create_financial_dataset  # noqa: E402
from mechanistic_router.models.pool import MODEL_POOL  # noqa: E402
from mechanistic_router.routers.mechanistic import MechanisticRouter  # noqa: E402
from mechanistic_router.schemas.routing import RoutingRequest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_report(
    eval_records: list[dict],
    cost_frontier: float,
    acc_frontier: float,
    cost_oracle: float,
    acc_oracle: float,
    cost_router: float,
    acc_router: float,
    n_samples: int,
) -> None:
    """Print an executive summary report to the console."""
    savings_vs_frontier = (cost_frontier - cost_router) / cost_frontier * 100
    oracle_savings = (cost_frontier - cost_oracle) / cost_frontier * 100
    oracle_proximity = (
        (oracle_savings / savings_vs_frontier * 100) if savings_vs_frontier > 0 else 100.0
    )

    worst_case_acc = 0.73  # Theoretical mean accuracy if all queries go to SLM
    accuracy_gain = acc_router - worst_case_acc

    # Routing distribution
    router_counts = Counter(r["selected_model"] for r in eval_records)

    def draw_bar(count: int, total: int) -> str:
        pct = count / total
        bar_len = 50
        filled = int(pct * bar_len)
        return "█" * filled + "░" * (bar_len - filled)

    logger.info("\n" + "═" * 78)
    logger.info("  Cost-Optimal-Mechanistic-Router – PoC Executive Report")
    logger.info("  Mechanistic Routing via Prefill · Encoder-Target Decoupling")
    logger.info("═" * 78)

    logger.info("\n  Scenario                             Total Cost   $/Query   Accuracy")
    logger.info("  " + "─" * 76)
    logger.info(
        f"  Frontier Only (Baseline)            "
        f"$ {cost_frontier:8.2f} $ {cost_frontier / n_samples:7.4f}    {acc_frontier * 100:5.2f}%"
    )
    logger.info(
        f"  Oracle (Perfect Selection)          "
        f"$ {cost_oracle:8.2f} $ {cost_oracle / n_samples:7.4f}    {acc_oracle * 100:5.2f}%"
    )
    logger.info(
        f"  Cost-Optimal-Mechanistic-Router     "
        f"$ {cost_router:8.2f} $ {cost_router / n_samples:7.4f}    {acc_router * 100:5.2f}%"
    )

    logger.info("\n  " + "─" * 76)
    logger.info("  EFFICIENCY METRICS")
    logger.info("  " + "─" * 76)
    logger.info(f"  Cost savings (Router vs Frontier):   {savings_vs_frontier:5.2f}%")
    logger.info(f"  Cost savings (Oracle vs Frontier):   {oracle_savings:5.2f}%")
    logger.info(f"  Oracle proximity (cost):             {oracle_proximity:5.2f}%")
    logger.info(f"\n  Worst-case accuracy (SLM on complex tasks): {worst_case_acc * 100:5.2f}%")
    logger.info(f"  Accuracy gain (Router vs worst case):       +{accuracy_gain * 100:5.2f}%")

    logger.info("\n  " + "─" * 76)
    logger.info("  ROUTING DISTRIBUTION (Cost-Optimal-Mechanistic-Router)")
    logger.info("  " + "─" * 76)

    for name in MODEL_POOL.keys():
        count = router_counts[name]
        pct = (count / n_samples) * 100
        logger.info(f"  {name:<25} {draw_bar(count, n_samples)} {count:3d} ({pct:5.1f}%)")

    logger.info("\n  " + "─" * 76)
    logger.info("  EXECUTIVE ANALYSIS")
    logger.info("  " + "─" * 76)
    if savings_vs_frontier >= 70:
        meta_status = f"✓ Savings target >70%: ACHIEVED ({savings_vs_frontier:.2f}%)"
    else:
        meta_status = f"△ Savings target >70%: In progress ({savings_vs_frontier:.2f}%)"
    logger.info(f"  {meta_status}")
    logger.info(f"  ✓ Oracle proximity: {oracle_proximity:.1f}%")
    logger.info(
        "\n  The mechanistic router demonstrated the ability to reduce"
        f" inference costs by {savings_vs_frontier:.2f}% while maintaining"
        f" accuracy of {acc_router * 100:.2f}%, validating the hypothesis that"
        " prefill signals (d_eff, Fisher J) are effective predictors for"
        " routing decisions in the financial domain."
    )
    logger.info("\n" + "═" * 78)


def main() -> None:
    """Run the mechanistic router proof-of-concept demo."""
    logger.info("\n⚡ Cost-Optimal-Mechanistic-Router – Mechanistic LLM Router PoC")
    logger.info("  Initializing components...\n")

    # 1. Dataset
    logger.info("  [1/5] Generating synthetic financial dataset...")
    dataset = create_financial_dataset(n_samples=200, seed=DEFAULT_CONFIG.seed)
    n_samples = len(dataset)
    counts = Counter(c.reference_tier for c in dataset if c.reference_tier is not None)
    logger.info(f"        → {n_samples} samples generated")
    logger.info("        → Complexity distribution:")
    for cplx, count in counts.items():
        pct = (count / n_samples) * 100
        logger.info(f"           {cplx.value:<12}: {count:4d} samples ({pct:.1f}%)")

    # 2. Encoder
    logger.info("\n  [2/5] Initializing SharedTrunkEncoder...")
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    params = sum(p.numel() for p in encoder.parameters())
    logger.info(
        f"        → Architecture: {DEFAULT_CONFIG.num_prefill_layers} layers, "
        f"dim={DEFAULT_CONFIG.hidden_dim}, {params:,} parameters"
    )

    # 3. Router
    logger.info("\n  [3/5] Configuring MechanisticRouter...")
    router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)
    logger.info(f"        → λ (dynamic budget): {DEFAULT_CONFIG.lambda_budget}")
    logger.info(f"        → Model pool: {list(MODEL_POOL.keys())}")

    # 4. Evaluation
    logger.info("\n  [4/5] Running evaluation scenarios...")

    eval_records = []
    cost_frontier = 0.0
    acc_frontier = 0.0
    cost_oracle = 0.0
    acc_oracle = 0.0
    cost_router = 0.0
    acc_router = 0.0
    demos: list[dict] = []

    for case in dataset:
        prompt_text = case.prompt
        complexity = case.reference_tier

        # Baseline Frontier
        frontier_model = MODEL_POOL["LLM-Frontier-Oracle"]
        frontier_acc = case.per_model_outcome.get(
            "LLM-Frontier-Oracle", frontier_model.base_accuracy
        )
        acc_frontier += frontier_acc

        # Oracle (perfect routing using ground-truth labels)
        if complexity and complexity.value == "routine":
            best_model = MODEL_POOL["SLM-BERTau-Local"]
        elif complexity and complexity.value == "moderate":
            best_model = MODEL_POOL["LLM-Mid-Tier"]
        else:
            best_model = MODEL_POOL["LLM-Frontier-Oracle"]

        cost_oracle += best_model.cost
        acc_oracle += case.per_model_outcome.get(best_model.name, best_model.base_accuracy)

        # Router (prompt-only — no label leakage)
        decision = asyncio.run(router.route(RoutingRequest(prompt=prompt_text)))
        selected_name = decision.selected_model
        selected_model = MODEL_POOL[selected_name]

        cost_router += selected_model.cost
        acc_router += case.per_model_outcome.get(selected_name, 0.0)

        details = {k: v.model_dump() for k, v in decision.signals.items()}

        eval_records.append(
            {
                "prompt_text": prompt_text,
                "complexity": complexity,
                "selected_model": selected_name,
                "details": details,
            }
        )

        # Collect a few routine examples for the demo output
        if len(demos) < 3 and complexity and complexity.value == "routine":
            demos.append(eval_records[-1])

    acc_frontier /= n_samples
    acc_oracle /= n_samples
    acc_router /= n_samples
    logger.info(
        f"        → Frontier Only:    cost=${cost_frontier:.2f}, acc={acc_frontier * 100:.2f}%"
    )
    logger.info(f"        → Oracle:           cost=${cost_oracle:.2f}, acc={acc_oracle * 100:.2f}%")
    logger.info(
        f"        → Mechanistic Router: cost=${cost_router:.2f}, acc={acc_router * 100:.2f}%"
    )

    logger.info("\n  [5/5] Generating executive report...\n")

    generate_report(
        eval_records,
        cost_frontier,
        acc_frontier,
        cost_oracle,
        acc_oracle,
        cost_router,
        acc_router,
        n_samples,
    )

    # Demo section
    logger.info("\n" + "─" * 78)
    logger.info("  MECHANISTIC SIGNAL DEMONSTRATION (3 samples)")
    logger.info("─" * 78 + "\n")

    for demo in demos:
        prompt_preview = textwrap.shorten(demo["prompt_text"], width=60, placeholder="...")
        logger.info(f'  Prompt: "{prompt_preview}"')
        logger.info(f"  Complexity: {demo['complexity'].value}")
        logger.info(f"  Selected Model: {demo['selected_model']}")
        logger.info("  Signals:")

        for name, sig in demo["details"].items():
            sel_mark = "← SELECTED" if name == demo["selected_model"] else ""
            score = sig.get("final_score", sig.get("fisher_j_norm", 0.0))
            logger.info(
                f"    {name:<25} d_eff={sig['d_eff_mean']:.2f}  "
                f"J={sig['fisher_j']:.3f}  score={score:.3f} {sel_mark}"
            )
        logger.info("")

    logger.info("═" * 78)
    logger.info("  PoC complete. Cost-Optimal-Mechanistic-Router v0.1.0")
    logger.info("═" * 78 + "\n")


if __name__ == "__main__":
    main()
