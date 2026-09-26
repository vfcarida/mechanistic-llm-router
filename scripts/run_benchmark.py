"""CLI Runner for Zero-Leakage Pareto Benchmark and Go/No-Go Gate Generation."""

import argparse
import os
import sys

from mechanistic_router.evaluation import (
    AlwaysCheapPolicy,
    AlwaysStrongPolicy,
    BenchmarkHarness,
    CausalProbePolicy,
    CostPerformancePolicy,
    LearnedLogisticPolicy,
    LengthThresholdPolicy,
    MechanisticPolicy,
    RandomPolicy,
    SemanticPolicy,
    load_routerbench_eval_dataset,
    load_synthetic_eval_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pareto Benchmark and generate Go/No-Go Report."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "routerbench"],
        default="synthetic",
        help="Dataset provider to evaluate on (default: synthetic).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/routerbench_eval.jsonl",
        help="Path to precomputed RouterBench benchmark file (when --dataset routerbench).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of synthetic evaluation samples to generate (default: 300).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1000,
        help="Number of paired bootstrap iterations (default: 1000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/BENCHMARK_REPORT.md",
        help="Output filepath for generated benchmark report.",
    )
    parser.add_argument(
        "--all-policies",
        action="store_true",
        help="Evaluate all 9 baseline policies (including CostPerformance, Semantic, CausalProbe).",
    )
    return parser.parse_args()


def format_report_markdown(summary, dataset_type: str) -> str:
    """Formats benchmark execution summary into comprehensive GitHub markdown report."""
    md = []
    md.append("# Baseline Pareto Benchmark & Go/No-Go Gate Report (MLR-T05)\n")

    if summary.is_synthetic:
        md.append("> [!WARNING]")
        md.append("> **Data Provenance**: Evaluated on synthetic financial prompt data. ")
        md.append(
            "> These figures are **illustrative** baseline demonstrations and do **NOT**\n"
            "> constitute empirical real-world validation. Results must be re-run on\n"
            "> RouterBench (arXiv:2403.12031) or production traces prior to deployment.\n"
        )
    else:
        md.append("> [!NOTE]")
        md.append("> **Data Provenance**: Evaluated on precomputed RouterBench dataset traces.\n")

    md.append("## 1. Executive Go/No-Go Decision Gate\n")
    md.append(f"**Gate Decision**: `{summary.go_no_go_decision}`\n")
    md.append(f"**Justification**: {summary.go_no_go_rationale}\n")

    md.append("## 2. Methodology & Leakage Isolation\n")
    md.append("- **Prompt-Disjoint Partitioning**: Dataset partitioned into disjoint sets:")
    md.append(f"  - **Train split**: {summary.n_train} cases (used for LearnedLogistic fitting).")
    md.append(f"  - **Dev split**: {summary.n_dev} cases (used for LengthThreshold tuning).")
    md.append(f"  - **Test split**: {summary.n_test} cases (used for unbiased Pareto evaluation).")
    md.append(
        "- **Zero-Leakage Guarantee**: Router policies receive only `prompt` text;\n"
        "  eval-only ground truth (`reference_tier`, `per_model_outcome`) is completely isolated."
    )
    md.append(
        "- **Uncertainty Quantification**: 1,000 paired bootstrap iterations over\n"
        "  identical test queries yielding empirical 95% confidence intervals.\n"
    )

    md.append("## 3. Cost-Quality Pareto Comparison Table\n")
    md.append(
        "| Policy | Mean Cost ($) [95% CI] | Mean Accuracy (%) [95% CI] | "
        "Δ Acc vs Cheap (%) [95% CI] | Δ Cost vs Cheap ($) [95% CI] | "
        "Denominators (N, Fail) | Pareto Front? |"
    )
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in summary.results:
        cost_str = f"${r.mean_cost:.4f} [${r.cost_ci[0]:.4f}, ${r.cost_ci[1]:.4f}]"
        acc_str = (
            f"{r.mean_accuracy * 100:.2f}% "
            f"[{r.accuracy_ci[0] * 100:.2f}%, {r.accuracy_ci[1] * 100:.2f}%]"
        )
        delta_acc_str = (
            f"{r.delta_accuracy * 100:+.2f}% "
            f"[{r.delta_accuracy_ci[0] * 100:+.2f}%, {r.delta_accuracy_ci[1] * 100:+.2f}%]"
        )
        delta_cost_str = (
            f"${r.delta_cost:+.4f} [${r.delta_cost_ci[0]:+.4f}, ${r.delta_cost_ci[1]:+.4f}]"
        )
        pareto_str = "**Yes**" if r.is_on_pareto_front else "No"
        denom_str = f"{r.n_evaluated} ({r.n_failures})"

        md.append(
            f"| {r.policy_name} | {cost_str} | {acc_str} | "
            f"{delta_acc_str} | {delta_cost_str} | {denom_str} | {pareto_str} |"
        )

    md.append("\n## 4. Pareto Frontier Analysis\n")
    md.append(f"- **Normalized Pareto AUC**: `{summary.pareto_auc:.4f}`")
    md.append("- **Mathematical Convex Hull Frontier Points (Cost, Accuracy)**:")
    for pt in summary.pareto_frontier:
        md.append(f"  - Cost: `${pt[0]:.4f}` → Accuracy: `{pt[1] * 100:.2f}%`")

    md.append("\n### Matched Trade-off Analysis")
    for key, val in summary.matched_cost_quality.items():
        md.append(f"- Quality achievable {key.replace('_', ' ')}: `{val * 100:.2f}%`")

    md.append("\n---\n")
    md.append(
        "*Generated by `scripts/run_benchmark.py` adhering to "
        "MLR-T05 specifications and SHARED-CONVENTIONS.md.*"
    )

    return "\n".join(md) + "\n"


def main() -> int:
    args = parse_args()

    print(f"=== Starting MLR-T05 Pareto Benchmark (Dataset: {args.dataset}) ===")
    if args.dataset == "synthetic":
        cases = load_synthetic_eval_dataset(n_samples=args.samples, seed=args.seed)
        is_synthetic = True
    else:
        cases = load_routerbench_eval_dataset(args.dataset_path)
        is_synthetic = False

    print(f"Loaded {len(cases)} evaluation records.")
    if args.all_policies:
        policies = [
            AlwaysCheapPolicy(),
            AlwaysStrongPolicy(),
            RandomPolicy(seed=args.seed),
            LengthThresholdPolicy(),
            LearnedLogisticPolicy(seed=args.seed),
            MechanisticPolicy(),
            CostPerformancePolicy(),
            SemanticPolicy(),
            CausalProbePolicy(),
        ]
        harness = BenchmarkHarness(policies=policies, seed=args.seed)
    else:
        harness = BenchmarkHarness(seed=args.seed)

    summary = harness.evaluate(cases=cases, is_synthetic=is_synthetic, n_bootstraps=args.bootstraps)

    report_content = format_report_markdown(summary, args.dataset)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("Benchmark completed successfully.")
    print(f"Report written to: {args.output}")
    print(f"\n[Go/No-Go Decision]: {summary.go_no_go_decision}")
    print(f"[Rationale]: {summary.go_no_go_rationale}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
