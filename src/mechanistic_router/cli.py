"""Unified Command-Line Interface for Mechanistic LLM Router.

Provides terminal subcommands for:
- `calibrate`: Build and persist multi-layer Group-Lasso probes and Conformal Risk Control bounds.
- `inspect`: Run white-box mechanistic routing diagnostics on any prompt without LLM API cost.
- `benchmark`: Evaluate multi-strategy cost, latency, and accuracy pareto metrics on the
  5-domain calibration benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from .config import DEFAULT_CONFIG
from .core.encoder import SharedTrunkEncoder
from .evaluation.calibration_pack import (
    build_and_save_calibration_pack,
    load_calibrated_causal_router,
    load_multidomain_calibration_dataset,
)
from .gateway.server import compute_cost_savings
from .models.pool import MODEL_POOL
from .routers.base import AbstractRouter
from .routers.cost_performance import CostPerformanceRouter
from .routers.linucb_probe import LinUCBActivationRouter
from .routers.mechanistic import MechanisticRouter
from .routers.semantic import SemanticRouter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mechanistic-router",
        description=(
            "Mechanistic LLM Router CLI: Multi-Layer Activation Probing, "
            "Conformal Risk Control, & SAE Circuit Inspection."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Fit multi-layer Group-Lasso probe and calibrate Conformal Risk Control thresholds.",
    )
    cal_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/checkpoints",
        help="Target directory for .npz probe and conformal checkpoints.",
    )
    cal_parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Target conformal risk level alpha in (0, 1).",
    )
    cal_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Per-layer residual activation hidden dimension.",
    )
    cal_parser.add_argument(
        "--l21-reg",
        type=float,
        default=0.02,
        help="Group-Lasso L2,1 regularization penalty.",
    )

    insp_parser = subparsers.add_parser(
        "inspect",
        help="Inspect white-box routing signals and SAE circuit telemetry for a prompt.",
    )
    insp_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt string to evaluate.",
    )
    insp_parser.add_argument(
        "--strategy",
        type=str,
        default="causal-probe-auto",
        choices=[
            "causal-probe-auto",
            "mechanistic-auto",
            "linucb-auto",
            "semantic-auto",
            "cost-performance-auto",
        ],
        help="Routing strategy to inspect.",
    )

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run multi-strategy routing benchmark over the 5-domain calibration dataset.",
    )
    bench_parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of benchmark prompts to evaluate.",
    )

    return parser


def _resolve_strategy(strategy: str) -> AbstractRouter:
    strat = strategy.lower().strip()
    if strat == "mechanistic-auto":
        return MechanisticRouter(SharedTrunkEncoder(DEFAULT_CONFIG), MODEL_POOL, DEFAULT_CONFIG)
    if strat == "linucb-auto":
        return LinUCBActivationRouter(MODEL_POOL, DEFAULT_CONFIG)
    if strat == "semantic-auto":
        return SemanticRouter(MODEL_POOL, DEFAULT_CONFIG)
    if strat == "cost-performance-auto":
        return CostPerformanceRouter(MODEL_POOL, DEFAULT_CONFIG)
    return load_calibrated_causal_router(MODEL_POOL, DEFAULT_CONFIG)


def _cmd_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    from pathlib import Path

    from .schemas.routing import RoutingRequest as _UnusedReq  # noqa: F401

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_file = out_dir / "multilayer_group_lasso_probe.npz"
    conformal_file = out_dir / "conformal_risk_controller.npz"

    probe, _controller, summary = build_and_save_calibration_pack(
        probe_path=probe_file,
        conformal_path=conformal_file,
        alpha=args.alpha,
        group_lasso_lambda=args.l21_reg,
    )
    return {
        "status": "calibrated",
        "output_dir": str(out_dir),
        "alpha": summary.alpha,
        "low_threshold": summary.low_threshold,
        "high_threshold": summary.high_threshold,
        "empirical_risk_low": summary.empirical_risk_low,
        "upper_risk_bound_low": summary.upper_risk_bound_low,
        "empirical_risk_high": summary.empirical_risk_high,
        "upper_risk_bound_high": summary.upper_risk_bound_high,
        "calibration_size": summary.calibration_size,
        "group_importances": [float(x) for x in probe.get_group_importances()],
    }


def _cmd_inspect(args: argparse.Namespace) -> dict[str, Any]:
    from .schemas.routing import RoutingRequest

    router_inst = _resolve_strategy(args.strategy)
    decision = asyncio.run(router_inst.route(RoutingRequest(prompt=args.prompt)))
    cost_saved = compute_cost_savings(decision.selected_model, MODEL_POOL)
    selected_sig = decision.signals[decision.selected_model]
    return {
        "selected_model": decision.selected_model,
        "strategy_used": decision.strategy_used,
        "estimated_cost_usd": decision.estimated_cost_usd,
        "cost_saved_usd": cost_saved,
        "latency_ms": round(decision.latency_ms, 4),
        "sae_features": selected_sig.sae_features,
        "metadata": selected_sig.extra_metadata,
    }


def _cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from .schemas.routing import RoutingRequest

    dataset = load_multidomain_calibration_dataset()
    subset = dataset[: max(1, min(int(args.samples), len(dataset)))]

    strategies: dict[str, AbstractRouter] = {
        "causal-probe-auto": load_calibrated_causal_router(MODEL_POOL, DEFAULT_CONFIG),
        "linucb-auto": LinUCBActivationRouter(MODEL_POOL, DEFAULT_CONFIG),
        "mechanistic-auto": MechanisticRouter(
            SharedTrunkEncoder(DEFAULT_CONFIG), MODEL_POOL, DEFAULT_CONFIG
        ),
        "semantic-auto": SemanticRouter(MODEL_POOL, DEFAULT_CONFIG),
        "cost-performance-auto": CostPerformanceRouter(MODEL_POOL, DEFAULT_CONFIG),
    }

    report: dict[str, Any] = {"samples_evaluated": len(subset), "strategies": {}}
    for strat_name, r_inst in strategies.items():
        total_cost = 0.0
        total_saved = 0.0
        total_latency = 0.0
        tier_counts: dict[str, int] = {}
        for sample in subset:
            dec = asyncio.run(r_inst.route(RoutingRequest(prompt=sample.prompt)))
            total_cost += dec.estimated_cost_usd
            total_saved += compute_cost_savings(dec.selected_model, MODEL_POOL)
            total_latency += dec.latency_ms
            tier_counts[dec.selected_model] = tier_counts.get(dec.selected_model, 0) + 1

        n = len(subset)
        report["strategies"][strat_name] = {
            "mean_cost_usd": round(total_cost / n, 6),
            "mean_cost_saved_usd": round(total_saved / n, 6),
            "mean_latency_ms": round(total_latency / n, 4),
            "tier_distribution": tier_counts,
        }

    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `mechanistic-router` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        result = _cmd_calibrate(args)
    elif args.command == "inspect":
        result = _cmd_inspect(args)
    else:
        result = _cmd_benchmark(args)

    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
