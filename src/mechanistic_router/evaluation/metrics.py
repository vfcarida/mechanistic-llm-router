"""Pareto Frontier, Trade-off Interpolation, and Bootstrap Metrics."""

from typing import Any

import numpy as np

from ..signals.math_utils import compute_convex_hull


def compute_pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Calculates non-dominated Pareto frontier points on the (cost, accuracy) plane.

    A point (c1, a1) dominates (c2, a2) if c1 <= c2 and a1 >= a2 (with at least one strict).
    Leverages mathematical convex hull to ensure monotonic non-decreasing quality trade-offs.

    Args:
        points: List of (cost, accuracy) coordinates.

    Returns:
        Sorted list of non-dominated (cost, accuracy) tuples.
    """
    if not points:
        return []
    return compute_convex_hull(points)


def compute_matched_cost_accuracy(
    pareto_points: list[tuple[float, float]], target_cost: float
) -> float:
    """Computes achievable quality/accuracy at an exact target cost budget via interpolation.

    Args:
        pareto_points: Pareto frontier points sorted by cost.
        target_cost: Evaluation budget constraint in USD.

    Returns:
        Interpolated accuracy float at target_cost.
    """
    if not pareto_points:
        return 0.0

    sorted_pts = sorted(pareto_points, key=lambda p: p[0])
    if target_cost <= sorted_pts[0][0]:
        return sorted_pts[0][1]
    if target_cost >= sorted_pts[-1][0]:
        return sorted_pts[-1][1]

    for i in range(len(sorted_pts) - 1):
        c1, a1 = sorted_pts[i]
        c2, a2 = sorted_pts[i + 1]
        if c1 <= target_cost <= c2:
            if abs(c2 - c1) < 1e-10:
                return max(a1, a2)
            alpha = (target_cost - c1) / (c2 - c1)
            return a1 + alpha * (a2 - a1)

    return sorted_pts[-1][1]


def compute_matched_accuracy_cost(
    pareto_points: list[tuple[float, float]], target_accuracy: float
) -> float:
    """Computes required USD cost to achieve an exact target accuracy level via interpolation.

    Args:
        pareto_points: Pareto frontier points sorted by quality.
        target_accuracy: Target quality/accuracy threshold in [0, 1].

    Returns:
        Interpolated cost in USD required to reach target_accuracy.
    """
    if not pareto_points:
        return 0.0

    sorted_pts = sorted(pareto_points, key=lambda p: (p[1], p[0]))
    if target_accuracy <= sorted_pts[0][1]:
        return sorted_pts[0][0]
    if target_accuracy >= sorted_pts[-1][1]:
        return sorted_pts[-1][0]

    for i in range(len(sorted_pts) - 1):
        c1, a1 = sorted_pts[i]
        c2, a2 = sorted_pts[i + 1]
        if a1 <= target_accuracy <= a2:
            if abs(a2 - a1) < 1e-10:
                return min(c1, c2)
            alpha = (target_accuracy - a1) / (a2 - a1)
            return c1 + alpha * (c2 - c1)

    return sorted_pts[-1][0]


def compute_pareto_auc(
    pareto_points: list[tuple[float, float]],
    cost_min: float | None = None,
    cost_max: float | None = None,
) -> float:
    """Calculates the normalized Area Under the Pareto Curve (quality integrated over cost).

    Args:
        pareto_points: Non-dominated Pareto frontier points.
        cost_min: Lower bound cost (defaults to min point cost).
        cost_max: Upper bound cost (defaults to max point cost).

    Returns:
        Normalized area in [0, 1].
    """
    if len(pareto_points) < 2:
        return pareto_points[0][1] if pareto_points else 0.0

    sorted_pts = sorted(pareto_points, key=lambda p: p[0])
    c_min = cost_min if cost_min is not None else sorted_pts[0][0]
    c_max = cost_max if cost_max is not None else sorted_pts[-1][0]

    if c_max <= c_min:
        return sorted_pts[0][1]

    # Trapezoidal integration across cost steps
    area = 0.0
    for i in range(len(sorted_pts) - 1):
        c1, a1 = sorted_pts[i]
        c2, a2 = sorted_pts[i + 1]
        step_area = 0.5 * (a1 + a2) * (c2 - c1)
        area += step_area

    normalized_auc = area / (c_max - c_min)
    return float(np.clip(normalized_auc, 0.0, 1.0))


def compute_paired_bootstrap_ci(
    candidate_costs: list[float] | np.ndarray,
    candidate_accs: list[float] | np.ndarray,
    baseline_costs: list[float] | np.ndarray,
    baseline_accs: list[float] | np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes paired bootstrap empirical 95% confidence intervals over identical test cases.

    Args:
        candidate_costs: Query-level costs incurred by candidate router.
        candidate_accs: Query-level accuracy outcomes of candidate router.
        baseline_costs: Query-level costs incurred by baseline (e.g. AlwaysCheap).
        baseline_accs: Query-level accuracy outcomes of baseline.
        n_bootstraps: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing means, paired delta statistics, and confidence intervals.
    """
    c_cand = np.asarray(candidate_costs, dtype=np.float64)
    a_cand = np.asarray(candidate_accs, dtype=np.float64)
    c_base = np.asarray(baseline_costs, dtype=np.float64)
    a_base = np.asarray(baseline_accs, dtype=np.float64)

    n = len(c_cand)
    if n == 0:
        return {}

    rng = np.random.RandomState(seed)

    indices = rng.randint(0, n, size=(n_bootstraps, n))
    boot_c_cand = np.mean(c_cand[indices], axis=1)
    boot_a_cand = np.mean(a_cand[indices], axis=1)
    boot_c_base = np.mean(c_base[indices], axis=1)
    boot_a_base = np.mean(a_base[indices], axis=1)
    boot_delta_a = boot_a_cand - boot_a_base
    boot_delta_c = boot_c_cand - boot_c_base

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    delta_a_ci = (
        float(np.percentile(boot_delta_a, lower_pct)),
        float(np.percentile(boot_delta_a, upper_pct)),
    )
    delta_c_ci = (
        float(np.percentile(boot_delta_c, lower_pct)),
        float(np.percentile(boot_delta_c, upper_pct)),
    )

    return {
        "candidate_cost_mean": float(np.mean(c_cand)),
        "candidate_cost_ci": (
            float(np.percentile(boot_c_cand, lower_pct)),
            float(np.percentile(boot_c_cand, upper_pct)),
        ),
        "candidate_acc_mean": float(np.mean(a_cand)),
        "candidate_acc_ci": (
            float(np.percentile(boot_a_cand, lower_pct)),
            float(np.percentile(boot_a_cand, upper_pct)),
        ),
        "baseline_cost_mean": float(np.mean(c_base)),
        "baseline_cost_ci": (
            float(np.percentile(boot_c_base, lower_pct)),
            float(np.percentile(boot_c_base, upper_pct)),
        ),
        "baseline_acc_mean": float(np.mean(a_base)),
        "baseline_acc_ci": (
            float(np.percentile(boot_a_base, lower_pct)),
            float(np.percentile(boot_a_base, upper_pct)),
        ),
        "delta_acc_mean": float(np.mean(a_cand) - np.mean(a_base)),
        "delta_acc_ci": delta_a_ci,
        "delta_cost_mean": float(np.mean(c_cand) - np.mean(c_base)),
        "delta_cost_ci": delta_c_ci,
        # Statistically significant quality improvement above baseline if lower bound > 0
        "is_significant_quality_gain": delta_a_ci[0] > 0.0,
        # Pareto dominance over baseline: strictly better on at least one
        # without being worse on the other
        "pareto_dominates_baseline": (
            delta_a_ci[0] >= 0.0
            and delta_c_ci[1] <= 0.0
            and (delta_a_ci[0] > 0.0 or delta_c_ci[1] < 0.0)
        ),
    }


__all__ = [
    "compute_pareto_front",
    "compute_matched_cost_accuracy",
    "compute_matched_accuracy_cost",
    "compute_pareto_auc",
    "compute_paired_bootstrap_ci",
]
