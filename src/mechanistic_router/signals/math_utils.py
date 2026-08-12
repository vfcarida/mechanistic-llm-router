"""Mathematical Signals and Pareto Optimization Utility Functions."""

import numpy as np
import torch


def compute_effective_dimensionality(activation: torch.Tensor) -> float:
    """Calculates Effective Dimensionality (d_eff) of a latent activation matrix.

    Estimates information processing spread across latent orthogonal dimensions.
    Dense/complex prompts activate multiple orthogonal dimensions, while trivial prompts
    collapse into few dominant singular values.

    Algorithm:
    1. Computes Singular Value Decomposition (SVD).
    2. Squares singular values to obtain Spectral Energy.
    3. Normalizes Energy into a probability distribution.
    4. Computes Shannon Entropy (H) over the distribution.
    5. Returns exp(H) representing number of effective dimensions.

    Args:
        activation: Latent activation tensor [N, D] extracted from encoder prefill.

    Returns:
        Continuous scalar d_eff >= 1.0. Returns 1.0 as safe fallback on singular instability.
    """
    if activation.ndim == 1:
        activation = activation.unsqueeze(0)

    try:
        _, s, _ = torch.linalg.svd(activation, full_matrices=False)
    except RuntimeError:
        return 1.0

    energy = s**2
    total_energy = torch.sum(energy)
    if total_energy < 1e-10:
        return 1.0

    p = energy / total_energy
    p = p[p > 1e-10]
    entropy = -torch.sum(p * torch.log(p))
    d_eff = torch.exp(entropy)
    return float(d_eff.item())


def compute_fisher_separability(
    success_activations: torch.Tensor,
    failure_activations: torch.Tensor,
) -> float:
    """Calculates Fisher Discriminant Separability Criterion J between two activation clusters.

    Evaluates how well a target model separates success vs failure patterns for a prompt.
    If inter-class variance (centroid distance) exceeds intra-class scatter (internal dispersion),
    Fisher J will be high, confirming mechanistic competence.

    Args:
        success_activations: Success class activations matrix [N, D].
        failure_activations: Failure class activations matrix [N, D].

    Returns:
        Fisher Discriminant ratio J >= 0.0.
    """
    mu_success = success_activations.mean(dim=0)
    mu_failure = failure_activations.mean(dim=0)

    var_success = success_activations.var(dim=0, unbiased=False)
    var_failure = failure_activations.var(dim=0, unbiased=False)

    within_class_scatter = var_success + var_failure + 1e-10
    fisher_per_dim = (mu_success - mu_failure) ** 2 / within_class_scatter
    return float(fisher_per_dim.mean().item())


def compute_convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Computes the Non-decreasing Upper Convex Hull for Pareto Cost-Quality Frontiers.

    Graphs optimal trade-off bounds mapping cost (x-axis) to quality/accuracy (y-axis).
    Filters dominated strategy points so that higher cost strictly guarantees monotonically non-decreasing quality.

    Args:
        points: List of (cost, quality) tuples.

    Returns:
        Sorted list of Pareto-optimal non-dominated (cost, quality) points.
    """
    if not points:
        return []

    # Sort points primarily by cost ascending, secondarily by quality descending
    sorted_points = sorted(points, key=lambda p: (p[0], -p[1]))
    hull: list[tuple[float, float]] = []

    for cost, quality in sorted_points:
        # If current point offers equal or lower quality than existing best for same/less cost, skip
        if hull and quality <= hull[-1][1]:
            continue

        # Maintain convex hull slope property
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = cost, quality
            # Cross product test for upper convex boundary
            if (y2 - y1) * (x3 - x2) <= (y3 - y2) * (x2 - x1):
                hull.pop()
            else:
                break

        hull.append((cost, quality))

    return hull
