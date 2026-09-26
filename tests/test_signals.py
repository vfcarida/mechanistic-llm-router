"""Unit Tests for Mathematical Signals and Pareto Convex Hull Functions."""

import torch

from mechanistic_router.signals.math_utils import (
    compute_convex_hull,
    compute_effective_dimensionality,
    compute_fisher_separability,
)


def test_compute_effective_dimensionality_basic() -> None:
    """Test SVD spectrum entropy calculation on rank-1 vs full-rank matrices."""
    # Rank 1 matrix -> d_eff near 1.0
    rank1_matrix = torch.ones((10, 128))
    d_eff_rank1 = compute_effective_dimensionality(rank1_matrix)
    assert 1.0 <= d_eff_rank1 < 1.5

    # Random orthogonal matrix -> d_eff higher
    torch.manual_seed(42)
    random_matrix = torch.randn((50, 128))
    d_eff_random = compute_effective_dimensionality(random_matrix)
    assert d_eff_random > d_eff_rank1


def test_compute_effective_dimensionality_edge_cases() -> None:
    """Test SVD spectrum entropy edge cases: zeros, 1D, single row, NaN, identity."""
    # All-zeros tensor -> total energy < 1e-10 -> fallback to 1.0
    zero_matrix = torch.zeros((10, 128))
    assert compute_effective_dimensionality(zero_matrix) == 1.0

    # 1D vector -> unsqueezed to (1, D) -> single singular value -> d_eff = 1.0
    vec_1d = torch.randn(128)
    assert compute_effective_dimensionality(vec_1d) == 1.0

    # Single-row tensor (1, D) -> single singular value -> d_eff = 1.0
    single_row = torch.randn((1, 128))
    assert compute_effective_dimensionality(single_row) == 1.0

    # NaN-containing tensor -> SVD numerical failure / instability -> graceful fallback to 1.0
    nan_tensor = torch.tensor([[float("nan"), 1.0], [1.0, 1.0]])
    assert compute_effective_dimensionality(nan_tensor) == 1.0

    # Identity matrix (N, N) -> uniform spectrum over N singular values -> d_eff = N
    eye_10 = torch.eye(10)
    d_eff_eye = compute_effective_dimensionality(eye_10)
    assert abs(d_eff_eye - 10.0) < 1e-4


def test_compute_fisher_separability() -> None:
    """Test Fisher Discriminant separability calculation."""
    torch.manual_seed(42)
    # Well-separated clusters
    c1 = torch.randn((20, 10)) + 5.0
    c2 = torch.randn((20, 10)) - 5.0
    j_separated = compute_fisher_separability(c1, c2)

    # Overlapping clusters
    c3 = torch.randn((20, 10))
    c4 = torch.randn((20, 10))
    j_overlapping = compute_fisher_separability(c3, c4)

    assert j_separated > j_overlapping


def test_compute_convex_hull() -> None:
    """Test Pareto Non-decreasing Convex Hull computation."""
    points = [
        (0.02, 0.50),  # SLM cheap, lower quality
        (0.25, 0.85),  # Mid tier
        (0.30, 0.80),  # Dominated (higher cost, lower quality than 0.25)
        (1.50, 0.97),  # Oracle frontier
    ]

    hull = compute_convex_hull(points)
    # Should exclude the dominated point (0.30, 0.80)
    assert len(hull) == 3
    assert (0.30, 0.80) not in hull
    assert hull[0] == (0.02, 0.50)
    assert hull[-1] == (1.50, 0.97)


def test_compute_convex_hull_edge_cases() -> None:
    """Test Pareto convex hull on empty input, single point, duplicates, and collinear points."""
    # Empty input
    assert compute_convex_hull([]) == []

    # Single point
    single = [(0.1, 0.75)]
    assert compute_convex_hull(single) == [(0.1, 0.75)]

    # Two identical points (deduplicated)
    duplicates = [(0.2, 0.8), (0.2, 0.8)]
    assert compute_convex_hull(duplicates) == [(0.2, 0.8)]

    # Points with same cost but different quality -> keeps only highest quality
    same_cost = [(0.2, 0.7), (0.2, 0.9), (0.2, 0.6)]
    assert compute_convex_hull(same_cost) == [(0.2, 0.9)]

    # All collinear points -> intermediate redundant points pruned from hull
    collinear = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert compute_convex_hull(collinear) == [(0.0, 0.0), (2.0, 2.0)]
