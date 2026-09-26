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
