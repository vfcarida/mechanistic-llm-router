"""Unit Tests for Metrics Normalization Helpers."""

import pytest
from mechanistic_router.utils.metrics import normalized_accuracy, normalized_inverse_cost


def test_normalized_inverse_cost() -> None:
    """Test inverse cost normalization mapping."""
    score_min = normalized_inverse_cost(0.02, 0.02, 1.50)
    score_max = normalized_inverse_cost(1.50, 0.02, 1.50)

    assert score_min == 1.0  # Cheapest model gets highest inverse cost score
    assert score_max == 0.0  # Most expensive gets lowest score

    with pytest.raises(ValueError, match="strictly positive"):
        normalized_inverse_cost(0.0, 0.02, 1.50)


def test_normalized_accuracy() -> None:
    """Test accuracy score linear normalization."""
    score = normalized_accuracy(0.88, 0.40, 0.97)
    assert 0.0 <= score <= 1.0
