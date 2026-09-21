"""Unit Tests for Financial Dataset Generator emitting EvalCases."""

import importlib

import pytest

from mechanistic_router.models.types import TaskComplexity
from mechanistic_router.schemas.eval import EvalCase
from tests.fixtures.mock_dataset import create_financial_dataset


def test_create_financial_dataset_structure() -> None:
    """Test financial dataset generation schema and EvalCase structure."""
    cases = create_financial_dataset(n_samples=50, seed=42)
    assert isinstance(cases, list)
    assert len(cases) == 50
    assert all(isinstance(c, EvalCase) for c in cases)

    first = cases[0]
    assert isinstance(first.prompt, str) and len(first.prompt) > 0
    assert isinstance(first.per_model_outcome, dict)
    assert "SLM-BERTau-Local" in first.per_model_outcome
    assert "LLM-Frontier-Oracle" in first.per_model_outcome
    assert isinstance(first.price_table, dict)
    assert first.reference_tier in (
        TaskComplexity.ROUTINE,
        TaskComplexity.MODERATE,
        TaskComplexity.COMPLEX,
    )


def test_create_financial_dataset_distribution() -> None:
    """Test target reference complexity distribution values."""
    cases = create_financial_dataset(n_samples=100, seed=42)
    tiers = {c.reference_tier for c in cases}
    assert TaskComplexity.ROUTINE in tiers
    assert TaskComplexity.MODERATE in tiers
    assert TaskComplexity.COMPLEX in tiers


def test_mock_dataset_deprecation_warning() -> None:
    """Test that importing mechanistic_router.data.mock_dataset emits DeprecationWarning."""
    import mechanistic_router.data.mock_dataset as deprecated_mock_dataset

    with pytest.deprecated_call():
        importlib.reload(deprecated_mock_dataset)
