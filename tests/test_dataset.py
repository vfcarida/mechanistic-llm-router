"""Unit Tests for Financial Dataset Generator."""

import pytest
import pandas as pd
from mechanistic_router.data.mock_dataset import create_financial_dataset
from mechanistic_router.models.types import TaskComplexity


def test_create_financial_dataset_structure() -> None:
    """Test financial dataset generation schema and columns."""
    df = create_financial_dataset(n_samples=50, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert list(df.columns) == ["prompt_id", "prompt_text", "category", "complexity"]


def test_create_financial_dataset_distribution() -> None:
    """Test target complexity distribution values."""
    df = create_financial_dataset(n_samples=100, seed=42)
    complexities = df["complexity"].unique()
    assert TaskComplexity.ROUTINE in complexities
    assert TaskComplexity.MODERATE in complexities
    assert TaskComplexity.COMPLEX in complexities
