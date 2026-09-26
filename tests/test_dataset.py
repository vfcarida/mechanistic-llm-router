"""Unit Tests for Financial Dataset Generator emitting EvalCases."""

import importlib
import json
from pathlib import Path

import pytest

from mechanistic_router.evaluation.dataset import load_routerbench_eval_dataset
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


def test_load_routerbench_eval_dataset_file_not_found() -> None:
    """Test that missing RouterBench dataset path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="RouterBench dataset file not found"):
        load_routerbench_eval_dataset("non_existent_routerbench_path.jsonl")


def test_load_routerbench_eval_dataset_json_and_jsonl(tmp_path: Path) -> None:
    """Test loading precomputed evaluation cases from both JSON and JSONL formats."""
    sample_records = [
        {
            "prompt": "Evaluate market liquidity",
            "per_model_outcome": {"SLM-BERTau-Local": 0.85, "LLM-Frontier-Oracle": 0.98},
            "price_table": {"SLM-BERTau-Local": 0.02, "LLM-Frontier-Oracle": 1.50},
            "reference_tier": "complex",
        },
        {
            "prompt": "Consult account balance",
            "per_model_outcome": {"SLM-BERTau-Local": 0.95, "LLM-Frontier-Oracle": 0.99},
            "price_table": {"SLM-BERTau-Local": 0.02, "LLM-Frontier-Oracle": 1.50},
            "reference_tier": "routine",
        },
    ]

    # Test JSON format
    json_file = tmp_path / "benchmark.json"
    json_file.write_text(json.dumps(sample_records), encoding="utf-8")
    loaded_json = load_routerbench_eval_dataset(str(json_file))
    assert len(loaded_json) == 2
    assert all(isinstance(c, EvalCase) for c in loaded_json)
    assert loaded_json[0].prompt == "Evaluate market liquidity"
    assert loaded_json[1].reference_tier == TaskComplexity.ROUTINE

    # Test JSONL format
    jsonl_file = tmp_path / "benchmark.jsonl"
    jsonl_content = "\n".join(json.dumps(r) for r in sample_records) + "\n"
    jsonl_file.write_text(jsonl_content, encoding="utf-8")
    loaded_jsonl = load_routerbench_eval_dataset(str(jsonl_file))
    assert len(loaded_jsonl) == 2
    assert all(isinstance(c, EvalCase) for c in loaded_jsonl)
    assert loaded_jsonl[0].prompt == "Evaluate market liquidity"


def test_load_routerbench_eval_dataset_malformed(tmp_path: Path) -> None:
    """Test that malformed JSONL content raises json.JSONDecodeError."""
    corrupt_file = tmp_path / "corrupt.jsonl"
    corrupt_file.write_text("{invalid_json_line\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_routerbench_eval_dataset(str(corrupt_file))
