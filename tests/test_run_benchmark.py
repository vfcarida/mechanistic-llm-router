"""Unit and Integration Tests for CLI Benchmark Runner (scripts/run_benchmark.py)."""

import sys
from pathlib import Path
from unittest.mock import patch

from scripts.run_benchmark import main, parse_args


def test_parse_args_defaults() -> None:
    """Test parse_args with default CLI parameters."""
    with patch.object(sys, "argv", ["run_benchmark.py"]):
        args = parse_args()
        assert args.dataset == "synthetic"
        assert args.samples == 300
        assert args.seed == 42
        assert args.bootstraps == 1000
        assert args.all_policies is False
        assert args.output == "docs/BENCHMARK_REPORT.md"


def test_parse_args_custom_flags() -> None:
    """Test parse_args with custom flags including --all-policies."""
    custom_argv = [
        "run_benchmark.py",
        "--dataset",
        "synthetic",
        "--samples",
        "50",
        "--bootstraps",
        "20",
        "--all-policies",
        "--output",
        "custom_out.md",
    ]
    with patch.object(sys, "argv", custom_argv):
        args = parse_args()
        assert args.samples == 50
        assert args.bootstraps == 20
        assert args.all_policies is True
        assert args.output == "custom_out.md"


def test_main_cli_execution_synthetic(tmp_path: Path) -> None:
    """Test end-to-end execution of main() generating report on synthetic data."""
    output_file = tmp_path / "test_benchmark_report.md"
    test_argv = [
        "run_benchmark.py",
        "--dataset",
        "synthetic",
        "--samples",
        "15",
        "--bootstraps",
        "5",
        "--output",
        str(output_file),
    ]
    with patch.object(sys, "argv", test_argv):
        exit_code = main()
        assert exit_code == 0

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "# Baseline Pareto Benchmark & Go/No-Go Gate Report" in content
    assert "Gate Decision" in content
    assert "Cost-Quality Pareto Comparison Table" in content


def test_main_cli_execution_all_policies(tmp_path: Path) -> None:
    """Test end-to-end execution of main() with --all-policies evaluating all 9 baselines."""
    output_file = tmp_path / "test_all_policies_report.md"
    test_argv = [
        "run_benchmark.py",
        "--dataset",
        "synthetic",
        "--samples",
        "15",
        "--bootstraps",
        "5",
        "--all-policies",
        "--output",
        str(output_file),
    ]
    with patch.object(sys, "argv", test_argv):
        exit_code = main()
        assert exit_code == 0

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "CostPerformanceRouter" in content
    assert "SemanticRouter" in content
    assert "CausalProbeRouter" in content
