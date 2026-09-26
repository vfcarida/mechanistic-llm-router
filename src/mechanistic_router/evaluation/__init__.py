"""Evaluation, Pareto Benchmarking, and Baselines Module."""

from .baselines import (
    AlwaysCheapPolicy,
    AlwaysStrongPolicy,
    BasePolicy,
    CausalProbePolicy,
    CostPerformancePolicy,
    LearnedLogisticPolicy,
    LengthThresholdPolicy,
    MechanisticPolicy,
    RandomPolicy,
    SemanticPolicy,
)
from .dataset import (
    load_routerbench_eval_dataset,
    load_synthetic_eval_dataset,
    split_dataset_prompt_disjoint,
)
from .harness import (
    BenchmarkExecutionSummary,
    BenchmarkHarness,
    PolicyEvaluationResult,
)
from .metrics import (
    compute_matched_accuracy_cost,
    compute_matched_cost_accuracy,
    compute_paired_bootstrap_ci,
    compute_pareto_auc,
    compute_pareto_front,
)

__all__ = [
    "BasePolicy",
    "AlwaysCheapPolicy",
    "AlwaysStrongPolicy",
    "RandomPolicy",
    "LengthThresholdPolicy",
    "LearnedLogisticPolicy",
    "MechanisticPolicy",
    "CostPerformancePolicy",
    "SemanticPolicy",
    "CausalProbePolicy",
    "split_dataset_prompt_disjoint",
    "load_synthetic_eval_dataset",
    "load_routerbench_eval_dataset",
    "compute_pareto_front",
    "compute_matched_cost_accuracy",
    "compute_matched_accuracy_cost",
    "compute_pareto_auc",
    "compute_paired_bootstrap_ci",
    "PolicyEvaluationResult",
    "BenchmarkExecutionSummary",
    "BenchmarkHarness",
]
