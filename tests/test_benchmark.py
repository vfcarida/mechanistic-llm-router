"""Unit and Integration Tests for MLR-T05 Baseline Pareto Benchmark and Policies."""

from mechanistic_router.evaluation.baselines import (
    AlwaysCheapPolicy,
    AlwaysStrongPolicy,
    LearnedLogisticPolicy,
    LengthThresholdPolicy,
    MechanisticPolicy,
    RandomPolicy,
)
from mechanistic_router.evaluation.dataset import (
    load_synthetic_eval_dataset,
    split_dataset_prompt_disjoint,
)
from mechanistic_router.evaluation.harness import BenchmarkHarness
from mechanistic_router.evaluation.metrics import (
    compute_matched_accuracy_cost,
    compute_matched_cost_accuracy,
    compute_paired_bootstrap_ci,
    compute_pareto_auc,
    compute_pareto_front,
)
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.schemas.eval import ComplexityTier, EvalCase


def test_pareto_front_toy_fixture() -> None:
    """Verify Pareto frontier correctly removes dominated points on toy coordinates."""
    # (0.3, 0.6) is dominated by (0.2, 0.7) which has lower cost and higher accuracy
    points = [(0.1, 0.5), (0.2, 0.7), (0.3, 0.6), (0.5, 0.9)]
    front = compute_pareto_front(points)

    assert (0.1, 0.5) in front
    assert (0.2, 0.7) in front
    assert (0.5, 0.9) in front
    assert (0.3, 0.6) not in front


def test_matched_cost_and_accuracy_interpolation() -> None:
    """Verify matched-cost accuracy and matched-accuracy cost interpolation functions."""
    pareto_pts = [(0.10, 0.60), (0.30, 0.80), (0.50, 0.90)]

    # Interpolation at mid cost (0.20 is halfway between 0.10 and 0.30 -> acc = 0.70)
    acc_mid = compute_matched_cost_accuracy(pareto_pts, 0.20)
    assert round(acc_mid, 4) == 0.7000

    # Bounds clipping
    assert compute_matched_cost_accuracy(pareto_pts, 0.05) == 0.60
    assert compute_matched_cost_accuracy(pareto_pts, 0.90) == 0.90

    # Interpolation at target accuracy (0.70 is halfway between 0.60 and 0.80 -> cost = 0.20)
    cost_mid = compute_matched_accuracy_cost(pareto_pts, 0.70)
    assert round(cost_mid, 4) == 0.2000


def test_pareto_auc_calculation() -> None:
    """Verify normalized Pareto AUC calculation over bounded domain."""
    points = [(0.0, 0.5), (1.0, 0.9)]
    # Trapezoid area = 0.5 * (0.5 + 0.9) * 1.0 = 0.70
    auc = compute_pareto_auc(points, cost_min=0.0, cost_max=1.0)
    assert round(auc, 4) == 0.7000


def test_bootstrap_ci_reproducibility() -> None:
    """Verify paired bootstrap returns identical intervals under fixed random seed."""
    c_cand = [0.05, 0.10, 0.25, 0.05, 0.25]
    a_cand = [0.85, 0.90, 0.95, 0.85, 0.95]
    c_base = [0.02, 0.02, 0.02, 0.02, 0.02]
    a_base = [0.70, 0.70, 0.70, 0.70, 0.70]

    res1 = compute_paired_bootstrap_ci(c_cand, a_cand, c_base, a_base, n_bootstraps=200, seed=123)
    res2 = compute_paired_bootstrap_ci(c_cand, a_cand, c_base, a_base, n_bootstraps=200, seed=123)

    assert res1["candidate_acc_ci"] == res2["candidate_acc_ci"]
    assert res1["delta_acc_ci"] == res2["delta_acc_ci"]
    assert res1["delta_cost_ci"] == res2["delta_cost_ci"]


def test_leakage_guard_policy_signatures() -> None:
    """Verify all policies accept only prompt string and have no access to eval labels."""
    policies = [
        AlwaysCheapPolicy(),
        AlwaysStrongPolicy(),
        RandomPolicy(seed=42),
        LengthThresholdPolicy(),
        LearnedLogisticPolicy(seed=42),
        MechanisticPolicy(),
    ]

    toy_prompt = "Qual meu saldo em conta corrente?"
    for policy in policies:
        # Must execute with pure string prompt only
        model = policy.select_model(toy_prompt)
        assert isinstance(model, str)
        assert model in MODEL_POOL


def test_split_dataset_prompt_disjoint() -> None:
    """Verify train, dev, and test splits have strictly disjoint prompt sets."""
    cases = [
        EvalCase(
            prompt="Prompt A",
            per_model_outcome={"SLM-BERTau-Local": 0.9},
            price_table={"SLM-BERTau-Local": 0.02},
            reference_tier=ComplexityTier.ROUTINE,
        ),
        EvalCase(
            prompt="Prompt B",
            per_model_outcome={"SLM-BERTau-Local": 0.9},
            price_table={"SLM-BERTau-Local": 0.02},
            reference_tier=ComplexityTier.ROUTINE,
        ),
        EvalCase(
            prompt="Prompt C",
            per_model_outcome={"SLM-BERTau-Local": 0.9},
            price_table={"SLM-BERTau-Local": 0.02},
            reference_tier=ComplexityTier.ROUTINE,
        ),
        EvalCase(
            prompt="Prompt D",
            per_model_outcome={"SLM-BERTau-Local": 0.9},
            price_table={"SLM-BERTau-Local": 0.02},
            reference_tier=ComplexityTier.ROUTINE,
        ),
        EvalCase(
            prompt="Prompt A",  # Duplicate prompt
            per_model_outcome={"SLM-BERTau-Local": 0.9},
            price_table={"SLM-BERTau-Local": 0.02},
            reference_tier=ComplexityTier.ROUTINE,
        ),
    ]

    train, dev, test = split_dataset_prompt_disjoint(
        cases, train_ratio=0.5, dev_ratio=0.25, test_ratio=0.25, seed=42
    )

    train_prompts = set(c.prompt for c in train)
    dev_prompts = set(c.prompt for c in dev)
    test_prompts = set(c.prompt for c in test)

    # Zero prompt overlap assertion
    assert train_prompts.intersection(test_prompts) == set()
    assert dev_prompts.intersection(test_prompts) == set()
    assert train_prompts.intersection(dev_prompts) == set()
    assert len(test) > 0


def test_benchmark_harness_integration() -> None:
    """End-to-end integration test of benchmark harness with all 6 policies."""
    dataset = load_synthetic_eval_dataset(n_samples=40, seed=42)
    harness = BenchmarkHarness(seed=42)
    summary = harness.evaluate(dataset, is_synthetic=True, n_bootstraps=50)

    assert len(summary.results) == 6
    assert summary.n_test > 0
    assert summary.n_train > 0
    assert summary.pareto_auc >= 0.0
    assert summary.go_no_go_decision in [
        "GO",
        "NO-GO",
        "CONDITIONAL GO (Pareto Efficient)",
    ]
    for res in summary.results:
        assert res.n_evaluated == summary.n_test
        assert res.n_failures == 0
        assert res.mean_cost > 0.0
        assert res.mean_accuracy > 0.0
