"""Benchmark Evaluation Harness for Zero-Leakage Pareto Front Comparisons."""

from dataclasses import dataclass, field

from ..models.pool import MODEL_POOL
from ..schemas.eval import EvalCase
from .baselines import (
    AlwaysCheapPolicy,
    AlwaysStrongPolicy,
    BasePolicy,
    LearnedLogisticPolicy,
    LengthThresholdPolicy,
    MechanisticPolicy,
    RandomPolicy,
)
from .dataset import split_dataset_prompt_disjoint
from .metrics import (
    compute_matched_cost_accuracy,
    compute_paired_bootstrap_ci,
    compute_pareto_auc,
    compute_pareto_front,
)


@dataclass
class PolicyEvaluationResult:
    """Evaluation summary metrics and confidence intervals for a single routing policy."""

    policy_name: str
    mean_cost: float
    cost_ci: tuple[float, float]
    mean_accuracy: float
    accuracy_ci: tuple[float, float]
    delta_accuracy: float
    delta_accuracy_ci: tuple[float, float]
    delta_cost: float
    delta_cost_ci: tuple[float, float]
    is_on_pareto_front: bool
    n_evaluated: int
    n_failures: int = 0
    raw_costs: list[float] = field(default_factory=list)
    raw_accs: list[float] = field(default_factory=list)


@dataclass
class BenchmarkExecutionSummary:
    """Complete summary of benchmark execution, Pareto points, and Go/No-Go decision."""

    results: list[PolicyEvaluationResult]
    pareto_frontier: list[tuple[float, float]]
    pareto_auc: float
    matched_cost_quality: dict[str, float]
    n_train: int
    n_dev: int
    n_test: int
    is_synthetic: bool
    go_no_go_decision: str
    go_no_go_rationale: str


class BenchmarkHarness:
    """Coordinates zero-leakage evaluation, bootstrap inference, and Pareto analysis."""

    def __init__(
        self,
        policies: list[BasePolicy] | None = None,
        seed: int = 42,
    ):
        self.seed = seed
        self.policies = policies or [
            AlwaysCheapPolicy(),
            AlwaysStrongPolicy(),
            RandomPolicy(seed=seed),
            LengthThresholdPolicy(),
            LearnedLogisticPolicy(seed=seed),
            MechanisticPolicy(),
        ]

    def evaluate(
        self,
        cases: list[EvalCase],
        is_synthetic: bool = True,
        train_ratio: float = 0.6,
        dev_ratio: float = 0.2,
        test_ratio: float = 0.2,
        n_bootstraps: int = 1000,
    ) -> BenchmarkExecutionSummary:
        """Executes full benchmark evaluation across train/dev/test splits."""
        train_cases, dev_cases, test_cases = split_dataset_prompt_disjoint(
            cases,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            seed=self.seed,
        )

        # 1. Fit policies on train set
        for policy in self.policies:
            policy.fit(train_cases)

        # 2. Tune policy thresholds on dev set
        for policy in self.policies:
            policy.tune(dev_cases)

        # 3. Evaluate strictly on test set
        policy_outcomes: dict[str, tuple[list[float], list[float]]] = {}
        for policy in self.policies:
            costs: list[float] = []
            accs: list[float] = []
            for case in test_cases:
                # ROUTING BOUNDARY: Router only receives case.prompt!
                selected = policy.select_model(case.prompt)
                cost = case.price_table.get(selected, MODEL_POOL[selected].cost)
                acc = case.per_model_outcome.get(selected, 0.0)
                costs.append(cost)
                accs.append(acc)
            policy_outcomes[policy.name] = (costs, accs)

        # Identify baseline outcomes (AlwaysCheap)
        cheap_policy_name = AlwaysCheapPolicy.name
        if cheap_policy_name in policy_outcomes:
            base_costs, base_accs = policy_outcomes[cheap_policy_name]
        else:
            base_costs, base_accs = list(policy_outcomes.values())[0]

        # 4. Compute paired bootstrap statistics per policy
        policy_results: list[PolicyEvaluationResult] = []
        point_candidates: list[tuple[float, float]] = []

        for policy in self.policies:
            costs, accs = policy_outcomes[policy.name]
            stats = compute_paired_bootstrap_ci(
                candidate_costs=costs,
                candidate_accs=accs,
                baseline_costs=base_costs,
                baseline_accs=base_accs,
                n_bootstraps=n_bootstraps,
                seed=self.seed,
            )
            point_candidates.append((stats["candidate_cost_mean"], stats["candidate_acc_mean"]))

            res = PolicyEvaluationResult(
                policy_name=policy.name,
                mean_cost=stats["candidate_cost_mean"],
                cost_ci=stats["candidate_cost_ci"],
                mean_accuracy=stats["candidate_acc_mean"],
                accuracy_ci=stats["candidate_acc_ci"],
                delta_accuracy=stats["delta_acc_mean"],
                delta_accuracy_ci=stats["delta_acc_ci"],
                delta_cost=stats["delta_cost_mean"],
                delta_cost_ci=stats["delta_cost_ci"],
                is_on_pareto_front=False,  # determined next
                n_evaluated=len(test_cases),
                raw_costs=costs,
                raw_accs=accs,
            )
            policy_results.append(res)

        # 5. Determine Pareto Front & Pareto AUC
        pareto_pts = compute_pareto_front(point_candidates)
        pareto_pt_set = set(pareto_pts)

        for res in policy_results:
            for p_c, p_a in pareto_pt_set:
                if abs(res.mean_cost - p_c) < 1e-3 and abs(res.mean_accuracy - p_a) < 1e-3:
                    res.is_on_pareto_front = True
                    break

        auc = compute_pareto_auc(pareto_pts)

        # 6. Matched trade-offs
        cheap_cost = min(r.mean_cost for r in policy_results)
        strong_cost = max(r.mean_cost for r in policy_results)
        mid_cost = (cheap_cost + strong_cost) / 2.0
        matched_quality = {
            "at_cheap_cost": compute_matched_cost_accuracy(pareto_pts, cheap_cost),
            "at_mid_cost": compute_matched_cost_accuracy(pareto_pts, mid_cost),
            "at_strong_cost": compute_matched_cost_accuracy(pareto_pts, strong_cost),
        }

        # 7. Formulate Objective Go/No-Go Decision
        # Question: Does any learned or mechanistic policy Pareto-dominate always-cheap
        # at matched quality with non-overlapping 95% CIs?
        excluded_baselines = {
            AlwaysCheapPolicy.name,
            AlwaysStrongPolicy.name,
            RandomPolicy.name,
        }
        adv_candidates = [r for r in policy_results if r.policy_name not in excluded_baselines]

        strictly_dominates_cheap = any(
            r.is_on_pareto_front
            and r.delta_accuracy_ci[0] > 0.0
            and r.delta_cost_ci[1] <= 0.0
            for r in adv_candidates
        )

        beats_cheap_accuracy_with_significance = any(
            r.delta_accuracy_ci[0] > 0.0 for r in adv_candidates
        )

        if strictly_dominates_cheap:
            go_no_go = "GO"
            rationale = (
                "A learned/mechanistic policy strictly Pareto-dominates AlwaysCheap: "
                "it achieves statistically significant quality gain (lower 95% CI > 0) "
                "with zero cost penalty."
            )
        elif beats_cheap_accuracy_with_significance:
            pareto_active = [r.policy_name for r in adv_candidates if r.is_on_pareto_front]
            go_no_go = "CONDITIONAL GO (Pareto Efficient)"
            rationale = (
                f"Candidate policies ({', '.join(pareto_active)}) expand the Pareto frontier "
                "with statistically significant quality gains over AlwaysCheap (95% CI > 0), "
                "but incur higher cost. Proceed to T06 with strict latency and memory constraints."
            )
        else:
            go_no_go = "NO-GO"
            rationale = (
                "No learned or mechanistic policy achieves statistically significant "
                "quality improvements over AlwaysCheap (95% CI crosses zero). "
                "Adopting complex routing is not justified."
            )

        return BenchmarkExecutionSummary(
            results=policy_results,
            pareto_frontier=pareto_pts,
            pareto_auc=auc,
            matched_cost_quality=matched_quality,
            n_train=len(train_cases),
            n_dev=len(dev_cases),
            n_test=len(test_cases),
            is_synthetic=is_synthetic,
            go_no_go_decision=go_no_go,
            go_no_go_rationale=rationale,
        )


__all__ = [
    "PolicyEvaluationResult",
    "BenchmarkExecutionSummary",
    "BenchmarkHarness",
]
