"""Primal-Dual & Multiple-Choice Knapsack Batch Portfolio Router.

Implements global FinOps budget-constrained routing across batches and online streams:
1. Batch Portfolio Knapsack Optimization (Agrawal & Devanur, EC 2014):
   Given N prompts with residual-stream complexity probabilities p_1, ..., p_N, a
   global USD budget B_total, and optional per-tier capacity caps, jointly assigns
   each prompt to a model tier (cheap, mid, strong) to maximize expected portfolio
   quality while strictly respecting sum_i cost(m_i) <= B_total (when >= N * c_min).
2. Online Dual Mirror Descent Controller (Balseiro et al., Operations Research):
   Maintains a dynamic non-negative Lagrange multiplier (shadow price of budget)
   mu_t >= 0 updated after each routing decision via projected subgradient ascent
   to enforce a long-run average per-request budget target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_CONFIG, RouterConfig
from ..models.pool import get_model_accuracy
from ..models.types import TargetModel, TaskComplexity
from .causal_probe import CausalProbeRouter


@dataclass(frozen=True)
class BatchAllocationItem:
    """Single prompt allocation decision within a portfolio-optimized batch."""

    index: int
    prompt: str
    prob_complex: float
    selected_model: str
    estimated_cost_usd: float
    expected_quality: float


@dataclass(frozen=True)
class BatchPortfolioResult:
    """Summary of joint portfolio allocation across a batch of prompts."""

    allocations: list[BatchAllocationItem]
    total_cost_usd: float
    total_budget_usd: float
    oracle_cost_usd: float
    cost_saved_usd: float
    mean_expected_quality: float
    tier_distribution: dict[str, int]
    budget_feasible: bool


def compute_expected_tier_quality(
    prob_complex: float,
    model: TargetModel | float,
    max_accuracy: float = 0.97,
) -> float:
    """Computes expected quality of assigning a prompt with complexity p_i to a model tier.

    When `model` is a `TargetModel`, evaluates the degree-2 Bernstein polynomial
    expectation over (ROUTINE, MODERATE, COMPLEX) competence ceilings:
        w_routine = (1 - p)^2,  w_moderate = 2 p (1 - p),  w_complex = p^2.
    """
    p = float(np.clip(prob_complex, 0.0, 1.0))
    if isinstance(model, TargetModel):
        w_routine = (1.0 - p) ** 2
        w_moderate = 2.0 * p * (1.0 - p)
        w_complex = p**2
        q_val = (
            w_routine * get_model_accuracy(model, TaskComplexity.ROUTINE)
            + w_moderate * get_model_accuracy(model, TaskComplexity.MODERATE)
            + w_complex * get_model_accuracy(model, TaskComplexity.COMPLEX)
        )
        return float(np.clip(q_val, 0.0, 1.0))

    gap = max(0.0, float(max_accuracy) - float(model))
    quality = float(max_accuracy) - (0.35 + 0.65 * p) * gap
    return float(np.clip(quality, 0.0, 1.0))


class PrimalDualBudgetController:
    """Online Dual Mirror Descent controller for average per-query budget governance."""

    def __init__(
        self,
        model_pool: dict[str, TargetModel],
        target_avg_budget_usd: float,
        step_size: float = 10.0,
        initial_dual_mu: float = 1.0,
    ) -> None:
        if not model_pool:
            raise ValueError("model_pool cannot be empty.")
        if target_avg_budget_usd <= 0.0:
            raise ValueError("target_avg_budget_usd must be > 0.")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0.")
        if initial_dual_mu < 0.0:
            raise ValueError("initial_dual_mu must be >= 0.")

        self.ordered_models = sorted(model_pool.values(), key=lambda m: (m.cost, m.base_accuracy))
        self.max_accuracy = max(m.base_accuracy for m in self.ordered_models)
        self.target_avg_budget_usd = float(target_avg_budget_usd)
        self.step_size = float(step_size)
        self._dual_mu = float(initial_dual_mu)
        self._steps = 0
        self._cumulative_spend_usd = 0.0

    @property
    def dual_mu(self) -> float:
        """Current Lagrange shadow price of budget mu_t >= 0."""
        return self._dual_mu

    @property
    def steps(self) -> int:
        """Total online requests processed."""
        return self._steps

    @property
    def average_spend_usd(self) -> float:
        """Empirical average USD spend per request."""
        if self._steps == 0:
            return 0.0
        return self._cumulative_spend_usd / self._steps

    def select_and_update(self, prob_complex: float) -> tuple[str, float, float]:
        """Selects Lagrangian-optimal model tier and updates shadow price mu_{t+1}."""
        best_model = self.ordered_models[0]
        best_lagrangian = -float("inf")

        for m in self.ordered_models:
            exp_q = compute_expected_tier_quality(
                prob_complex=prob_complex,
                model=m,
                max_accuracy=self.max_accuracy,
            )
            lagrangian_val = exp_q - self._dual_mu * m.cost
            if lagrangian_val > best_lagrangian:
                best_lagrangian = lagrangian_val
                best_model = m

        cost = float(best_model.cost)
        self._steps += 1
        self._cumulative_spend_usd += cost

        # Projected dual subgradient ascent: mu_{t+1} = max(0, mu_t + eta * (c_t - b_target))
        self._dual_mu = max(
            0.0,
            self._dual_mu + self.step_size * (cost - self.target_avg_budget_usd),
        )
        return best_model.name, cost, self._dual_mu


class BatchPortfolioRouter:
    """Joint batch portfolio optimizer maximizing expected quality under global budget."""

    def __init__(
        self,
        causal_router: CausalProbeRouter,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
    ) -> None:
        if not model_pool:
            raise ValueError("model_pool cannot be empty.")
        self.causal_router = causal_router
        self.model_pool = model_pool
        self.config = config
        self.ordered_models = sorted(model_pool.values(), key=lambda m: (m.cost, m.base_accuracy))
        self.max_accuracy = max(m.base_accuracy for m in self.ordered_models)

    def optimize_probabilities(
        self,
        prompts: list[str],
        probabilities: list[float] | np.ndarray,
        total_budget_usd: float,
        max_tier_counts: dict[str, int] | None = None,
    ) -> BatchPortfolioResult:
        """Solves the multiple-choice knapsack allocation given prompt complexity probabilities."""
        if not prompts:
            raise ValueError("prompts list cannot be empty.")
        probs = np.asarray(probabilities, dtype=np.float64).ravel()
        if probs.shape[0] != len(prompts):
            raise ValueError("Length of probabilities must match length of prompts.")
        if total_budget_usd < 0.0:
            raise ValueError("total_budget_usd must be >= 0.")

        n_items = len(prompts)
        n_tiers = len(self.ordered_models)
        cheapest_cost = float(self.ordered_models[0].cost)
        oracle_cost = float(self.ordered_models[-1].cost) * n_items

        # Initialize all items at cheapest tier (tier 0)
        assigned_tier_idx = [0] * n_items
        current_cost = cheapest_cost * n_items
        budget_feasible = bool(current_cost <= total_budget_usd + 1e-12)

        tier_counts: dict[str, int] = {m.name: 0 for m in self.ordered_models}
        tier_counts[self.ordered_models[0].name] = n_items

        # Iteratively perform marginal bang-per-buck upgrades while budget allows
        if budget_feasible and n_tiers > 1:
            while True:
                best_ratio = -1.0
                best_item = -1
                best_target_tier = -1
                best_delta_cost = 0.0

                for i in range(n_items):
                    curr_t = assigned_tier_idx[i]
                    if curr_t >= n_tiers - 1:
                        continue
                    curr_model = self.ordered_models[curr_t]
                    curr_q = compute_expected_tier_quality(
                        float(probs[i]), curr_model, self.max_accuracy
                    )

                    for next_t in range(curr_t + 1, n_tiers):
                        next_model = self.ordered_models[next_t]
                        if max_tier_counts is not None and next_model.name in max_tier_counts:
                            if tier_counts[next_model.name] >= max_tier_counts[next_model.name]:
                                continue

                        delta_c = float(next_model.cost - curr_model.cost)
                        if current_cost + delta_c > total_budget_usd + 1e-12:
                            continue

                        next_q = compute_expected_tier_quality(
                            float(probs[i]), next_model, self.max_accuracy
                        )
                        delta_q = max(0.0, next_q - curr_q)
                        if delta_q <= 1e-12:
                            continue

                        ratio = delta_q / max(delta_c, 1e-12) * (1.0 + float(probs[i]))
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_item = i
                            best_target_tier = next_t
                            best_delta_cost = delta_c

                if best_item < 0:
                    break

                prev_t = assigned_tier_idx[best_item]
                prev_name = self.ordered_models[prev_t].name
                target_name = self.ordered_models[best_target_tier].name

                assigned_tier_idx[best_item] = best_target_tier
                current_cost += best_delta_cost
                tier_counts[prev_name] -= 1
                tier_counts[target_name] += 1

        allocations: list[BatchAllocationItem] = []
        qualities: list[float] = []
        for i in range(n_items):
            t_idx = assigned_tier_idx[i]
            m = self.ordered_models[t_idx]
            exp_q = compute_expected_tier_quality(float(probs[i]), m, self.max_accuracy)
            qualities.append(exp_q)
            allocations.append(
                BatchAllocationItem(
                    index=i,
                    prompt=prompts[i],
                    prob_complex=float(probs[i]),
                    selected_model=m.name,
                    estimated_cost_usd=float(m.cost),
                    expected_quality=exp_q,
                )
            )

        total_cost = round(float(current_cost), 6)
        cost_saved = max(0.0, round(oracle_cost - total_cost, 6))
        mean_q = float(np.mean(qualities)) if qualities else 0.0

        return BatchPortfolioResult(
            allocations=allocations,
            total_cost_usd=total_cost,
            total_budget_usd=float(total_budget_usd),
            oracle_cost_usd=round(oracle_cost, 6),
            cost_saved_usd=cost_saved,
            mean_expected_quality=mean_q,
            tier_distribution=tier_counts,
            budget_feasible=budget_feasible,
        )

    def optimize_batch(
        self,
        prompts: list[str],
        total_budget_usd: float,
        max_tier_counts: dict[str, int] | None = None,
    ) -> BatchPortfolioResult:
        """Extracts batch prefill activations, predicts complexity, and optimizes portfolio."""
        if not prompts:
            raise ValueError("prompts list cannot be empty.")
        activations = self.causal_router.extractor.extract_batch(prompts)
        probs = self.causal_router.probe.predict_proba(activations)
        return self.optimize_probabilities(
            prompts=prompts,
            probabilities=probs,
            total_budget_usd=total_budget_usd,
            max_tier_counts=max_tier_counts,
        )


__all__ = [
    "BatchAllocationItem",
    "BatchPortfolioResult",
    "BatchPortfolioRouter",
    "PrimalDualBudgetController",
    "compute_expected_tier_quality",
]
