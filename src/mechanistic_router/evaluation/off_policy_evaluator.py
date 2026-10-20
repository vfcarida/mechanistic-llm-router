"""Inverse Propensity Scoring (IPS) & Doubly Robust Counterfactual Policy Evaluator.

Evaluates candidate conformal routing thresholds (lambda_low', lambda_high') offline
using logged bandit feedback collected under a historical logging policy pi_0(a | x):
1. Direct Method (DM) Reward/Loss Regression:
   Fits regularized per-tier expected loss models l_hat(p, a) over complexity scores p.
2. Clipped Inverse Propensity Scoring (IPS) & Self-Normalized IPS (SNIPS)
   (Swaminathan & Joachims, NeurIPS 2015):
   Reweights observed losses by clipped importance ratios w_i = pi_cand(a_i | x_i) / pi_0.
3. Doubly Robust (DR) Estimator (Dudik, Langford & Li, Statistical Science 2014):
   Combines the Direct Method baseline with propensity-weighted residuals:
       psi_i = sum_a pi_cand(a | p_i) l_hat(p_i, a) + w_i * (l_i - l_hat(p_i, a_i)),
   remaining unbiased if either the propensity model pi_0 or the direct model l_hat is accurate.
4. Kish's Effective Sample Size (ESS) & Safe Promotion Gate:
   Computes ESS = (sum_i w_i)^2 / sum_i w_i^2 and certifies candidate promotion only
   when the upper confidence bound of DR risk satisfies UCB_{DR} <= alpha_SLA and
   ESS / N >= min_ess_ratio.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

_STD_NORMAL = NormalDist(0.0, 1.0)


@dataclass(frozen=True)
class LoggedRoutingSample:
    """Single historical routing interaction with bandit feedback."""

    prob_complex: float
    logged_tier_index: int
    logging_propensity: float
    observed_loss: float
    observed_cost_usd: float = 0.02


@dataclass(frozen=True)
class OffPolicyEvaluationResult:
    """Counterfactual policy evaluation metrics and safe promotion decision."""

    num_samples: int
    candidate_low_threshold: float
    candidate_high_threshold: float
    direct_method_risk: float
    ips_risk: float
    snips_risk: float
    doubly_robust_risk: float
    dr_risk_ci_lower: float
    dr_risk_ci_upper: float
    expected_cost_usd: float
    logging_policy_mean_loss: float
    logging_policy_mean_cost_usd: float
    effective_sample_size: float
    ess_ratio: float
    target_sla_alpha: float
    safe_to_promote: bool

    def to_dict(self) -> dict[str, Any]:
        """Serializes off-policy evaluation result to a JSON-compatible dictionary."""
        return asdict(self)


class DoublyRobustPolicyEvaluator:
    """Counterfactual offline evaluator for routing threshold policies."""

    def __init__(
        self,
        tier_costs_usd: tuple[float, float, float] = (0.02, 0.25, 1.50),
        max_importance_weight: float = 20.0,
        min_ess_ratio: float = 0.20,
        confidence_level: float = 0.95,
    ) -> None:
        if len(tier_costs_usd) != 3 or any(c < 0.0 for c in tier_costs_usd):
            raise ValueError("tier_costs_usd must be a 3-tuple of non-negative USD costs.")
        if max_importance_weight <= 0.0:
            raise ValueError(f"max_importance_weight must be > 0, got {max_importance_weight}.")
        if not (0.0 < min_ess_ratio <= 1.0):
            raise ValueError(f"min_ess_ratio must be in (0, 1], got {min_ess_ratio}.")
        if not (0.50 <= confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in [0.50, 1.0), got {confidence_level}.")

        self.tier_costs_usd = tuple(float(c) for c in tier_costs_usd)
        self.max_importance_weight = float(max_importance_weight)
        self.min_ess_ratio = float(min_ess_ratio)
        self.confidence_level = float(confidence_level)

    @staticmethod
    def _candidate_tier_probabilities(
        prob_complex: float,
        low_threshold: float,
        high_threshold: float,
        temperature: float = 0.08,
    ) -> np.ndarray:
        """Computes smooth or deterministic tier selection probabilities over {0, 1, 2}."""
        p = float(np.clip(prob_complex, 0.0, 1.0))
        if temperature <= 1e-6:
            out = np.zeros(3, dtype=np.float64)
            if p < low_threshold:
                out[0] = 1.0
            elif p < high_threshold:
                out[1] = 1.0
            else:
                out[2] = 1.0
            return out

        # Smooth logistic cumulative transitions at low_threshold and high_threshold
        temp = float(max(temperature, 1e-4))
        z_low = np.clip((p - low_threshold) / temp, -35.0, 35.0)
        z_high = np.clip((p - high_threshold) / temp, -35.0, 35.0)
        prob_ge_mid = float(1.0 / (1.0 + math.exp(-float(z_low))))
        prob_ge_high = float(1.0 / (1.0 + math.exp(-float(z_high))))

        p_tier0 = max(0.0, 1.0 - prob_ge_mid)
        p_tier2 = max(0.0, min(prob_ge_mid, prob_ge_high))
        p_tier1 = max(0.0, prob_ge_mid - p_tier2)
        vec = np.array([p_tier0, p_tier1, p_tier2], dtype=np.float64)
        return vec / max(float(np.sum(vec)), 1e-12)

    @staticmethod
    def _fit_direct_loss_model(
        samples: list[LoggedRoutingSample],
    ) -> dict[int, tuple[float, float]]:
        """Fits regularized linear loss model l_hat(p, a) = intercept_a + slope_a * p per tier."""
        # Structural priors: cheap tier loss grows with p; mid tier moderate; oracle near zero
        priors: dict[int, tuple[float, float]] = {
            0: (0.02, 0.70),
            1: (0.01, 0.25),
            2: (0.01, 0.02),
        }
        fitted: dict[int, tuple[float, float]] = {}

        for tier_idx in (0, 1, 2):
            tier_samples = [s for s in samples if int(s.logged_tier_index) == tier_idx]
            prior_b, prior_w = priors[tier_idx]
            if len(tier_samples) < 2:
                fitted[tier_idx] = (prior_b, prior_w)
                continue

            p_arr = np.array([float(s.prob_complex) for s in tier_samples], dtype=np.float64)
            l_arr = np.array([float(s.observed_loss) for s in tier_samples], dtype=np.float64)
            # Ridge regression shrunk toward structural prior with penalty lambda = 2.0
            X_design = np.column_stack([np.ones_like(p_arr), p_arr])
            reg_eye = 2.0 * np.eye(2, dtype=np.float64)
            prior_vec = np.array([prior_b, prior_w], dtype=np.float64)
            theta = np.linalg.solve(
                X_design.T @ X_design + reg_eye,
                X_design.T @ l_arr + reg_eye @ prior_vec,
            )
            fitted[tier_idx] = (float(theta[0]), float(theta[1]))

        return fitted

    @staticmethod
    def _predict_loss(
        model_params: dict[int, tuple[float, float]],
        prob_complex: float,
        tier_idx: int,
    ) -> float:
        """Predicts bounded loss in [0, 1] for (prob_complex, tier_idx)."""
        b_val, w_val = model_params.get(int(tier_idx), (0.02, 0.30))
        return float(np.clip(b_val + w_val * float(prob_complex), 0.0, 1.0))

    def evaluate_thresholds(
        self,
        samples: list[LoggedRoutingSample],
        candidate_low_threshold: float,
        candidate_high_threshold: float,
        target_sla_alpha: float = 0.10,
        temperature: float = 0.08,
    ) -> OffPolicyEvaluationResult:
        """Evaluates a candidate (low, high) threshold pair using DM, IPS, SNIPS, and DR."""
        if not samples:
            raise ValueError("samples list must be non-empty.")
        if not (0.0 < candidate_low_threshold <= candidate_high_threshold < 1.0):
            raise ValueError(
                "Thresholds must satisfy "
                "0.0 < candidate_low_threshold <= candidate_high_threshold < 1.0."
            )
        if not (0.0 < target_sla_alpha < 1.0):
            raise ValueError(f"target_sla_alpha must be in (0, 1), got {target_sla_alpha}.")

        n = len(samples)
        loss_model = self._fit_direct_loss_model(samples)
        cost_vec = np.asarray(self.tier_costs_usd, dtype=np.float64)

        dm_values = np.zeros(n, dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)
        obs_losses = np.zeros(n, dtype=np.float64)
        obs_costs = np.zeros(n, dtype=np.float64)
        dr_values = np.zeros(n, dtype=np.float64)
        expected_costs = np.zeros(n, dtype=np.float64)

        for i, s in enumerate(samples):
            a_i = int(np.clip(s.logged_tier_index, 0, 2))
            pi_0 = float(max(np.clip(s.logging_propensity, 1e-4, 1.0), 1e-4))
            l_i = float(np.clip(s.observed_loss, 0.0, 1.0))
            c_i = float(max(0.0, s.observed_cost_usd))

            pi_cand = self._candidate_tier_probabilities(
                prob_complex=s.prob_complex,
                low_threshold=candidate_low_threshold,
                high_threshold=candidate_high_threshold,
                temperature=temperature,
            )

            # Direct method expected loss under pi_cand
            tier_preds = np.array(
                [self._predict_loss(loss_model, s.prob_complex, a) for a in (0, 1, 2)],
                dtype=np.float64,
            )
            dm_i = float(np.dot(pi_cand, tier_preds))
            w_i = float(min(self.max_importance_weight, float(pi_cand[a_i]) / pi_0))
            l_hat_ai = float(tier_preds[a_i])

            dm_values[i] = dm_i
            weights[i] = w_i
            obs_losses[i] = l_i
            obs_costs[i] = c_i
            dr_values[i] = dm_i + w_i * (l_i - l_hat_ai)
            expected_costs[i] = float(np.dot(pi_cand, cost_vec))

        dm_risk = float(np.clip(np.mean(dm_values), 0.0, 1.0))
        ips_risk = float(np.clip(np.mean(weights * obs_losses), 0.0, 1.0))
        weight_sum = float(np.sum(weights))
        snips_risk = (
            float(np.clip(np.sum(weights * obs_losses) / weight_sum, 0.0, 1.0))
            if weight_sum > 1e-12
            else dm_risk
        )
        dr_risk = float(np.clip(np.mean(dr_values), 0.0, 1.0))

        # Standard error and confidence interval for Doubly Robust estimator
        if n > 1:
            se_dr = float(np.std(dr_values, ddof=1) / math.sqrt(float(n)))
        else:
            se_dr = 0.0
        z_crit = float(_STD_NORMAL.inv_cdf(0.5 + 0.5 * self.confidence_level))
        ci_low = float(np.clip(dr_risk - z_crit * se_dr, 0.0, 1.0))
        ci_high = float(np.clip(dr_risk + z_crit * se_dr, 0.0, 1.0))

        # Kish's Effective Sample Size (ESS)
        sum_w_sq = float(np.sum(weights**2))
        ess = float((weight_sum**2) / max(sum_w_sq, 1e-12)) if weight_sum > 1e-12 else 0.0
        ess = float(min(float(n), ess))
        ess_ratio = float(ess / float(n))

        safe_promote = bool(ci_high <= target_sla_alpha and ess_ratio >= self.min_ess_ratio)

        return OffPolicyEvaluationResult(
            num_samples=n,
            candidate_low_threshold=round(float(candidate_low_threshold), 6),
            candidate_high_threshold=round(float(candidate_high_threshold), 6),
            direct_method_risk=round(dm_risk, 6),
            ips_risk=round(ips_risk, 6),
            snips_risk=round(snips_risk, 6),
            doubly_robust_risk=round(dr_risk, 6),
            dr_risk_ci_lower=round(ci_low, 6),
            dr_risk_ci_upper=round(ci_high, 6),
            expected_cost_usd=round(float(np.mean(expected_costs)), 6),
            logging_policy_mean_loss=round(float(np.mean(obs_losses)), 6),
            logging_policy_mean_cost_usd=round(float(np.mean(obs_costs)), 6),
            effective_sample_size=round(ess, 4),
            ess_ratio=round(ess_ratio, 6),
            target_sla_alpha=round(float(target_sla_alpha), 6),
            safe_to_promote=safe_promote,
        )


__all__ = [
    "DoublyRobustPolicyEvaluator",
    "LoggedRoutingSample",
    "OffPolicyEvaluationResult",
]
