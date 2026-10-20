"""Contextual Bandit Routing in Activation Space via Disjoint LinUCB (Sherman-Morrison).

Implements `LinUCBActivationRouter` (Li et al., WWW 2010) over normalized prefill
residual-stream activations. Maintains per-model ridge precision inverses `A_m^-1`
updated in O(D^2) time via the rank-1 Sherman-Morrison identity without matrix
inversion, balancing expected quality, epistemic uncertainty exploration, and token cost.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_CONFIG, RouterConfig
from ..evaluation.calibration_pack import DeterministicMultiLayerExtractor
from ..models.types import TargetModel
from ..schemas.routing import ProbingSignals, RoutingDecision, RoutingRequest
from .base import AbstractRouter


class LinUCBActivationRouter(AbstractRouter):
    """Contextual bandit router operating on residual-stream hidden activations.

    For each candidate model m in the pool, maintains:
        - Precision inverse A_m^-1 in R^{D x D}, initialized to (1 / ridge_lambda) * I_D
        - Reward-weighted context accumulator b_m in R^D, initialized to 0

    Given an L2-normalized activation context x in R^D, computes:
        - Ridge parameter estimate: theta_m = A_m^-1 @ b_m
        - Predicted quality mean:   mu_m(x) = clip(base_accuracy_m + theta_m^T x, 0, 1)
        - Epistemic uncertainty:    sigma_m(x) = sqrt(max(0, x^T A_m^-1 x))
        - Cost-penalized UCB score: UCB_m(x) = mu_m(x) + alpha_ucb * sigma_m(x) - cost_penalty_m
    """

    def __init__(
        self,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
        extractor: Any | None = None,
        feature_dim: int = 32,
        alpha_ucb: float = 0.35,
        ridge_lambda: float = 1.0,
        cost_weight: float | None = None,
    ) -> None:
        super().__init__(model_pool, config)
        if feature_dim < 2:
            raise ValueError(f"feature_dim must be >= 2, got {feature_dim}.")
        if alpha_ucb < 0.0:
            raise ValueError(f"alpha_ucb must be >= 0.0, got {alpha_ucb}.")
        if ridge_lambda <= 0.0:
            raise ValueError(f"ridge_lambda must be > 0.0, got {ridge_lambda}.")

        self.feature_dim = int(feature_dim)
        self.alpha_ucb = float(alpha_ucb)
        self.ridge_lambda = float(ridge_lambda)
        self.cost_weight = (
            float(cost_weight) if cost_weight is not None else float(config.lambda_budget)
        )
        self.extractor = (
            extractor
            if extractor is not None
            else DeterministicMultiLayerExtractor(
                d_layer=self.feature_dim,
                layer_indices=[-1],
                layer_fusion="concat",
            )
        )

        self._max_cost = max((m.cost for m in model_pool.values()), default=1.0) or 1.0
        self._a_inv: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._pull_counts: dict[str, int] = {}
        self._init_arms(self.feature_dim)

    def _init_arms(self, dim: int) -> None:
        """Initializes per-arm inverse covariance matrices and reward accumulators."""
        self.feature_dim = int(dim)
        init_cov = np.eye(self.feature_dim, dtype=np.float64) / self.ridge_lambda
        for m_name in self.model_pool:
            self._a_inv[m_name] = init_cov.copy()
            self._b[m_name] = np.zeros(self.feature_dim, dtype=np.float64)
            self._pull_counts[m_name] = 0

    def _extract_context(self, prompt_or_activation: str | np.ndarray) -> np.ndarray:
        """Extracts and L2-normalizes a 1D context vector x in R^D."""
        if isinstance(prompt_or_activation, str):
            raw = np.asarray(self.extractor.extract_one(prompt_or_activation), dtype=np.float64)
        else:
            raw = np.asarray(prompt_or_activation, dtype=np.float64).reshape(-1)

        if raw.shape[0] != self.feature_dim:
            if sum(self._pull_counts.values()) == 0:
                self._init_arms(int(raw.shape[0]))
            elif raw.shape[0] > self.feature_dim:
                raw = raw[: self.feature_dim]
            else:
                padded = np.zeros(self.feature_dim, dtype=np.float64)
                padded[: raw.shape[0]] = raw
                raw = padded

        norm = float(np.linalg.norm(raw))
        if norm > 1e-12:
            return raw / norm
        return raw

    def compute_arm_statistics(self, x: np.ndarray) -> dict[str, dict[str, float]]:
        """Computes (expected_quality, uncertainty, cost_penalty, ucb_score) per model arm."""
        stats: dict[str, dict[str, float]] = {}
        for m_name, model in self.model_pool.items():
            a_inv = self._a_inv[m_name]
            b_vec = self._b[m_name]
            theta = a_inv @ b_vec
            residual_pred = float(np.dot(theta, x))
            expected_quality = float(np.clip(model.base_accuracy + residual_pred, 0.0, 1.0))

            quad_form = float(np.dot(x, a_inv @ x))
            uncertainty = float(np.sqrt(max(0.0, quad_form)))

            normalized_cost = model.cost / self._max_cost
            cost_penalty = self.cost_weight * normalized_cost
            ucb_score = expected_quality + self.alpha_ucb * uncertainty - cost_penalty

            stats[m_name] = {
                "expected_quality": expected_quality,
                "uncertainty": uncertainty,
                "cost_penalty": cost_penalty,
                "ucb_score": ucb_score,
                "theta_norm": float(np.linalg.norm(theta)),
                "pull_count": float(self._pull_counts[m_name]),
            }
        return stats

    def update_feedback(
        self,
        model_name: str,
        prompt_or_activation: str | np.ndarray,
        reward: float,
    ) -> dict[str, float | int]:
        """Updates arm parameters in O(D^2) via the rank-1 Sherman-Morrison formula.

        Args:
            model_name: Name of the routed model arm in `model_pool`.
            prompt_or_activation: Prompt string or raw activation vector.
            reward: Observed quality reward in [0.0, 1.0].

        Returns:
            Dictionary with uncertainty before/after update, theta norm, and pull count.
        """
        if model_name not in self.model_pool:
            raise ValueError(f"Unknown model_name '{model_name}' not in model_pool.")
        clipped_reward = float(np.clip(reward, 0.0, 1.0))

        x = self._extract_context(prompt_or_activation)
        a_inv = self._a_inv[model_name]

        ax = a_inv @ x
        quad_before = float(np.dot(x, ax))
        uncertainty_before = float(np.sqrt(max(0.0, quad_before)))

        denom = 1.0 + quad_before
        # Rank-1 Sherman-Morrison update: A_new^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        self._a_inv[model_name] = a_inv - np.outer(ax, ax) / denom

        # Target residual relative to model prior base_accuracy
        residual_reward = clipped_reward - self.model_pool[model_name].base_accuracy
        self._b[model_name] = self._b[model_name] + residual_reward * x
        self._pull_counts[model_name] += 1

        quad_after = float(np.dot(x, self._a_inv[model_name] @ x))
        uncertainty_after = float(np.sqrt(max(0.0, quad_after)))
        theta_norm = float(np.linalg.norm(self._a_inv[model_name] @ self._b[model_name]))

        return {
            "uncertainty_before": uncertainty_before,
            "uncertainty_after": uncertainty_after,
            "theta_norm": theta_norm,
            "pull_count": self._pull_counts[model_name],
        }

    async def route(self, request: RoutingRequest | str) -> RoutingDecision:
        """Selects the candidate model arm maximizing the cost-penalized LinUCB score."""
        start_time = time.perf_counter()

        if isinstance(request, str):
            request = RoutingRequest(prompt=request)
        elif not isinstance(request, RoutingRequest):
            raise TypeError("request must be an instance of RoutingRequest or str")

        x = self._extract_context(request.prompt)
        arm_stats = self.compute_arm_statistics(x)

        # Filter by optional max_budget_usd if specified
        eligible_models = [
            m_name
            for m_name, m_obj in self.model_pool.items()
            if request.max_budget_usd is None or m_obj.cost <= request.max_budget_usd
        ]
        if not eligible_models:
            eligible_models = list(self.model_pool.keys())

        selected_model = max(eligible_models, key=lambda m: arm_stats[m]["ucb_score"])
        selected_target = self.model_pool[selected_model]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        signals = {
            m_name: ProbingSignals(
                d_eff_mean=float(np.linalg.norm(x)),
                fisher_j=arm_stats[m_name]["ucb_score"],
                fisher_j_norm=arm_stats[m_name]["expected_quality"],
                is_competent=(m_name in eligible_models),
                sae_features=[],
                extra_metadata={
                    "expected_quality": arm_stats[m_name]["expected_quality"],
                    "ucb_uncertainty": arm_stats[m_name]["uncertainty"],
                    "cost_penalty": arm_stats[m_name]["cost_penalty"],
                    "ucb_score": arm_stats[m_name]["ucb_score"],
                    "theta_norm": arm_stats[m_name]["theta_norm"],
                    "pull_count": int(arm_stats[m_name]["pull_count"]),
                },
            )
            for m_name in self.model_pool
        }

        return RoutingDecision(
            selected_model=selected_model,
            strategy_used="LinUCBActivationRouter",
            estimated_cost_usd=selected_target.cost,
            latency_ms=latency_ms,
            signals=signals,
        )

    def save(self, filepath: str | Path) -> None:
        """Persists bandit precision inverses and reward vectors to a .npz archive."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        arm_names = sorted(self.model_pool.keys())
        a_inv_stack = np.stack([self._a_inv[m] for m in arm_names], axis=0)
        b_stack = np.stack([self._b[m] for m in arm_names], axis=0)
        pulls = np.array([self._pull_counts[m] for m in arm_names], dtype=np.int64)
        meta = np.array(
            [float(self.feature_dim), self.alpha_ucb, self.ridge_lambda, self.cost_weight],
            dtype=np.float64,
        )
        np.savez_compressed(
            str(path),
            a_inv=a_inv_stack,
            b=b_stack,
            pulls=pulls,
            meta=meta,
        )

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
        extractor: Any | None = None,
    ) -> LinUCBActivationRouter:
        """Loads a persisted LinUCBActivationRouter from a .npz archive."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"LinUCB checkpoint not found at: {path}")

        with np.load(str(path)) as data:
            meta = data["meta"]
            feature_dim = int(meta[0])
            alpha_ucb = float(meta[1])
            ridge_lambda = float(meta[2])
            cost_weight = float(meta[3])
            a_inv_stack = data["a_inv"]
            b_stack = data["b"]
            pulls = data["pulls"]

        router = cls(
            model_pool=model_pool,
            config=config,
            extractor=extractor,
            feature_dim=feature_dim,
            alpha_ucb=alpha_ucb,
            ridge_lambda=ridge_lambda,
            cost_weight=cost_weight,
        )
        arm_names = sorted(model_pool.keys())
        for idx, m_name in enumerate(arm_names):
            router._a_inv[m_name] = np.asarray(a_inv_stack[idx], dtype=np.float64)
            router._b[m_name] = np.asarray(b_stack[idx], dtype=np.float64)
            router._pull_counts[m_name] = int(pulls[idx])
        return router


__all__ = ["LinUCBActivationRouter"]
