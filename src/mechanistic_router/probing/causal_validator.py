"""Causal Subspace Ablation and Directional Projection Validation Harness.

Implements directional subspace projection (Zou et al., arXiv:2310.01405) and paired
bootstrap hypothesis testing against random unit-sphere control directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DirectionalProbeProtocol(Protocol):
    """Protocol for linear probes exposing a unit direction vector and binary prediction."""

    @property
    def direction_vector(self) -> np.ndarray: ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...


@dataclass
class CausalValidationResult:
    """Quantitative results from a causal directional ablation experiment."""

    unablated_accuracy: float
    unablated_ci: tuple[float, float]
    target_ablated_accuracy: float
    target_ablated_ci: tuple[float, float]
    control_ablated_accuracy: float
    control_ablated_ci: tuple[float, float]
    target_delta: float
    control_delta: float
    causal_effect_size: float
    causal_effect_ci: tuple[float, float]
    is_causal: bool
    num_controls: int
    sample_size: int

    def summary(self) -> str:
        """Formats a human-readable summary of the causal ablation experiment."""
        verdict = "CAUSAL (Validated)" if self.is_causal else "NON-CAUSAL (Correlational Only)"
        u_ci = f"[{self.unablated_ci[0]:.2%}, {self.unablated_ci[1]:.2%}]"
        t_ci = f"[{self.target_ablated_ci[0]:.2%}, {self.target_ablated_ci[1]:.2%}]"
        c_ci = f"[{self.control_ablated_ci[0]:.2%}, {self.control_ablated_ci[1]:.2%}]"
        e_ci = f"[{self.causal_effect_ci[0]:+.2%}, {self.causal_effect_ci[1]:+.2%}]"
        return (
            f"Verdict: {verdict}\n"
            f"Unablated Accuracy: {self.unablated_accuracy:.2%} {u_ci}\n"
            f"Target Ablated Accuracy: {self.target_ablated_accuracy:.2%} {t_ci}\n"
            f"Control Ablated Accuracy: {self.control_ablated_accuracy:.2%} {c_ci}\n"
            f"Causal Effect Size: {self.causal_effect_size:+.2%} {e_ci}"
        )


class CausalAblationHarness:
    """Validates whether an activation direction is causal via orthogonal subspace ablation.

    A learned probe direction v in R^D is validated as causal if projecting out v from test
    representations X_test degrades classification performance significantly more than
    projecting out K random unit vectors r_k uniformly sampled from the unit sphere S^{D-1}.
    """

    def __init__(
        self,
        num_controls: int = 50,
        n_bootstraps: int = 1000,
        seed: int = 42,
    ) -> None:
        self.num_controls = num_controls
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    @staticmethod
    def ablate_direction(X: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Projects out the unit vector v from activation matrix X.

        Args:
            X: Activation matrix of shape (N, D).
            v: Direction vector of shape (D,).

        Returns:
            Orthogonally projected matrix X_ablated of shape (N, D) such that X_ablated @ v == 0.
        """
        norm_v = float(np.linalg.norm(v))
        if norm_v == 0.0:
            return X.copy()
        v_unit = v / norm_v
        projections = X @ v_unit
        return np.asarray(X - np.outer(projections, v_unit))

    def generate_random_controls(self, dim: int, rng: np.random.Generator) -> np.ndarray:
        """Generates K random unit vectors in R^dim (shape: (K, dim))."""
        raw = rng.standard_normal(size=(self.num_controls, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return np.asarray(raw / norms)

    def validate(
        self,
        probe: DirectionalProbeProtocol,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> CausalValidationResult:
        """Runs directional ablation of the probe's primary direction vs matched random controls."""
        n_samples, dim = X_test.shape
        rng = np.random.default_rng(self.seed)

        # 1. Unablated performance
        pred_unablated = probe.predict(X_test, threshold=threshold)
        scores_unablated = (pred_unablated == y_test).astype(float)

        # 2. Target direction ablation
        v_target = probe.direction_vector
        X_target_ablated = self.ablate_direction(X_test, v_target)
        pred_target_ablated = probe.predict(X_target_ablated, threshold=threshold)
        scores_target = (pred_target_ablated == y_test).astype(float)

        # 3. Control direction ablations (K random unit vectors)
        control_dirs = self.generate_random_controls(dim, rng)
        control_scores_matrix = np.zeros((self.num_controls, n_samples), dtype=float)

        for k in range(self.num_controls):
            X_ctrl = self.ablate_direction(X_test, control_dirs[k])
            pred_ctrl = probe.predict(X_ctrl, threshold=threshold)
            control_scores_matrix[k] = (pred_ctrl == y_test).astype(float)

        scores_control_mean = np.mean(control_scores_matrix, axis=0)

        delta_target_samples = scores_target - scores_unablated
        delta_control_samples = scores_control_mean - scores_unablated
        causal_effect_samples = scores_target - scores_control_mean

        # 4. Paired bootstrap uncertainty estimation
        boot_indices = rng.integers(0, n_samples, size=(self.n_bootstraps, n_samples))

        boot_unablated = np.mean(scores_unablated[boot_indices], axis=1)
        boot_target = np.mean(scores_target[boot_indices], axis=1)
        boot_control = np.mean(scores_control_mean[boot_indices], axis=1)
        boot_causal = np.mean(causal_effect_samples[boot_indices], axis=1)

        ci_unablated = (
            float(np.percentile(boot_unablated, 2.5)),
            float(np.percentile(boot_unablated, 97.5)),
        )
        ci_target = (
            float(np.percentile(boot_target, 2.5)),
            float(np.percentile(boot_target, 97.5)),
        )
        ci_control = (
            float(np.percentile(boot_control, 2.5)),
            float(np.percentile(boot_control, 97.5)),
        )
        ci_causal = (
            float(np.percentile(boot_causal, 2.5)),
            float(np.percentile(boot_causal, 97.5)),
        )

        mean_unablated = float(np.mean(scores_unablated))
        mean_target = float(np.mean(scores_target))
        mean_control = float(np.mean(scores_control_mean))
        mean_causal = float(np.mean(causal_effect_samples))

        target_delta = float(np.mean(delta_target_samples))
        control_delta = float(np.mean(delta_control_samples))

        # Causal criterion: upper bound of 95% CI must be strictly below 0
        is_causal = bool(ci_causal[1] < 0.0)

        return CausalValidationResult(
            unablated_accuracy=mean_unablated,
            unablated_ci=ci_unablated,
            target_ablated_accuracy=mean_target,
            target_ablated_ci=ci_target,
            control_ablated_accuracy=mean_control,
            control_ablated_ci=ci_control,
            target_delta=target_delta,
            control_delta=control_delta,
            causal_effect_size=mean_causal,
            causal_effect_ci=ci_causal,
            is_causal=is_causal,
            num_controls=self.num_controls,
            sample_size=n_samples,
        )


__all__ = ["CausalAblationHarness", "CausalValidationResult", "DirectionalProbeProtocol"]
