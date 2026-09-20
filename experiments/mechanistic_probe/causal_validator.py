"""Causal ablation and directional projection validation harness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probe import LinearActivationProbe


@dataclass
class CausalValidationResult:
    """Quantitative results from causal directional ablation experiment."""

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
    """Validates whether an activation direction is causal via directional ablation."""

    def __init__(self, num_controls: int = 50, n_bootstraps: int = 1000, seed: int = 42) -> None:
        self.num_controls = num_controls
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    @staticmethod
    def ablate_direction(X: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Projects out the unit vector v from activation matrix X.

        X: (N, D)
        v: (D,) unit vector
        Returns X_ablated: (N, D) such that X_ablated @ v == 0
        """
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return X.copy()
        v_unit = v / norm_v
        # Projections: (N,)
        projections = X @ v_unit
        # Subtraction: X - proj[:, None] * v_unit[None, :]
        return X - np.outer(projections, v_unit)

    def generate_random_controls(self, dim: int, rng: np.random.Generator) -> np.ndarray:
        """Generates random unit vectors in R^dim (shape: (K, dim))."""
        raw = rng.standard_normal(size=(self.num_controls, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return raw / norms

    def validate(
        self,
        probe: LinearActivationProbe,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> CausalValidationResult:
        """Runs directional ablation of the probe's primary direction vs matched random controls."""
        N, D = X_test.shape
        rng = np.random.default_rng(self.seed)

        # 1. Unablated performance
        pred_unablated = probe.predict(X_test, threshold=threshold)
        scores_unablated = (pred_unablated == y_test).astype(float)  # (N,)

        # 2. Target direction ablation
        v_target = probe.direction_vector
        X_target_ablated = self.ablate_direction(X_test, v_target)
        pred_target_ablated = probe.predict(X_target_ablated, threshold=threshold)
        scores_target = (pred_target_ablated == y_test).astype(float)  # (N,)

        # 3. Control direction ablations (K random unit vectors)
        control_dirs = self.generate_random_controls(D, rng)
        control_scores_matrix = np.zeros((self.num_controls, N), dtype=float)

        for k in range(self.num_controls):
            X_ctrl = self.ablate_direction(X_test, control_dirs[k])
            pred_ctrl = probe.predict(X_ctrl, threshold=threshold)
            control_scores_matrix[k] = (pred_ctrl == y_test).astype(float)

        # Average control score per sample
        scores_control_mean = np.mean(control_scores_matrix, axis=0)  # (N,)

        # Sample-level delta differences
        delta_target_samples = scores_target - scores_unablated  # (N,)
        delta_control_samples = scores_control_mean - scores_unablated  # (N,)
        causal_effect_samples = scores_target - scores_control_mean  # (N,)

        # 4. Bootstrap uncertainty estimation (1000 resamples)
        boot_indices = rng.integers(0, N, size=(self.n_bootstraps, N))

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
        # (meaning probe ablation degrades performance significantly more than random control)
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
            sample_size=N,
        )
