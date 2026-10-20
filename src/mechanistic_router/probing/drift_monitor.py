"""Population-Level Activation Drift Monitor via Kernel MMD.

Detects non-stationary dataset and representation shift between calibration
reference activations and live production traffic using a multi-bandwidth
Radial Basis Function (RBF) Maximum Mean Discrepancy (MMD^2) two-sample test
with exact permutation p-values.

When statistically significant population drift is detected, automatically
recommends or applies conservative tightening to Conformal Risk Control
thresholds.

References:
- Gretton et al. (JMLR 2012). A Kernel Two-Sample Test.
- Rabanser et al. (NeurIPS 2019). Failing Loudly: An Empirical Study of Methods
  for Detecting Dataset Shift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .conformal import ConformalRiskController


@dataclass(frozen=True)
class DriftTestResult:
    """Result of a two-sample kernel MMD^2 population drift evaluation."""

    mmd_squared: float
    p_value: float
    mean_cosine_shift: float
    is_drift_detected: bool
    reference_size: int
    window_size: int
    recommended_alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes the drift test result to a JSON-compatible dictionary."""
        return asdict(self)


class PopulationDriftMonitor:
    """Monitors residual-stream activation distributions for covariate shift via MMD^2."""

    def __init__(
        self,
        reference_activations: np.ndarray,
        window_capacity: int = 128,
        alpha_test: float = 0.05,
        min_mmd_threshold: float = 0.02,
        num_permutations: int = 100,
        seed: int = 42,
    ) -> None:
        ref = np.asarray(reference_activations, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[0] < 4:
            raise ValueError(
                "reference_activations must be a 2D array with at least 4 samples (N, D)."
            )
        if window_capacity < 4:
            raise ValueError(f"window_capacity must be >= 4, got {window_capacity}.")
        if not (0.0 < alpha_test < 1.0):
            raise ValueError(f"alpha_test must be in (0, 1), got {alpha_test}.")
        if num_permutations < 10:
            raise ValueError(f"num_permutations must be >= 10, got {num_permutations}.")

        self.reference_activations = ref
        self.feature_dim = int(ref.shape[1])
        self.window_capacity = window_capacity
        self.alpha_test = alpha_test
        self.min_mmd_threshold = min_mmd_threshold
        self.num_permutations = num_permutations
        self.seed = seed

        self._window: deque[np.ndarray] = deque(maxlen=window_capacity)
        self._ref_mean = np.mean(ref, axis=0)

    def record_activation(self, activation: np.ndarray) -> None:
        """Appends a single production activation vector into the sliding window."""
        vec = np.asarray(activation, dtype=np.float64).ravel()
        if vec.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected activation dimension {self.feature_dim}, got {vec.shape[0]}."
            )
        self._window.append(vec)

    def clear_window(self) -> None:
        """Clears the current production sliding window."""
        self._window.clear()

    @staticmethod
    def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_norm = np.sum(x**2, axis=1, keepdims=True)
        y_norm = np.sum(y**2, axis=1, keepdims=True).T
        sq_dists = x_norm + y_norm - 2.0 * np.dot(x, y.T)
        return np.asarray(np.maximum(sq_dists, 0.0), dtype=np.float64)

    @classmethod
    def compute_multi_rbf_kernel(cls, z: np.ndarray) -> np.ndarray:
        """Computes a 3-bandwidth RBF kernel matrix over pooled samples Z."""
        sq_dists = cls._pairwise_sq_dists(z, z)
        upper_tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]
        positive_dists = upper_tri[upper_tri > 1e-12]
        median_sq = float(np.median(positive_dists)) if positive_dists.size > 0 else 1.0
        base_gamma = 1.0 / max(2.0 * median_sq, 1e-8)

        gammas = (0.5 * base_gamma, 1.0 * base_gamma, 2.0 * base_gamma)
        k_mat = np.zeros_like(sq_dists)
        for g in gammas:
            k_mat += np.exp(-g * sq_dists)
        k_mat /= float(len(gammas))
        return k_mat

    @staticmethod
    def _mmd_sq_from_kernel(k_mat: np.ndarray, n_x: int, n_y: int) -> float:
        k_xx = k_mat[:n_x, :n_x]
        k_yy = k_mat[n_x:, n_x:]
        k_xy = k_mat[:n_x, n_x:]

        # Unbiased off-diagonal U-statistic for XX and YY
        sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / max(float(n_x * (n_x - 1)), 1.0)
        sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / max(float(n_y * (n_y - 1)), 1.0)
        sum_xy = float(np.mean(k_xy))
        return float(max(0.0, sum_xx + sum_yy - 2.0 * sum_xy))

    def evaluate_drift(
        self,
        window_activations: np.ndarray | None = None,
        base_conformal_alpha: float = 0.10,
    ) -> DriftTestResult:
        """Executes the two-sample kernel MMD^2 permutation test against reference activations."""
        if window_activations is not None:
            win = np.asarray(window_activations, dtype=np.float64)
            if win.ndim == 1:
                win = win.reshape(1, -1)
        else:
            if len(self._window) < 4:
                raise ValueError(
                    "At least 4 activations are required in the sliding window to evaluate drift."
                )
            win = np.vstack(list(self._window))

        if win.ndim != 2 or win.shape[0] < 4:
            raise ValueError("window_activations must be a 2D array with at least 4 samples.")
        if win.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim {self.feature_dim}, got {win.shape[1]}.")

        n_x = int(self.reference_activations.shape[0])
        n_y = int(win.shape[0])
        pooled = np.vstack([self.reference_activations, win])
        k_mat = self.compute_multi_rbf_kernel(pooled)

        observed_mmd = self._mmd_sq_from_kernel(k_mat, n_x, n_y)

        rng = np.random.default_rng(self.seed)
        exceed_count = 0
        total_n = n_x + n_y
        for _ in range(self.num_permutations):
            perm = rng.permutation(total_n)
            k_perm = k_mat[np.ix_(perm, perm)]
            perm_mmd = self._mmd_sq_from_kernel(k_perm, n_x, n_y)
            if perm_mmd >= observed_mmd:
                exceed_count += 1

        p_val = float(exceed_count + 1) / float(self.num_permutations + 1)

        win_mean = np.mean(win, axis=0)
        norm_ref = float(np.linalg.norm(self._ref_mean))
        norm_win = float(np.linalg.norm(win_mean))
        if norm_ref > 1e-12 and norm_win > 1e-12:
            cos_sim = float(
                np.clip(np.dot(self._ref_mean, win_mean) / (norm_ref * norm_win), -1.0, 1.0)
            )
            mean_cosine_shift = float(max(0.0, 1.0 - cos_sim))
        else:
            mean_cosine_shift = 0.0

        is_drift = bool(p_val <= self.alpha_test and observed_mmd >= self.min_mmd_threshold)
        if is_drift:
            # Conservatively tighten risk budget alpha proportional to MMD^2 shift
            tightened = base_conformal_alpha * float(np.exp(-1.5 * observed_mmd))
            rec_alpha = float(np.clip(tightened, 0.01, base_conformal_alpha))
        else:
            rec_alpha = float(base_conformal_alpha)

        return DriftTestResult(
            mmd_squared=observed_mmd,
            p_value=p_val,
            mean_cosine_shift=mean_cosine_shift,
            is_drift_detected=is_drift,
            reference_size=n_x,
            window_size=n_y,
            recommended_alpha=rec_alpha,
        )

    def auto_tighten_conformal_controller(
        self,
        controller: ConformalRiskController,
        window_activations: np.ndarray | None = None,
    ) -> DriftTestResult:
        """Evaluates drift and automatically tightens conformal thresholds if shift is detected."""
        res = self.evaluate_drift(
            window_activations=window_activations,
            base_conformal_alpha=controller.alpha,
        )
        if res.is_drift_detected and controller.is_calibrated:
            controller._effective_alpha = res.recommended_alpha
            ratio = res.recommended_alpha / max(controller.alpha, 1e-6)
            base_low = controller.summary.low_threshold
            base_high = controller.summary.high_threshold
            new_low = float(np.clip(base_low * ratio, 0.0, 1.0))
            new_high = float(np.clip(max(new_low, base_high * ratio), 0.0, 1.0))
            controller._low_threshold = new_low
            controller._high_threshold = new_high
        return res


__all__ = [
    "DriftTestResult",
    "PopulationDriftMonitor",
]
