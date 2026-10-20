"""Out-of-Distribution (OOD) Detection in Residual Activation Space via Mahalanobis Distance.

Implements class-conditional regularized Mahalanobis distance (Lee et al., NeurIPS 2018)
coupled with finite-sample split conformal quantile calibration to detect prompts that
violate the exchangeability assumption of Conformal Risk Control.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class MahalanobisOODDetector:
    """Detects out-of-distribution residual activations via shrinkage Mahalanobis distance.

    Given calibration activations X in R^(N x D) (and optional class labels y), estimates:
        1. Class-conditional centroids mu_c (or global centroid mu_0 if unlabeled).
        2. Shrinkage-regularized covariance:
               Sigma_reg = (1 - gamma) * Sigma_emp + gamma * (Tr(Sigma_emp) / D) * I_D
           and its precision matrix Sigma_inv = pinv(Sigma_reg).
        3. Finite-sample conformal quantile threshold tau_ood at level (1 - quantile_alpha):
               q_level = min(1.0, ceil((N + 1) * (1 - quantile_alpha)) / N)

    At inference time, any activation h with min_c D_M(h, mu_c) > tau_ood is flagged as
    Out-of-Distribution (`is_ood=True`), triggering safe escalation to the frontier oracle.
    """

    def __init__(
        self,
        shrinkage: float = 0.10,
        quantile_alpha: float = 0.05,
    ) -> None:
        if not (0.0 < shrinkage <= 1.0):
            raise ValueError(f"shrinkage must be in (0, 1], got {shrinkage}.")
        if not (0.0 < quantile_alpha < 1.0):
            raise ValueError(f"quantile_alpha must be in (0, 1), got {quantile_alpha}.")

        self.shrinkage = float(shrinkage)
        self.quantile_alpha = float(quantile_alpha)

        self._centroids: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._threshold: float | None = None
        self._feature_dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the Mahalanobis OOD detector has been fitted."""
        return (
            self._centroids is not None
            and self._precision is not None
            and self._threshold is not None
        )

    @property
    def threshold(self) -> float:
        """Calibrated conformal Mahalanobis distance threshold tau_ood."""
        if self._threshold is None:
            raise ValueError("MahalanobisOODDetector has not been fitted yet.")
        return self._threshold

    def fit(
        self,
        activations: np.ndarray,
        labels: list[int] | np.ndarray | None = None,
    ) -> MahalanobisOODDetector:
        """Fits class centroids, shrinkage precision matrix, and conformal threshold."""
        x_mat = np.asarray(activations, dtype=np.float64)
        if x_mat.ndim != 2 or x_mat.shape[0] < 2:
            raise ValueError("activations must be a 2D array with at least 2 samples.")

        n_samples, dim = x_mat.shape
        self._feature_dim = int(dim)

        if labels is not None:
            y_arr = np.asarray(labels, dtype=int)
            if y_arr.shape[0] != n_samples:
                raise ValueError("labels length must match activations sample count.")
            unique_classes = np.unique(y_arr)
            centroids_list = [np.mean(x_mat[y_arr == c], axis=0) for c in unique_classes]
            self._centroids = np.stack(centroids_list, axis=0)
            class_to_idx = {int(c): idx for idx, c in enumerate(unique_classes)}
            residuals = np.stack(
                [x_mat[i] - self._centroids[class_to_idx[int(y_arr[i])]] for i in range(n_samples)],
                axis=0,
            )
        else:
            global_mean = np.mean(x_mat, axis=0, keepdims=True)
            self._centroids = global_mean
            residuals = x_mat - global_mean

        emp_cov = (residuals.T @ residuals) / float(n_samples)
        avg_var = float(np.trace(emp_cov)) / float(dim)
        target_scale = avg_var if avg_var > 1e-8 else 1.0

        reg_cov = (
            (1.0 - self.shrinkage) * emp_cov
            + self.shrinkage * target_scale * np.eye(dim, dtype=np.float64)
            + 1e-6 * np.eye(dim, dtype=np.float64)
        )
        self._precision = np.linalg.pinv(reg_cov, hermitian=True)

        cal_scores = self.score(x_mat)
        q_level = min(1.0, math.ceil((n_samples + 1) * (1.0 - self.quantile_alpha)) / n_samples)
        self._threshold = float(np.quantile(cal_scores, q_level))
        return self

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Computes minimum class-conditional Mahalanobis distance for each sample."""
        if self._centroids is None or self._precision is None or self._feature_dim is None:
            raise ValueError("MahalanobisOODDetector has not been fitted yet.")

        arr = np.asarray(activations, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self._feature_dim:
            raise ValueError(f"Expected feature_dim {self._feature_dim}, got {arr.shape[1]}.")

        n_samples = arr.shape[0]
        n_classes = self._centroids.shape[0]
        dist_per_class = np.zeros((n_samples, n_classes), dtype=np.float64)

        for c_idx in range(n_classes):
            diff = arr - self._centroids[c_idx]
            quad = np.sum((diff @ self._precision) * diff, axis=1)
            dist_per_class[:, c_idx] = np.sqrt(np.maximum(0.0, quad))

        min_dist: np.ndarray = np.min(dist_per_class, axis=1)
        return min_dist

    def is_ood(self, activations: np.ndarray) -> bool | np.ndarray:
        """Returns whether activation(s) exceed the calibrated conformal OOD threshold."""
        raw = np.asarray(activations, dtype=np.float64)
        scores = self.score(raw)
        flags = scores > self.threshold
        if raw.ndim == 1:
            return bool(flags[0])
        return flags

    def save(self, filepath: str | Path) -> None:
        """Persists fitted OOD detector state to a compressed .npz archive."""
        if not self.is_fitted or self._centroids is None or self._precision is None:
            raise ValueError("Cannot save an unfitted MahalanobisOODDetector.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            centroids=self._centroids,
            precision=self._precision,
            meta=np.array(
                [self.shrinkage, self.quantile_alpha, self.threshold],
                dtype=np.float64,
            ),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> MahalanobisOODDetector:
        """Loads a fitted MahalanobisOODDetector from a compressed .npz archive."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Mahalanobis OOD archive not found at: {path}")

        with np.load(str(path)) as data:
            centroids = data["centroids"]
            precision = data["precision"]
            meta = data["meta"]

        detector = cls(shrinkage=float(meta[0]), quantile_alpha=float(meta[1]))
        detector._centroids = np.asarray(centroids, dtype=np.float64)
        detector._precision = np.asarray(precision, dtype=np.float64)
        detector._threshold = float(meta[2])
        detector._feature_dim = int(centroids.shape[1])
        return detector


__all__ = ["MahalanobisOODDetector"]
