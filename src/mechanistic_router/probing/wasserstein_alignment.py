"""Cross-Lingual & Cross-Family Bures-Wasserstein Optimal Transport Alignment.

Implements closed-form Gaussian Optimal Transport (Monge-Kantorovich map) and
Álvarez-Esteban fixed-point Wasserstein barycenters over residual-stream
activation distributions:
  1. Closed-form L2 Bures-Wasserstein distance W_2(N(mu_s, Sigma_s), N(mu_t, Sigma_t))
     capturing both first-moment mean shift and second-moment covariance geometry
  2. Symmetric positive-definite Monge optimal transport operator:
       A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}
       T(x) = mu_t + (x - mu_s) A^T
  3. Zero-shot LinearActivationProbe pullback across languages or model families
     preserving exact logit and probability invariance:
       w_s = A^T w_t,  b_s = b_t + w_t^T (mu_t - A mu_s)
  4. Multi-distribution Wasserstein barycenter computation via Álvarez-Esteban
     fixed-point iteration for multilingual anchor subspace construction

References:
  - Bhatia, Jain & Lim (2019). "On the Bures-Wasserstein distance between
    positive definite matrices." Expositiones Mathematicae 37(2): 165-191.
  - Peyré & Cuturi (2019). "Computational Optimal Transport." Foundations and
    Trends in Machine Learning 11(5-6): 355-607.
  - Agueh & Carlier (2011). "Barycenters in the Wasserstein space." SIAM Journal
    on Mathematical Analysis 43(2): 904-924.
  - Álvarez-Esteban et al. (2016). "A fixed-point approach to barycenters in
    Wasserstein space." Journal of Mathematical Analysis and Applications 441(2).
  - Flamary et al. (2021). "POT: Python Optimal Transport." JMLR 22(78): 1-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_router.probing.linear_probe import LinearActivationProbe


def _spd_powers(
    matrix: np.ndarray,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes symmetric positive-definite square root, inverse square root, and eigenvalues.

    Args:
        matrix: Symmetric matrix of shape (D, D).
        eps: Minimum eigenvalue floor for numerical stability.

    Returns:
        Tuple of (matrix^{1/2}, matrix^{-1/2}, clipped_eigenvalues).
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.maximum(eigvals, eps)
    sqrt_eig = np.sqrt(clipped)
    inv_sqrt_eig = 1.0 / sqrt_eig
    mat_sqrt = (eigvecs * sqrt_eig[None, :]) @ eigvecs.T
    mat_inv_sqrt = (eigvecs * inv_sqrt_eig[None, :]) @ eigvecs.T
    return (
        0.5 * (mat_sqrt + mat_sqrt.T),
        0.5 * (mat_inv_sqrt + mat_inv_sqrt.T),
        clipped,
    )


@dataclass(frozen=True)
class WassersteinAlignmentTelemetry:
    """Telemetry from Bures-Wasserstein optimal transport subspace alignment.

    Attributes:
        wasserstein_distance: L2 Bures-Wasserstein distance W_2 between source and target.
        mean_shift_l2: Euclidean distance ||mu_s - mu_t||_2 between empirical means.
        bures_covariance_distance: Square root of the Bures trace divergence between covariances.
        covariance_frobenius_residual: Post-alignment Frobenius error ||A Sigma_s A^T - Sigma_t||_F.
        source_condition_number: Spectral condition number of regularized Sigma_s.
        target_condition_number: Spectral condition number of regularized Sigma_t.
        feature_dim: Residual-stream activation dimension D.
        n_source_samples: Number of source calibration samples N_s.
        n_target_samples: Number of target calibration samples N_t.
    """

    wasserstein_distance: float
    mean_shift_l2: float
    bures_covariance_distance: float
    covariance_frobenius_residual: float
    source_condition_number: float
    target_condition_number: float
    feature_dim: int
    n_source_samples: int
    n_target_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes alignment telemetry to a JSON-compatible dictionary."""
        return {
            "wasserstein_distance": round(float(self.wasserstein_distance), 6),
            "mean_shift_l2": round(float(self.mean_shift_l2), 6),
            "bures_covariance_distance": round(float(self.bures_covariance_distance), 6),
            "covariance_frobenius_residual": round(float(self.covariance_frobenius_residual), 8),
            "source_condition_number": round(float(self.source_condition_number), 4),
            "target_condition_number": round(float(self.target_condition_number), 4),
            "feature_dim": int(self.feature_dim),
            "n_source_samples": int(self.n_source_samples),
            "n_target_samples": int(self.n_target_samples),
        }


class BuresWassersteinAligner:
    """Closed-form Gaussian Optimal Transport aligner for residual-stream subspaces.

    Unlike rigid Orthogonal Procrustes alignment (which requires paired samples
    and preserves isotropic Euclidean distances), Bures-Wasserstein alignment operates
    on unpaired empirical distributions X_s ~ N(mu_s, Sigma_s) and X_t ~ N(mu_t, Sigma_t)
    and aligns anisotropic covariance ellipsoids across languages or model families.
    """

    def __init__(self, reg_lambda: float = 1e-4, eig_floor: float = 1e-7) -> None:
        if reg_lambda < 0.0:
            raise ValueError(f"reg_lambda must be non-negative, got {reg_lambda}")
        if eig_floor <= 0.0:
            raise ValueError(f"eig_floor must be strictly positive, got {eig_floor}")
        self.reg_lambda = float(reg_lambda)
        self.eig_floor = float(eig_floor)

        self.transport_matrix_: np.ndarray | None = None
        self.source_mean_: np.ndarray | None = None
        self.target_mean_: np.ndarray | None = None
        self.source_cov_: np.ndarray | None = None
        self.target_cov_: np.ndarray | None = None
        self.telemetry_: WassersteinAlignmentTelemetry | None = None

    @property
    def is_fitted(self) -> bool:
        """Returns True if the Monge optimal transport map has been fitted."""
        return (
            self.transport_matrix_ is not None
            and self.source_mean_ is not None
            and self.target_mean_ is not None
        )

    def _estimate_moments(self, X: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array (N, D), got shape {arr.shape}")
        n_samples, dim = arr.shape
        if n_samples < 2:
            raise ValueError(f"{name} requires at least 2 samples to estimate covariance.")
        if dim < 1:
            raise ValueError(f"{name} feature dimension D must be >= 1.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite activation values.")

        mean = np.mean(arr, axis=0)
        centered = arr - mean
        cov = (centered.T @ centered) / float(n_samples - 1)
        cov_reg = 0.5 * (cov + cov.T) + self.reg_lambda * np.eye(dim, dtype=np.float64)
        return mean, cov_reg

    def fit(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
    ) -> WassersteinAlignmentTelemetry:
        """Fits the Monge optimal transport map T(x) = mu_t + A (x - mu_s).

        Args:
            X_source: Source activation matrix of shape (N_s, D) ( need not equal N_t ).
            X_target: Target activation matrix of shape (N_t, D).

        Returns:
            WassersteinAlignmentTelemetry containing W_2 distance and spectral metrics.
        """
        mu_s, cov_s = self._estimate_moments(X_source, "X_source")
        mu_t, cov_t = self._estimate_moments(X_target, "X_target")

        if mu_s.shape[0] != mu_t.shape[0]:
            raise ValueError(
                f"Source dimension ({mu_s.shape[0]}) and target dimension ({mu_t.shape[0]}) "
                "must match for Bures-Wasserstein alignment."
            )

        dim = int(mu_s.shape[0])
        n_s = int(np.asarray(X_source).shape[0])
        n_t = int(np.asarray(X_target).shape[0])

        cov_s_sqrt, cov_s_inv_sqrt, eig_s = _spd_powers(cov_s, eps=self.eig_floor)
        _, _, eig_t = _spd_powers(cov_t, eps=self.eig_floor)

        inner = cov_s_sqrt @ cov_t @ cov_s_sqrt
        inner_sqrt, _, _ = _spd_powers(inner, eps=self.eig_floor)

        # Monge SPD linear map:
        # A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_t Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}
        A = cov_s_inv_sqrt @ inner_sqrt @ cov_s_inv_sqrt
        A = 0.5 * (A + A.T)

        mean_shift_sq = float(np.sum((mu_s - mu_t) ** 2))
        bures_sq = float(max(0.0, np.trace(cov_s + cov_t - 2.0 * inner_sqrt)))
        w2_dist = float(np.sqrt(max(0.0, mean_shift_sq + bures_sq)))

        transported_cov = A @ cov_s @ A.T
        frob_residual = float(np.linalg.norm(transported_cov - cov_t, ord="fro"))

        cond_s = float(np.max(eig_s) / max(float(np.min(eig_s)), self.eig_floor))
        cond_t = float(np.max(eig_t) / max(float(np.min(eig_t)), self.eig_floor))

        self.transport_matrix_ = A
        self.source_mean_ = mu_s
        self.target_mean_ = mu_t
        self.source_cov_ = cov_s
        self.target_cov_ = cov_t
        self.telemetry_ = WassersteinAlignmentTelemetry(
            wasserstein_distance=w2_dist,
            mean_shift_l2=float(np.sqrt(mean_shift_sq)),
            bures_covariance_distance=float(np.sqrt(bures_sq)),
            covariance_frobenius_residual=frob_residual,
            source_condition_number=cond_s,
            target_condition_number=cond_t,
            feature_dim=dim,
            n_source_samples=n_s,
            n_target_samples=n_t,
        )
        return self.telemetry_

    def transform(self, X_source: np.ndarray) -> np.ndarray:
        """Applies the fitted Monge map T(x) = mu_t + (x - mu_s) A^T to source activations.

        Args:
            X_source: 1D vector (D,) or 2D matrix (N, D) in source activation space.

        Returns:
            Aligned activations in target space with identical shape.
        """
        if not self.is_fitted:
            raise RuntimeError("BuresWassersteinAligner must be fitted before transform().")
        assert self.transport_matrix_ is not None
        assert self.source_mean_ is not None
        assert self.target_mean_ is not None

        arr = np.asarray(X_source, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

        if arr.shape[1] != self.source_mean_.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.source_mean_.shape[0]}, "
                f"got {arr.shape[1]}"
            )

        aligned = (arr - self.source_mean_) @ self.transport_matrix_.T + self.target_mean_
        if is_1d:
            return np.asarray(aligned[0], dtype=np.float64)
        return np.asarray(aligned, dtype=np.float64)

    def inverse_transform(self, X_target: np.ndarray) -> np.ndarray:
        """Maps target-space activations back to the source space via T^{-1}(y)."""
        if not self.is_fitted:
            raise RuntimeError("BuresWassersteinAligner must be fitted before inverse_transform().")
        assert self.transport_matrix_ is not None
        assert self.source_mean_ is not None
        assert self.target_mean_ is not None

        arr = np.asarray(X_target, dtype=np.float64)
        is_1d = arr.ndim == 1
        if is_1d:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

        inv_A = np.linalg.pinv(self.transport_matrix_, rcond=1e-10)
        recovered = (arr - self.target_mean_) @ inv_A.T + self.source_mean_
        if is_1d:
            return np.asarray(recovered[0], dtype=np.float64)
        return np.asarray(recovered, dtype=np.float64)

    def pullback_probe(self, target_probe: LinearActivationProbe) -> LinearActivationProbe:
        """Pulls a target-space LinearActivationProbe back into the source space.

        Guarantees exact logit equivalence:
            target_probe.predict_proba(self.transform(X)) == source_probe.predict_proba(X)

        Args:
            target_probe: Fitted LinearActivationProbe trained on target activations.

        Returns:
            A fitted LinearActivationProbe operating natively on unaligned source activations.
        """
        if not self.is_fitted:
            raise RuntimeError("BuresWassersteinAligner must be fitted before pullback_probe().")

        assert self.transport_matrix_ is not None
        assert self.source_mean_ is not None
        assert self.target_mean_ is not None

        w_t = np.asarray(target_probe.weights, dtype=np.float64).reshape(-1)
        b_t = float(target_probe.intercept)

        if w_t.shape[0] != self.source_mean_.shape[0]:
            raise ValueError(
                f"Target probe dimension ({w_t.shape[0]}) does not match aligner "
                f"dimension ({self.source_mean_.shape[0]})."
            )

        # w_t^T (mu_t + A (x - mu_s)) + b_t = (A^T w_t)^T x + (b_t + w_t^T (mu_t - A mu_s))
        w_s = self.transport_matrix_.T @ w_t
        b_s = b_t + float(
            np.dot(w_t, self.target_mean_ - self.transport_matrix_ @ self.source_mean_)
        )

        norm_s = float(np.linalg.norm(w_s))
        dir_s = (
            (w_s / norm_s).astype(np.float32)
            if norm_s > 0.0
            else np.zeros_like(w_s, dtype=np.float32)
        )

        pulled_probe = LinearActivationProbe(
            C=target_probe.C,
            random_state=target_probe.random_state,
            max_iter=target_probe.max_iter,
            feature_dim=int(self.source_mean_.shape[0]),
            num_groups=1,
            group_lasso_lambda=0.0,
        )
        pulled_probe._weights = w_s.astype(np.float32)
        pulled_probe._direction_vector = dir_s
        pulled_probe._norm = norm_s
        pulled_probe._intercept = float(b_s)
        pulled_probe._classes = np.array([0, 1], dtype=int)
        return pulled_probe

    def compute_barycenter(
        self,
        distributions: list[np.ndarray],
        weights: list[float] | np.ndarray | None = None,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the L2-Wasserstein barycenter (mu_bar, Sigma_bar) across M distributions.

        Uses the Álvarez-Esteban et al. (2016) fixed-point iteration:
            Sigma_{k+1} = Sigma_k^{-1/2} (
                sum_m w_m (Sigma_k^{1/2} Sigma_m Sigma_k^{1/2})^{1/2}
            )^2 Sigma_k^{-1/2}

        Args:
            distributions: List of M >= 2 activation matrices, each of shape (N_m, D).
            weights: Optional simplex weights of length M (defaults to uniform 1/M).
            max_iter: Maximum number of fixed-point iterations.
            tol: Frobenius convergence tolerance on Sigma_bar.

        Returns:
            Tuple (barycenter_mean, barycenter_covariance) of shapes (D,) and (D, D).
        """
        if len(distributions) < 2:
            raise ValueError("compute_barycenter requires at least 2 activation distributions.")

        m_count = len(distributions)
        if weights is None:
            w = np.full(m_count, 1.0 / float(m_count), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != m_count:
                raise ValueError(f"Expected {m_count} weights, got {w.shape[0]}.")
            if np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
                raise ValueError("Barycenter weights must be non-negative with positive sum.")
            w = w / float(np.sum(w))

        means: list[np.ndarray] = []
        covs: list[np.ndarray] = []
        dim: int | None = None

        for idx, dist in enumerate(distributions):
            mu_m, cov_m = self._estimate_moments(dist, f"distributions[{idx}]")
            if dim is None:
                dim = int(mu_m.shape[0])
            elif int(mu_m.shape[0]) != dim:
                raise ValueError("All distributions must share the same feature dimension D.")
            means.append(mu_m)
            covs.append(cov_m)

        assert dim is not None
        mu_bar = np.zeros(dim, dtype=np.float64)
        sigma_k = np.zeros((dim, dim), dtype=np.float64)
        for weight, mu_m, cov_m in zip(w, means, covs, strict=True):
            mu_bar += weight * mu_m
            sigma_k += weight * cov_m

        for _ in range(max_iter):
            sig_sqrt, sig_inv_sqrt, _ = _spd_powers(sigma_k, eps=self.eig_floor)
            weighted_sum = np.zeros((dim, dim), dtype=np.float64)
            for weight, cov_m in zip(w, covs, strict=True):
                inner = sig_sqrt @ cov_m @ sig_sqrt
                inner_sqrt, _, _ = _spd_powers(inner, eps=self.eig_floor)
                weighted_sum += weight * inner_sqrt

            sigma_next = sig_inv_sqrt @ (weighted_sum @ weighted_sum) @ sig_inv_sqrt
            sigma_next = 0.5 * (sigma_next + sigma_next.T)
            diff = float(np.linalg.norm(sigma_next - sigma_k, ord="fro"))
            sigma_k = sigma_next
            if diff <= tol:
                break

        return mu_bar, sigma_k

    def save(self, filepath: str | Path) -> None:
        """Persists the fitted Bures-Wasserstein transport map to a compressed .npz archive."""
        if not self.is_fitted or self.telemetry_ is None:
            raise RuntimeError("Cannot save an unfitted BuresWassersteinAligner.")
        assert self.transport_matrix_ is not None
        assert self.source_mean_ is not None
        assert self.target_mean_ is not None
        assert self.source_cov_ is not None
        assert self.target_cov_ is not None

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            transport_matrix=self.transport_matrix_,
            source_mean=self.source_mean_,
            target_mean=self.target_mean_,
            source_cov=self.source_cov_,
            target_cov=self.target_cov_,
            reg_lambda=np.array([self.reg_lambda], dtype=np.float64),
            eig_floor=np.array([self.eig_floor], dtype=np.float64),
            wasserstein_distance=np.array([self.telemetry_.wasserstein_distance], dtype=np.float64),
            mean_shift_l2=np.array([self.telemetry_.mean_shift_l2], dtype=np.float64),
            bures_covariance_distance=np.array(
                [self.telemetry_.bures_covariance_distance], dtype=np.float64
            ),
            covariance_frobenius_residual=np.array(
                [self.telemetry_.covariance_frobenius_residual], dtype=np.float64
            ),
            source_condition_number=np.array(
                [self.telemetry_.source_condition_number], dtype=np.float64
            ),
            target_condition_number=np.array(
                [self.telemetry_.target_condition_number], dtype=np.float64
            ),
            n_source_samples=np.array([self.telemetry_.n_source_samples], dtype=np.int64),
            n_target_samples=np.array([self.telemetry_.n_target_samples], dtype=np.int64),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> BuresWassersteinAligner:
        """Loads a saved BuresWassersteinAligner from a .npz archive."""
        data = np.load(Path(filepath), allow_pickle=False)
        aligner = cls(
            reg_lambda=float(data["reg_lambda"][0]),
            eig_floor=float(data["eig_floor"][0]),
        )
        aligner.transport_matrix_ = np.asarray(data["transport_matrix"], dtype=np.float64)
        aligner.source_mean_ = np.asarray(data["source_mean"], dtype=np.float64)
        aligner.target_mean_ = np.asarray(data["target_mean"], dtype=np.float64)
        aligner.source_cov_ = np.asarray(data["source_cov"], dtype=np.float64)
        aligner.target_cov_ = np.asarray(data["target_cov"], dtype=np.float64)
        aligner.telemetry_ = WassersteinAlignmentTelemetry(
            wasserstein_distance=float(data["wasserstein_distance"][0]),
            mean_shift_l2=float(data["mean_shift_l2"][0]),
            bures_covariance_distance=float(data["bures_covariance_distance"][0]),
            covariance_frobenius_residual=float(data["covariance_frobenius_residual"][0]),
            source_condition_number=float(data["source_condition_number"][0]),
            target_condition_number=float(data["target_condition_number"][0]),
            feature_dim=int(aligner.source_mean_.shape[0]),
            n_source_samples=int(data["n_source_samples"][0]),
            n_target_samples=int(data["n_target_samples"][0]),
        )
        return aligner
