"""Low-Rank Bilinear Cross-Layer Interaction Probe.

Models second-order multiplicative circuit composition between shallow and deep
residual-stream subspaces:
1. Virtual Weight Cross-Covariance SVD Initialization (Elhage et al., 2021):
   Extracts the optimal rank-r bilinear interaction factors (U, V) via Eckart-Young
   SVD of the class-conditional cross-layer covariance difference:
       Delta_Sigma = E[h^(1) h^(2)^T | y=1] - E[h^(1) h^(2)^T | y=0].
2. Low-Rank Bilinear Pooling & Joint Logistic Refinement (Rendle, ICDM 2010;
   Kim et al., ICLR 2017):
   Evaluates logit(h) = w^T h + h^(1)^T (U V^T) h^(2) + b in O(r * D) time and
   jointly optimizes (w, b, U, V) under L2 regularization.
3. Compositional Energy Telemetry:
   Decomposes each routing prediction into additive first-order linear energy and
   multiplicative second-order bilinear interaction energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BilinearInteractionTelemetry:
    """Decomposed first-order linear and second-order bilinear routing telemetry."""

    linear_logit: float
    bilinear_logit: float
    total_logit: float
    prob_complex: float
    bilinear_energy_ratio: float
    top_singular_values: list[float]


class LowRankBilinearProbe:
    """Rank-r bilinear activation probe capturing cross-layer multiplicative interactions."""

    def __init__(
        self,
        rank: int = 4,
        l2_reg: float = 1e-2,
        learning_rate: float = 0.05,
        num_steps: int = 120,
        seed: int = 42,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be >= 1.")
        if l2_reg < 0.0:
            raise ValueError("l2_reg must be >= 0.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        self.rank = int(rank)
        self.l2_reg = float(l2_reg)
        self.learning_rate = float(learning_rate)
        self.num_steps = int(num_steps)
        self.seed = int(seed)

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._w: np.ndarray | None = None
        self._b: float = 0.0
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._singular_values: list[float] = []
        self._d1: int = 0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the bilinear probe parameters have been fitted."""
        return self._is_fitted

    def _split_subspaces(self, X_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Splits normalized activations into shallow h^(1) and deep h^(2) subspaces."""
        return X_norm[:, : self._d1], X_norm[:, self._d1 :]

    def fit(
        self,
        activations: np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> LowRankBilinearProbe:
        """Fits first-order linear weights and rank-r bilinear interaction factors."""
        X = np.asarray(activations, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError("activations must be a 2D array of shape (N, D).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in activations and labels must match.")
        if X.shape[1] < 2:
            raise ValueError("Feature dimension D must be >= 2 for bilinear splitting.")
        unique_classes = np.unique(y)
        if unique_classes.size < 2:
            raise ValueError("At least two distinct classes (0 and 1) are required.")

        n_samples, d_dim = X.shape
        self._d1 = d_dim // 2

        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self._mean[None, :]) / self._std[None, :]

        h1, h2 = self._split_subspaces(X_norm)
        pos_mask = y >= 0.5
        neg_mask = ~pos_mask

        # 1. Initialize rank-r (U, V) from SVD of class-conditional cross-covariance difference
        cov_pos = (h1[pos_mask].T @ h2[pos_mask]) / max(1, int(np.sum(pos_mask)))
        cov_neg = (h1[neg_mask].T @ h2[neg_mask]) / max(1, int(np.sum(neg_mask)))
        delta_cov = cov_pos - cov_neg

        u_svd, s_svd, vt_svd = np.linalg.svd(delta_cov, full_matrices=False)
        eff_r = min(self.rank, u_svd.shape[1], vt_svd.shape[0])
        sqrt_s = np.sqrt(np.maximum(s_svd[:eff_r], 1e-6))

        u_mat = u_svd[:, :eff_r] * sqrt_s[None, :]
        v_mat = vt_svd[:eff_r, :].T * sqrt_s[None, :]

        # Initialize linear weights from mean difference
        w_vec = np.mean(X_norm[pos_mask], axis=0) - np.mean(X_norm[neg_mask], axis=0)
        w_norm = float(np.linalg.norm(w_vec))
        if w_norm > 1e-8:
            w_vec = w_vec / w_norm
        b_val = float(np.log(max(np.mean(y), 1e-4) / max(1.0 - np.mean(y), 1e-4)))

        # 2. Joint gradient descent refinement on regularized binary cross-entropy
        lr = self.learning_rate
        reg = self.l2_reg
        for _ in range(self.num_steps):
            proj_u = h1 @ u_mat  # (N, r)
            proj_v = h2 @ v_mat  # (N, r)
            z_lin = X_norm @ w_vec + b_val
            z_bilin = np.sum(proj_u * proj_v, axis=1)
            logits = np.clip(z_lin + z_bilin, -35.0, 35.0)
            probs = 1.0 / (1.0 + np.exp(-logits))

            err = (probs - y) / n_samples  # (N,)
            grad_w = (X_norm.T @ err) + reg * w_vec
            grad_b = float(np.sum(err))
            grad_u = (h1.T @ (err[:, None] * proj_v)) + reg * u_mat
            grad_v = (h2.T @ (err[:, None] * proj_u)) + reg * v_mat

            w_vec -= lr * grad_w
            b_val -= lr * grad_b
            u_mat -= lr * grad_u
            v_mat -= lr * grad_v

        # Record singular values of fitted virtual interaction matrix U V^T
        _, final_s, _ = np.linalg.svd(u_mat @ v_mat.T, full_matrices=False)
        self._singular_values = [float(sv) for sv in final_s[:eff_r]]
        self._w = w_vec
        self._b = b_val
        self._u = u_mat
        self._v = v_mat
        self._is_fitted = True
        return self

    def _compute_logits(self, activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (linear_logits, bilinear_logits) for a 1D or 2D activation array."""
        if (
            not self._is_fitted
            or self._mean is None
            or self._std is None
            or self._w is None
            or self._u is None
            or self._v is None
        ):
            raise RuntimeError("LowRankBilinearProbe must be fitted before inference.")

        X = np.asarray(activations, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        elif X.ndim != 2:
            raise ValueError("activations must be 1D or 2D.")
        if X.shape[1] != self._mean.shape[0]:
            raise ValueError(
                f"Expected activation dimension {self._mean.shape[0]}, got {X.shape[1]}."
            )

        X_norm = (X - self._mean[None, :]) / self._std[None, :]
        h1, h2 = self._split_subspaces(X_norm)
        z_lin = X_norm @ self._w + self._b
        z_bilin = np.sum((h1 @ self._u) * (h2 @ self._v), axis=1)
        return z_lin, z_bilin

    def decision_function(self, activations: np.ndarray) -> np.ndarray:
        """Computes total combined logits z_lin + z_bilin."""
        z_lin, z_bilin = self._compute_logits(activations)
        return np.asarray(z_lin + z_bilin, dtype=np.float64)

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """Predicts complexity probability sigma(z_lin + z_bilin) in [0, 1]."""
        logits = np.clip(self.decision_function(activations), -35.0, 35.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)

    def inspect_interaction(self, activation: np.ndarray) -> BilinearInteractionTelemetry:
        """Decomposes a single activation's prediction into linear and bilinear components."""
        z_lin_arr, z_bilin_arr = self._compute_logits(activation)
        z_lin = float(z_lin_arr[0])
        z_bilin = float(z_bilin_arr[0])
        z_tot = z_lin + z_bilin
        prob = float(1.0 / (1.0 + np.exp(-np.clip(z_tot, -35.0, 35.0))))
        ratio = float(abs(z_bilin) / (abs(z_lin) + abs(z_bilin) + 1e-12))

        return BilinearInteractionTelemetry(
            linear_logit=z_lin,
            bilinear_logit=z_bilin,
            total_logit=z_tot,
            prob_complex=prob,
            bilinear_energy_ratio=ratio,
            top_singular_values=list(self._singular_values),
        )

    def local_gradient_and_hessian_bound(self, activation: np.ndarray) -> tuple[np.ndarray, float]:
        """Computes exact input gradient nabla_x z(x) and Hessian spectral norm bound L_H."""
        if (
            not self._is_fitted
            or self._mean is None
            or self._std is None
            or self._w is None
            or self._u is None
            or self._v is None
        ):
            raise RuntimeError("LowRankBilinearProbe must be fitted before gradient computation.")

        x_vec = np.asarray(activation, dtype=np.float64).ravel()
        if x_vec.shape[0] != self._mean.shape[0]:
            raise ValueError(
                f"Expected activation dimension {self._mean.shape[0]}, got {x_vec.shape[0]}."
            )

        x_norm = (x_vec - self._mean) / self._std
        h1 = x_norm[: self._d1]
        h2 = x_norm[self._d1 :]

        grad_h1 = self._w[: self._d1] + self._u @ (self._v.T @ h2)
        grad_h2 = self._w[self._d1 :] + self._v @ (self._u.T @ h1)
        grad_norm_space = np.concatenate([grad_h1, grad_h2])
        grad_x = grad_norm_space / self._std

        # Hessian in raw x-space is diag(1/std) @ [[0, U V^T], [V U^T, 0]] @ diag(1/std)
        top_sv = float(self._singular_values[0]) if self._singular_values else 0.0
        min_std_sq = float(np.min(self._std) ** 2)
        hessian_spectral_bound = float(top_sv / max(min_std_sq, 1e-12))
        return grad_x, hessian_spectral_bound


__all__ = [
    "BilinearInteractionTelemetry",
    "LowRankBilinearProbe",
]
