"""Cross-Family Universal Activation Alignment via Orthogonal Procrustes Mapping.

Enables zero-label transfer of calibrated linear activation probes across
heterogeneous prefill encoders (e.g., SmolLM-135M <-> Llama-3.2-1B <-> Qwen2.5-1.5B)
by learning an orthogonal (or semi-orthogonal) alignment map on anchor prompts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .linear_probe import LinearActivationProbe


class OrthogonalProcrustesAligner:
    """Aligns residual-stream activation spaces across model families via SVD Procrustes.

    Given paired anchor activations X_source in R^(N x D_s) and X_target in R^(N x D_t)
    extracted for the same N anchor prompts:
    1. Centers and normalizes scale across source and target spaces.
    2. Solves the Orthogonal (or Reduced-Rank Semi-Orthogonal) Procrustes problem:
           Omega* = argmin_{Omega^T Omega = I_k} || X_source_tilde @ Omega - X_target_tilde ||_F
       via Singular Value Decomposition of X_source_tilde^T @ X_target_tilde.
    3. Projects source activations into the target space (`transform`) or directly pulls back
       a `LinearActivationProbe` trained on the target space into the source space (`align_probe`).
    """

    def __init__(self, center: bool = True, scale: bool = True) -> None:
        self.center = bool(center)
        self.scale = bool(scale)

        self._rotation_matrix: np.ndarray | None = None
        self._source_mean: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None
        self._source_scale: float = 1.0
        self._target_scale: float = 1.0
        self._source_dim: int | None = None
        self._target_dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the Procrustes alignment map has been fitted."""
        return self._rotation_matrix is not None

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Fitted transformation matrix of shape (D_source, D_target)."""
        if self._rotation_matrix is None:
            raise ValueError("OrthogonalProcrustesAligner has not been fitted yet.")
        return self._rotation_matrix

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> OrthogonalProcrustesAligner:
        """Fits the Procrustes transformation from X_source (N, D_s) to X_target (N, D_t)."""
        src = np.asarray(X_source, dtype=np.float64)
        tgt = np.asarray(X_target, dtype=np.float64)

        if src.ndim != 2 or tgt.ndim != 2:
            raise ValueError(
                f"X_source and X_target must be 2D matrices, got {src.ndim}D and {tgt.ndim}D."
            )
        if src.shape[0] != tgt.shape[0] or src.shape[0] < 2:
            raise ValueError(
                "X_source and X_target must have the same number of anchor samples (N >= 2)."
            )

        self._source_dim = int(src.shape[1])
        self._target_dim = int(tgt.shape[1])

        if self.center:
            self._source_mean = np.mean(src, axis=0)
            self._target_mean = np.mean(tgt, axis=0)
            src_c = src - self._source_mean
            tgt_c = tgt - self._target_mean
        else:
            self._source_mean = np.zeros(self._source_dim, dtype=np.float64)
            self._target_mean = np.zeros(self._target_dim, dtype=np.float64)
            src_c = src.copy()
            tgt_c = tgt.copy()

        if self.scale:
            src_norm = float(np.linalg.norm(src_c, ord="fro"))
            tgt_norm = float(np.linalg.norm(tgt_c, ord="fro"))
            self._source_scale = src_norm if src_norm > 1e-12 else 1.0
            self._target_scale = tgt_norm if tgt_norm > 1e-12 else 1.0
            src_c = src_c / self._source_scale
            tgt_c = tgt_c / self._target_scale
        else:
            self._source_scale = 1.0
            self._target_scale = 1.0

        # Cross-covariance matrix M = src_c^T @ tgt_c of shape (D_source, D_target)
        cross_cov = src_c.T @ tgt_c
        u_mat, _, vt_mat = np.linalg.svd(cross_cov, full_matrices=False)
        # Semi-orthogonal / orthogonal Procrustes map: U @ V^T in R^(D_source x D_target)
        self._rotation_matrix = (u_mat @ vt_mat).astype(np.float64)
        return self

    def transform(self, X_source: np.ndarray) -> np.ndarray:
        """Maps source activations (N, D_s) or (D_s,) into the target activation space."""
        if self._rotation_matrix is None or self._source_mean is None or self._target_mean is None:
            raise ValueError("OrthogonalProcrustesAligner has not been fitted yet.")

        src = np.asarray(X_source, dtype=np.float64)
        is_1d = src.ndim == 1
        if is_1d:
            src = src[None, :]

        if src.shape[1] != self._source_dim:
            raise ValueError(f"Expected source dimension {self._source_dim}, got {src.shape[1]}.")

        src_centered = (src - self._source_mean) / self._source_scale
        aligned = (src_centered @ self._rotation_matrix) * self._target_scale + self._target_mean
        aligned_f32 = np.asarray(aligned, dtype=np.float32)
        if is_1d:
            return np.asarray(aligned_f32[0], dtype=np.float32)
        return aligned_f32

    def alignment_cosine_similarity(self, X_source: np.ndarray, X_target: np.ndarray) -> float:
        """Computes mean row-wise cosine similarity between aligned X_source and X_target."""
        aligned = self.transform(X_source).astype(np.float64)
        tgt = np.asarray(X_target, dtype=np.float64)
        if aligned.shape != tgt.shape:
            raise ValueError("Aligned source and target shapes do not match.")

        num = np.sum(aligned * tgt, axis=1)
        denom = np.linalg.norm(aligned, axis=1) * np.linalg.norm(tgt, axis=1)
        valid = denom > 1e-12
        if not np.any(valid):
            return 0.0
        return float(np.mean(num[valid] / denom[valid]))

    def align_probe(self, target_probe: LinearActivationProbe) -> LinearActivationProbe:
        """Pulls back a fitted target-space LinearActivationProbe into the source space.

        Since target margin is:
            m(x_s) = w_t^T ( ((x_s - mu_s) / s_s) @ R * s_t + mu_t ) + b_t
                   = x_s^T ( (s_t / s_s) R w_t ) + [ b_t + w_t^T mu_t - (s_t / s_s) mu_s^T R w_t ]
        this constructs an exact equivalent `LinearActivationProbe` operating directly
        on raw source activations `x_s` without runtime alignment overhead.
        """
        if (
            self._rotation_matrix is None
            or self._source_mean is None
            or self._target_mean is None
            or self._source_dim is None
        ):
            raise ValueError("OrthogonalProcrustesAligner has not been fitted yet.")

        w_target = np.asarray(target_probe.weights, dtype=np.float64)
        b_target = float(target_probe.intercept)

        if w_target.shape[0] != self._target_dim:
            raise ValueError(
                f"Target probe dimension {w_target.shape[0]} does not match "
                f"aligner target dimension {self._target_dim}."
            )

        scale_ratio = self._target_scale / self._source_scale
        w_source = scale_ratio * (self._rotation_matrix @ w_target)
        b_source = (
            b_target
            + float(np.dot(w_target, self._target_mean))
            - float(np.dot(self._source_mean, w_source))
        )

        norm_s = float(np.linalg.norm(w_source))
        dir_s = (
            (w_source / norm_s).astype(np.float32)
            if norm_s > 0.0
            else np.zeros_like(w_source, dtype=np.float32)
        )

        source_probe = LinearActivationProbe(
            C=target_probe.C,
            random_state=target_probe.random_state,
            max_iter=target_probe.max_iter,
            feature_dim=self._source_dim,
            num_groups=1,
            group_lasso_lambda=0.0,
        )
        source_probe._weights = w_source.astype(np.float32)
        source_probe._direction_vector = dir_s
        source_probe._norm = norm_s
        source_probe._intercept = float(b_source)
        source_probe._classes = np.array([0, 1], dtype=int)
        return source_probe

    def save(self, filepath: str | Path) -> None:
        """Saves fitted Procrustes alignment parameters to a compressed .npz archive."""
        if self._rotation_matrix is None or self._source_mean is None or self._target_mean is None:
            raise ValueError("Cannot save an unfitted OrthogonalProcrustesAligner.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            rotation_matrix=self._rotation_matrix,
            source_mean=self._source_mean,
            target_mean=self._target_mean,
            meta=np.array(
                [
                    float(self.center),
                    float(self.scale),
                    self._source_scale,
                    self._target_scale,
                ],
                dtype=np.float64,
            ),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> OrthogonalProcrustesAligner:
        """Loads a fitted OrthogonalProcrustesAligner from a compressed .npz archive."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Procrustes alignment archive not found at: {path}")

        with np.load(str(path)) as data:
            rot = data["rotation_matrix"]
            src_mean = data["source_mean"]
            tgt_mean = data["target_mean"]
            meta = data["meta"]

        aligner = cls(center=bool(meta[0]), scale=bool(meta[1]))
        aligner._rotation_matrix = np.asarray(rot, dtype=np.float64)
        aligner._source_mean = np.asarray(src_mean, dtype=np.float64)
        aligner._target_mean = np.asarray(tgt_mean, dtype=np.float64)
        aligner._source_scale = float(meta[2])
        aligner._target_scale = float(meta[3])
        aligner._source_dim = int(rot.shape[0])
        aligner._target_dim = int(rot.shape[1])
        return aligner


__all__ = ["OrthogonalProcrustesAligner"]
