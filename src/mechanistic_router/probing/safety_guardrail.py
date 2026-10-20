"""Mechanistic Safety & Jailbreak Subspace Rejection Gate.

Implements white-box prefill safety screening in residual-stream activation space:
1. Primary Refusal / Harm Direction Extraction (Arditi et al., NeurIPS 2024):
   Computes the unit difference-in-means vector v_harm between harmful/adversarial
   and benign prefill activations.
2. Principal Jailbreak Subspace via SVD (Zou et al., 2023):
   Extracts the top-r orthonormal basis U_sub of centered adversarial activations
   to capture multi-directional jailbreak and prompt-injection manifolds.
3. Split Conformal Safety Calibration (Angelopoulos et al., 2024):
   Calibrates the non-conformity rejection threshold tau_safety on benign calibration
   activations so that the false-positive rejection rate on benign traffic is bounded
   by alpha_fp in finite samples.
4. Orthogonal Subspace Ablation:
   Provides closed-form projection removal of both v_harm and U_sub from residual
   activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyEvaluationResult:
    """Diagnostic result from evaluating a prefill activation against the safety guardrail."""

    is_unsafe: bool
    safety_score: float
    direction_projection: float
    subspace_energy_ratio: float
    threshold: float
    false_positive_alpha: float


class MechanisticSafetyGuardrail:
    """White-box residual-stream guardrail for detecting and ablating jailbreak subspaces."""

    def __init__(
        self,
        subspace_rank: int = 3,
        false_positive_alpha: float = 0.05,
        subspace_weight: float = 0.35,
    ) -> None:
        if subspace_rank < 1:
            raise ValueError("subspace_rank must be >= 1.")
        if not (0.0 < false_positive_alpha < 1.0):
            raise ValueError("false_positive_alpha must lie strictly in (0, 1).")
        if not (0.0 <= subspace_weight <= 1.0):
            raise ValueError("subspace_weight must lie in [0, 1].")

        self.subspace_rank = int(subspace_rank)
        self.false_positive_alpha = float(false_positive_alpha)
        self.subspace_weight = float(subspace_weight)

        self._benign_mean: np.ndarray | None = None
        self._harm_direction: np.ndarray | None = None
        self._subspace_basis: np.ndarray | None = None
        self._proj_scale: float = 1.0
        self._proj_shift: float = 0.0
        self._threshold: float = 0.5
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the safety guardrail has been calibrated on benign and harmful activations."""
        return self._is_fitted

    @property
    def threshold(self) -> float:
        """Calibrated conformal safety non-conformity threshold."""
        if not self._is_fitted:
            raise RuntimeError("MechanisticSafetyGuardrail has not been fitted yet.")
        return self._threshold

    @property
    def harm_direction(self) -> np.ndarray:
        """Unit refusal/harm direction vector in R^D."""
        if not self._is_fitted or self._harm_direction is None:
            raise RuntimeError("MechanisticSafetyGuardrail has not been fitted yet.")
        return self._harm_direction.copy()

    def _compute_raw_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes signed harm-direction projection and subspace energy ratio for batch X."""
        assert self._benign_mean is not None
        assert self._harm_direction is not None
        assert self._subspace_basis is not None

        centered = X - self._benign_mean[None, :]
        dir_proj = centered @ self._harm_direction

        sub_coords = centered @ self._subspace_basis
        sub_energy = np.sum(np.square(sub_coords), axis=1)
        total_energy = np.sum(np.square(centered), axis=1) + 1e-12
        energy_ratio = np.clip(sub_energy / total_energy, 0.0, 1.0)
        return dir_proj, energy_ratio

    def _compute_nonconformity_scores(
        self, dir_proj: np.ndarray, energy_ratio: np.ndarray
    ) -> np.ndarray:
        """Combines normalized directional projection and subspace energy into [0, 1] score."""
        z_proj = (dir_proj - self._proj_shift) / max(self._proj_scale, 1e-8)
        z_clipped = np.clip(z_proj, -40.0, 40.0)
        prob_dir = 1.0 / (1.0 + np.exp(-z_clipped))
        combined = (1.0 - self.subspace_weight) * prob_dir + self.subspace_weight * energy_ratio
        return np.clip(combined, 0.0, 1.0)

    def fit(
        self,
        benign_activations: np.ndarray,
        harmful_activations: np.ndarray,
    ) -> MechanisticSafetyGuardrail:
        """Fits the harm direction, SVD jailbreak subspace, and conformal threshold.

        Args:
            benign_activations: 2D array (N_benign, D) of safe calibration activations.
            harmful_activations: 2D array (N_harm, D) of adversarial/harmful activations.

        Returns:
            Self for method chaining.
        """
        X_benign = np.asarray(benign_activations, dtype=np.float64)
        X_harm = np.asarray(harmful_activations, dtype=np.float64)

        if X_benign.ndim != 2 or X_harm.ndim != 2:
            raise ValueError("benign_activations and harmful_activations must be 2D arrays.")
        if X_benign.shape[0] < 2 or X_harm.shape[0] < 2:
            raise ValueError("At least 2 benign and 2 harmful samples are required.")
        if X_benign.shape[1] != X_harm.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: {X_benign.shape[1]} != {X_harm.shape[1]}."
            )

        d_dim = X_benign.shape[1]
        self._benign_mean = np.mean(X_benign, axis=0)
        harm_mean = np.mean(X_harm, axis=0)

        diff = harm_mean - self._benign_mean
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm < 1e-12:
            v_harm = np.zeros(d_dim, dtype=np.float64)
            v_harm[0] = 1.0
        else:
            v_harm = diff / diff_norm
        self._harm_direction = v_harm

        # Extract principal adversarial subspace via SVD on centered harmful activations
        centered_harm = X_harm - self._benign_mean[None, :]
        _, _, vt = np.linalg.svd(centered_harm, full_matrices=False)
        eff_rank = min(self.subspace_rank, vt.shape[0], d_dim)
        # Ensure primary harm direction is included in the span of the orthonormal basis
        raw_basis = np.column_stack([v_harm, vt[:eff_rank].T])
        q_basis, _ = np.linalg.qr(raw_basis)
        self._subspace_basis = q_basis[:, :eff_rank]

        # Calibrate logistic projection scaling from benign vs harmful distributions
        benign_proj = (X_benign - self._benign_mean[None, :]) @ self._harm_direction
        harm_proj = centered_harm @ self._harm_direction
        self._proj_shift = float(0.5 * (np.mean(benign_proj) + np.mean(harm_proj)))
        pooled_std = float(np.sqrt(0.5 * (np.var(benign_proj) + np.var(harm_proj))) + 1e-6)
        self._proj_scale = max(pooled_std, 1e-6)

        # Split conformal quantile over benign calibration non-conformity scores
        b_proj, b_energy = self._compute_raw_components(X_benign)
        benign_scores = self._compute_nonconformity_scores(b_proj, b_energy)

        n_b = X_benign.shape[0]
        q_level = min(1.0, np.ceil((n_b + 1) * (1.0 - self.false_positive_alpha)) / n_b)
        self._threshold = float(np.quantile(benign_scores, q_level))
        self._is_fitted = True
        return self

    def evaluate(self, activation: np.ndarray) -> SafetyEvaluationResult:
        """Evaluates a single activation vector (or first row of 2D array) for safety."""
        if not self._is_fitted:
            raise RuntimeError("MechanisticSafetyGuardrail must be fitted before evaluation.")
        arr = np.asarray(activation, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim != 2:
            raise ValueError("activation must be a 1D or 2D array.")
        return self.evaluate_batch(arr)[0]

    def evaluate_batch(self, activations: np.ndarray) -> list[SafetyEvaluationResult]:
        """Evaluates a batch of activation vectors of shape (N, D)."""
        if not self._is_fitted or self._benign_mean is None:
            raise RuntimeError("MechanisticSafetyGuardrail must be fitted before evaluation.")
        X = np.asarray(activations, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self._benign_mean.shape[0]:
            raise ValueError(
                f"Expected activation dimension {self._benign_mean.shape[0]}, got {X.shape[1]}."
            )

        dir_proj, energy_ratio = self._compute_raw_components(X)
        scores = self._compute_nonconformity_scores(dir_proj, energy_ratio)

        results: list[SafetyEvaluationResult] = []
        for i in range(X.shape[0]):
            sc = float(scores[i])
            results.append(
                SafetyEvaluationResult(
                    is_unsafe=bool(sc > self._threshold),
                    safety_score=sc,
                    direction_projection=float(dir_proj[i]),
                    subspace_energy_ratio=float(energy_ratio[i]),
                    threshold=self._threshold,
                    false_positive_alpha=self.false_positive_alpha,
                )
            )
        return results

    def ablate_harm_subspace(self, activation: np.ndarray) -> np.ndarray:
        """Projects out the learned harmful refusal direction and SVD jailbreak subspace."""
        if not self._is_fitted or self._subspace_basis is None:
            raise RuntimeError("MechanisticSafetyGuardrail must be fitted before ablation.")
        arr = np.asarray(activation, dtype=np.float64)
        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr[None, :]
        if arr.shape[1] != self._subspace_basis.shape[0]:
            raise ValueError(
                f"Expected activation dimension {self._subspace_basis.shape[0]}, "
                f"got {arr.shape[1]}."
            )

        # Orthogonal projection removal: h_clean = h - U_sub (U_sub^T h)
        proj = (arr @ self._subspace_basis) @ self._subspace_basis.T
        cleaned = (arr - proj).astype(np.float32)
        if was_1d:
            return np.asarray(cleaned[0], dtype=np.float32)
        return np.asarray(cleaned, dtype=np.float32)


__all__ = [
    "MechanisticSafetyGuardrail",
    "SafetyEvaluationResult",
]
