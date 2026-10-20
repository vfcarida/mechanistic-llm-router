"""Layer-Depth Adaptive Early-Exit Probe (CALM / Depth-Adaptive Probing).

Implements sequential layerwise probing with early-exit halting criteria:
1. Per-Layer Linear Probes (Alain & Bengio, ICLR 2017):
   Trains an independent calibrated LinearActivationProbe at each residual-stream
   depth l in {0, ..., L - 1}.
2. Depth-Adaptive Halting Criterion (Schuster et al., NeurIPS 2022 — CALM;
   Elbayad et al., ICLR 2020):
   Halts evaluation at the earliest layer l* where predictive Shannon entropy
   H(p^{(l*)}) <= tau_entropy and inter-layer angular stability cos(h^{(l*)}, h^{(l*-1)})
   and probability drift |p^{(l*)} - p^{(l*-1)}| satisfy consistency gates.
3. Conformal Disagreement Calibration:
   Calibrates tau_entropy on a validation split so that the empirical disagreement
   rate between early-exit predictions and the full-depth layer (L - 1) prediction
   is bounded by target_disagreement_alpha with finite-sample conformal padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conformal import compute_epistemic_uncertainty
from .linear_probe import LinearActivationProbe


@dataclass(frozen=True)
class EarlyExitDecision:
    """Telemetry and routing complexity decision from layerwise early-exit probing."""

    exit_layer_index: int
    total_layers: int
    halted_early: bool
    prob_complex: float
    epistemic_uncertainty: float
    compute_saved_ratio: float
    layer_probabilities: list[float]
    inter_layer_cosine_similarity: float


class LayerwiseEarlyExitProbe:
    """Depth-adaptive linear probe cascade that halts early on confident shallow layers."""

    def __init__(
        self,
        num_layers: int = 3,
        entropy_threshold: float = 0.35,
        min_cosine_stability: float = 0.80,
        max_prob_delta: float = 0.10,
        patience: int = 1,
        l2_reg: float = 1e-2,
        seed: int = 42,
    ) -> None:
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for early-exit probing.")
        if not (0.0 < entropy_threshold <= 1.0):
            raise ValueError("entropy_threshold must lie in (0, 1].")
        if not (-1.0 <= min_cosine_stability <= 1.0):
            raise ValueError("min_cosine_stability must lie in [-1, 1].")
        if not (0.0 <= max_prob_delta <= 1.0):
            raise ValueError("max_prob_delta must lie in [0, 1].")
        if patience < 1:
            raise ValueError("patience must be >= 1.")

        self.num_layers = int(num_layers)
        self.entropy_threshold = float(entropy_threshold)
        self.min_cosine_stability = float(min_cosine_stability)
        self.max_prob_delta = float(max_prob_delta)
        self.patience = int(patience)
        self.l2_reg = float(l2_reg)
        self.seed = int(seed)

        self.layer_probes: list[LinearActivationProbe] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether all layerwise linear probes have been fitted."""
        return self._is_fitted and len(self.layer_probes) == self.num_layers

    def _normalize_batch_layers(
        self, layer_activations: list[np.ndarray] | np.ndarray
    ) -> list[np.ndarray]:
        """Normalizes input into a list of L 2D arrays of shape (N, D_l)."""
        if isinstance(layer_activations, list):
            if len(layer_activations) != self.num_layers:
                raise ValueError(
                    f"Expected {self.num_layers} layer arrays, got {len(layer_activations)}."
                )
            out = [np.asarray(arr, dtype=np.float32) for arr in layer_activations]
            for idx, arr in enumerate(out):
                if arr.ndim == 1:
                    out[idx] = arr[None, :]
                elif arr.ndim != 2:
                    raise ValueError(f"Layer {idx} array must be 1D or 2D.")
            n_samples = out[0].shape[0]
            if any(a.shape[0] != n_samples for a in out):
                raise ValueError("All layer arrays must have the same number of samples N.")
            return out

        arr = np.asarray(layer_activations, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[1] != self.num_layers:
                raise ValueError(
                    f"3D array second dimension ({arr.shape[1]}) must equal "
                    f"num_layers ({self.num_layers})."
                )
            return [arr[:, i, :] for i in range(self.num_layers)]
        if arr.ndim == 2:
            if arr.shape[1] % self.num_layers != 0:
                raise ValueError(
                    f"Concatenated 2D feature dimension ({arr.shape[1]}) is not divisible by "
                    f"num_layers ({self.num_layers})."
                )
            return list(np.split(arr, self.num_layers, axis=1))
        if arr.ndim == 1:
            if arr.shape[0] % self.num_layers != 0:
                raise ValueError(
                    f"Concatenated 1D feature dimension ({arr.shape[0]}) is not divisible by "
                    f"num_layers ({self.num_layers})."
                )
            return [chunk[None, :] for chunk in np.split(arr, self.num_layers)]
        raise ValueError("Unsupported layer_activations shape.")

    @staticmethod
    def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors of matching dimension."""
        if u.shape != v.shape:
            return 1.0
        norm_u = float(np.linalg.norm(u))
        norm_v = float(np.linalg.norm(v))
        if norm_u < 1e-12 or norm_v < 1e-12:
            return 0.0
        return float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))

    def fit(
        self,
        layer_activations: list[np.ndarray] | np.ndarray,
        labels: list[int] | np.ndarray,
    ) -> LayerwiseEarlyExitProbe:
        """Fits an independent LinearActivationProbe at each layer depth l in {0, ..., L-1}."""
        layers = self._normalize_batch_layers(layer_activations)
        y = np.asarray(labels, dtype=int)
        if y.shape[0] != layers[0].shape[0]:
            raise ValueError(
                f"Label count ({y.shape[0]}) does not match sample count ({layers[0].shape[0]})."
            )

        self.layer_probes = []
        c_val = 1.0 / max(self.l2_reg, 1e-6)
        for l_idx in range(self.num_layers):
            probe = LinearActivationProbe(
                C=c_val,
                random_state=self.seed + l_idx,
            )
            probe.fit(layers[l_idx], y)
            self.layer_probes.append(probe)

        self._is_fitted = True
        return self

    def calibrate_halting_threshold(
        self,
        layer_activations: list[np.ndarray] | np.ndarray,
        target_disagreement_alpha: float = 0.05,
        grid_size: int = 25,
    ) -> float:
        """Calibrates entropy_threshold via finite-sample Conformal Risk Control.

        Finds the largest entropy_threshold tau in (0, 1] such that the conformalized
        disagreement rate between the early-exit binary classification and the full-depth
        (layer L - 1) classification is <= target_disagreement_alpha.
        """
        if not self.is_fitted:
            raise RuntimeError("LayerwiseEarlyExitProbe must be fitted before calibration.")
        if not (0.0 < target_disagreement_alpha < 1.0):
            raise ValueError("target_disagreement_alpha must lie strictly in (0, 1).")

        layers = self._normalize_batch_layers(layer_activations)
        n_samples = layers[0].shape[0]
        if n_samples < 2:
            raise ValueError("At least 2 validation samples are required for calibration.")

        # Precompute probabilities across all layers for the calibration set
        all_probs = np.column_stack(
            [
                self.layer_probes[l_idx].predict_proba(layers[l_idx])
                for l_idx in range(self.num_layers)
            ]
        )
        full_depth_preds = (all_probs[:, -1] >= 0.5).astype(int)

        candidate_taus = np.linspace(0.05, 0.90, grid_size)
        best_tau = float(candidate_taus[0])

        for tau in candidate_taus:
            self.entropy_threshold = float(tau)
            decisions = self.evaluate_batch(layers)
            early_preds = np.array(
                [1 if d.prob_complex >= 0.5 else 0 for d in decisions], dtype=int
            )
            disagreements = (early_preds != full_depth_preds).astype(np.float64)
            # Finite-sample conformal upper risk bound: (sum(L_i) + 1) / (n + 1)
            conformal_risk = float((np.sum(disagreements) + 1.0) / (n_samples + 1.0))
            if conformal_risk <= target_disagreement_alpha:
                best_tau = float(tau)

        self.entropy_threshold = best_tau
        return best_tau

    def evaluate(
        self, sample_layer_activations: list[np.ndarray] | np.ndarray
    ) -> EarlyExitDecision:
        """Evaluates a single sample sequentially across layers and halts as soon as confident."""
        if not self.is_fitted:
            raise RuntimeError("LayerwiseEarlyExitProbe must be fitted before evaluation.")

        layers = self._normalize_batch_layers(sample_layer_activations)
        layer_probs: list[float] = []
        stable_streak = 0
        last_cos = 1.0

        for l_idx in range(self.num_layers):
            h_curr = layers[l_idx][0]
            prob = float(self.layer_probes[l_idx].predict_proba(h_curr[None, :])[0])
            ent = compute_epistemic_uncertainty(prob)
            layer_probs.append(prob)

            if l_idx > 0:
                h_prev = layers[l_idx - 1][0]
                last_cos = self._cosine_similarity(h_curr, h_prev)
                prob_delta = abs(prob - layer_probs[l_idx - 1])
                same_class = (prob >= 0.5) == (layer_probs[l_idx - 1] >= 0.5)
                if (
                    ent <= self.entropy_threshold
                    and last_cos >= self.min_cosine_stability
                    and prob_delta <= self.max_prob_delta
                    and same_class
                ):
                    stable_streak += 1
                else:
                    stable_streak = 0
            else:
                # Layer 0 can halt immediately only when patience == 1 and ultra-confident
                if self.patience == 1 and ent <= 0.5 * self.entropy_threshold:
                    stable_streak = 1

            if l_idx < self.num_layers - 1 and stable_streak >= self.patience:
                saved_ratio = float((self.num_layers - 1 - l_idx) / max(1, self.num_layers))
                return EarlyExitDecision(
                    exit_layer_index=l_idx,
                    total_layers=self.num_layers,
                    halted_early=True,
                    prob_complex=prob,
                    epistemic_uncertainty=ent,
                    compute_saved_ratio=saved_ratio,
                    layer_probabilities=layer_probs,
                    inter_layer_cosine_similarity=last_cos,
                )

        final_prob = layer_probs[-1]
        final_ent = compute_epistemic_uncertainty(final_prob)
        return EarlyExitDecision(
            exit_layer_index=self.num_layers - 1,
            total_layers=self.num_layers,
            halted_early=False,
            prob_complex=final_prob,
            epistemic_uncertainty=final_ent,
            compute_saved_ratio=0.0,
            layer_probabilities=layer_probs,
            inter_layer_cosine_similarity=last_cos,
        )

    def evaluate_batch(
        self, layer_activations: list[np.ndarray] | np.ndarray
    ) -> list[EarlyExitDecision]:
        """Evaluates a batch of multi-layer activations."""
        layers = self._normalize_batch_layers(layer_activations)
        n_samples = layers[0].shape[0]
        decisions: list[EarlyExitDecision] = []
        for i in range(n_samples):
            sample_layers = [layers[l_idx][i : i + 1] for l_idx in range(self.num_layers)]
            decisions.append(self.evaluate(sample_layers))
        return decisions


__all__ = [
    "EarlyExitDecision",
    "LayerwiseEarlyExitProbe",
]
