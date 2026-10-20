"""Counterfactual Activation Steering (CAA) & Sparse Autoencoder Feature Clamping.

Implements Contrastive Activation Addition (Rimsky et al., ACL 2024; Zou et al., 2023)
and error-preserving Sparse Autoencoder (SAE) latent feature clamping (Templeton et al.,
2024) to measure counterfactual routing sensitivity and steer residual representations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .linear_probe import LinearActivationProbe
from .sae_engine import SAEEngine


@dataclass(frozen=True)
class SteeringCounterfactualResult:
    """Result of a counterfactual activation steering or SAE clamping intervention."""

    original_prob: float
    steered_prob: float
    probability_delta: float
    min_flip_gamma: float
    decision_flipped: bool
    steered_activation: np.ndarray


class ActivationSteeringEngine:
    """Performs Contrastive Activation Addition (CAA) and SAE latent clamping."""

    def __init__(self, steering_direction: np.ndarray | None = None) -> None:
        self._direction: np.ndarray | None = None
        if steering_direction is not None:
            self._direction = self._normalize(steering_direction)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            raise ValueError("Steering vector cannot have zero norm.")
        return (arr / norm).astype(np.float32)

    @property
    def steering_direction(self) -> np.ndarray:
        """Unit-norm contrastive steering vector v_hat in R^D."""
        if self._direction is None:
            raise ValueError("No steering_direction has been fitted or configured.")
        return self._direction

    def fit_contrastive_direction(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray,
    ) -> np.ndarray:
        """Computes the unit difference-in-means vector (mu_pos - mu_neg) / ||mu_pos - mu_neg||."""
        pos = np.asarray(positive_activations, dtype=np.float64)
        neg = np.asarray(negative_activations, dtype=np.float64)
        if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
            raise ValueError(
                "positive_activations and negative_activations must be 2D arrays with matching D."
            )
        if pos.shape[0] < 1 or neg.shape[0] < 1:
            raise ValueError("Both positive and negative sets must be non-empty.")

        diff = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        self._direction = self._normalize(diff)
        return self._direction

    def steer_activation(
        self,
        activation: np.ndarray,
        gamma: float,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """Applies additive contrastive steering h_steered = h + gamma * v_hat."""
        v_hat = self._normalize(direction) if direction is not None else self.steering_direction
        h = np.asarray(activation, dtype=np.float32)
        if h.shape[-1] != v_hat.shape[0]:
            raise ValueError(
                f"Activation dim {h.shape[-1]} does not match steering vector {v_hat.shape[0]}."
            )
        return np.asarray(h + float(gamma) * v_hat, dtype=np.float32)

    def clamp_sae_features(
        self,
        activation: np.ndarray,
        sae_engine: SAEEngine,
        feature_clamps: dict[int, float],
    ) -> np.ndarray:
        """Clamps specified SAE latent features while preserving the SAE reconstruction error term.

        Follows the error-preserving intervention formulation:
            h_clamped = h + (decode(z_clamped) - decode(z_orig))
        so that unmodeled residual components e = h - h_hat(z_orig) are exactly preserved.
        """
        h_arr = np.asarray(activation, dtype=np.float32)
        is_1d = h_arr.ndim == 1
        h_tensor = torch.from_numpy(h_arr[None, :] if is_1d else h_arr).float()

        with torch.no_grad():
            recon_orig, z_orig = sae_engine(h_tensor)
            z_clamped = z_orig.clone()
            for feat_idx, clamp_val in feature_clamps.items():
                if not (0 <= int(feat_idx) < sae_engine.d_sae):
                    raise ValueError(
                        f"Feature index {feat_idx} out of bounds for d_sae={sae_engine.d_sae}."
                    )
                z_clamped[..., int(feat_idx)] = float(clamp_val)

            recon_clamped = sae_engine.decode(z_clamped)
            h_clamped = h_tensor + (recon_clamped - recon_orig)

        out = h_clamped.cpu().numpy().astype(np.float32)
        return np.asarray(out[0], dtype=np.float32) if is_1d else out

    def evaluate_counterfactual_sensitivity(
        self,
        activation: np.ndarray,
        probe: LinearActivationProbe,
        gamma: float,
        target_threshold: float = 0.5,
        direction: np.ndarray | None = None,
    ) -> SteeringCounterfactualResult:
        """Evaluates routing probability shift under steering and exact minimum flip gamma*."""
        if not (0.0 < target_threshold < 1.0):
            raise ValueError(f"target_threshold must be in (0, 1), got {target_threshold}.")

        h = np.asarray(activation, dtype=np.float32).reshape(-1)
        v_hat = (
            self._normalize(direction)
            if direction is not None
            else (
                self._direction
                if self._direction is not None
                else np.asarray(probe.direction_vector, dtype=np.float32)
            )
        )

        orig_prob = float(probe.predict_proba(h[None, :])[0])
        h_steered = self.steer_activation(h, gamma=gamma, direction=v_hat)
        steered_prob = float(probe.predict_proba(h_steered[None, :])[0])

        w = np.asarray(probe.weights, dtype=np.float64)
        b = float(probe.intercept)
        margin_orig = float(np.dot(w, h.astype(np.float64)) + b)
        target_logit = math.log(target_threshold / (1.0 - target_threshold))
        proj_wv = float(np.dot(w, v_hat.astype(np.float64)))

        if abs(proj_wv) > 1e-12:
            min_flip_gamma = (target_logit - margin_orig) / proj_wv
        else:
            min_flip_gamma = float("inf")

        orig_decision = orig_prob >= target_threshold
        steered_decision = steered_prob >= target_threshold

        return SteeringCounterfactualResult(
            original_prob=orig_prob,
            steered_prob=steered_prob,
            probability_delta=steered_prob - orig_prob,
            min_flip_gamma=float(min_flip_gamma),
            decision_flipped=bool(orig_decision != steered_decision),
            steered_activation=h_steered,
        )


__all__ = ["ActivationSteeringEngine", "SteeringCounterfactualResult"]
