"""Multi-Turn Conversation Trajectory Probing & Context Saturation Detection.

Analyzes the temporal evolution of residual-stream activations [h_1, h_2, ..., h_T]
across multi-turn dialogues and agentic tool-use trajectories. Quantifies directional
cosine drift, residual norm growth, and spectral Effective Dimensionality (d_eff)
over the turn trajectory matrix to detect mid-session reasoning escalation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..signals.math_utils import compute_effective_dimensionality


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Diagnostic metrics characterizing a multi-turn residual-stream trajectory."""

    num_turns: int
    mean_cosine_drift: float
    max_cosine_drift: float
    norm_growth_ratio: float
    trajectory_d_eff: float
    saturation_escalated: bool
    pooled_activation: np.ndarray


class ConversationTrajectoryAnalyzer:
    """Evaluates multi-turn residual activation trajectories for reasoning drift and saturation."""

    def __init__(
        self,
        extractor: Any | None = None,
        drift_threshold: float = 0.45,
        d_eff_threshold: float = 1.65,
        recency_decay: float = 0.65,
    ) -> None:
        if not (0.0 < drift_threshold <= 2.0):
            raise ValueError(f"drift_threshold must be in (0, 2], got {drift_threshold}.")
        if d_eff_threshold < 1.0:
            raise ValueError(f"d_eff_threshold must be >= 1.0, got {d_eff_threshold}.")
        if not (0.0 < recency_decay <= 1.0):
            raise ValueError(f"recency_decay must be in (0, 1], got {recency_decay}.")

        if extractor is None:
            from ..evaluation.calibration_pack import DeterministicMultiLayerExtractor

            self.extractor: Any = DeterministicMultiLayerExtractor(
                d_layer=32,
                layer_indices=[-9, -5, -1],
                layer_fusion="concat",
            )
        else:
            self.extractor = extractor

        self.drift_threshold = float(drift_threshold)
        self.d_eff_threshold = float(d_eff_threshold)
        self.recency_decay = float(recency_decay)

    def analyze_turns(self, turns: Sequence[str] | np.ndarray) -> TrajectoryMetrics:
        """Computes trajectory drift, spectral dimensionality, and recency-pooled activation."""
        if isinstance(turns, np.ndarray):
            if turns.ndim != 2 or turns.shape[0] < 1:
                raise ValueError("Activation array `turns` must have shape (T, D) with T >= 1.")
            h_mat = np.asarray(turns, dtype=np.float64)
        else:
            turn_list = [str(t) for t in turns if str(t).strip()]
            if not turn_list:
                raise ValueError("`turns` must contain at least one non-empty turn string.")
            h_mat = np.asarray(self.extractor.extract_batch(turn_list), dtype=np.float64)

        n_turns, _ = h_mat.shape
        norms = np.linalg.norm(h_mat, axis=1)

        if n_turns == 1:
            return TrajectoryMetrics(
                num_turns=1,
                mean_cosine_drift=0.0,
                max_cosine_drift=0.0,
                norm_growth_ratio=1.0,
                trajectory_d_eff=1.0,
                saturation_escalated=False,
                pooled_activation=h_mat[0].astype(np.float32),
            )

        drifts: list[float] = []
        for idx in range(1, n_turns):
            denom = max(float(norms[idx] * norms[idx - 1]), 1e-12)
            cos_sim = float(np.clip(np.dot(h_mat[idx], h_mat[idx - 1]) / denom, -1.0, 1.0))
            drifts.append(1.0 - cos_sim)

        mean_drift = float(np.mean(drifts))
        max_drift = float(np.max(drifts))
        norm_growth = float(norms[-1] / max(float(norms[0]), 1e-8))

        # Spectral effective dimensionality across the (1, T, D) turn trajectory matrix
        tensor_3d = torch.from_numpy(h_mat[None, :, :]).float()
        traj_d_eff = float(compute_effective_dimensionality(tensor_3d))

        # Recency-weighted pooling: w_t proportional to recency_decay^(T - 1 - t)
        exponents = np.arange(n_turns - 1, -1, -1, dtype=np.float64)
        weights = np.power(self.recency_decay, exponents)
        weights = weights / np.sum(weights)
        pooled = np.sum(h_mat * weights[:, None], axis=0).astype(np.float32)

        escalated = bool(max_drift >= self.drift_threshold or traj_d_eff >= self.d_eff_threshold)

        return TrajectoryMetrics(
            num_turns=n_turns,
            mean_cosine_drift=mean_drift,
            max_cosine_drift=max_drift,
            norm_growth_ratio=norm_growth,
            trajectory_d_eff=traj_d_eff,
            saturation_escalated=escalated,
            pooled_activation=pooled,
        )


__all__ = ["ConversationTrajectoryAnalyzer", "TrajectoryMetrics"]
