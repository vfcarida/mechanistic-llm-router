"""Mondrian Group-Conditional Conformal Risk Control.

Guarantees group-conditional finite-sample risk bounds across disjoint domains
or Sparse Autoencoder (SAE) circuit taxonomies:
1. Mondrian Taxonomy Partitioning (Vovk et al., 2005; Romano et al., NeurIPS 2020):
   Partitions calibration samples into disjoint groups g in G (such as benchmark
   domains or SAE `circuit_type` classifications) and enforces:
       E[L_{n+1}(lambda_g) | G_{n+1} = g] <= alpha   for every group g in G.
2. Conservative Empirical Bayes Shrinkage:
   Blends small-sample group thresholds with the pooled global threshold via
   min(lambda_g^raw, w_g * lambda_g^raw + (1 - w_g) * lambda_global) where
   w_g = n_g / (n_g + kappa), preserving finite-sample validity while stabilizing
   rare groups.
3. Group Risk Disparity Auditing:
   Quantifies the maximum empirical risk gap across groups (`max_group_risk_disparity`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conformal import (
    ConformalRiskController,
    compute_epistemic_uncertainty,
)


@dataclass(frozen=True)
class MondrianGroupCalibration:
    """Calibrated conformal thresholds and risk metrics for a single Mondrian group."""

    group: str
    num_samples: int
    low_threshold: float
    high_threshold: float
    empirical_risk_low: float
    empirical_risk_high: float
    upper_bound_low: float
    upper_bound_high: float


@dataclass(frozen=True)
class MondrianCalibrationSummary:
    """Complete summary across all Mondrian taxonomy groups."""

    alpha: float
    global_low_threshold: float
    global_high_threshold: float
    group_calibrations: dict[str, MondrianGroupCalibration]
    max_group_risk_disparity: float


class MondrianConformalController:
    """Group-conditional Conformal Risk Controller with hierarchical shrinkage."""

    def __init__(
        self,
        alpha: float = 0.10,
        loss_bound: float = 1.0,
        grid_size: int = 201,
        shrinkage_kappa: float = 4.0,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")
        if loss_bound <= 0.0:
            raise ValueError(f"loss_bound must be > 0, got {loss_bound}.")
        if grid_size < 11:
            raise ValueError(f"grid_size must be >= 11, got {grid_size}.")
        if shrinkage_kappa < 0.0:
            raise ValueError(f"shrinkage_kappa must be >= 0, got {shrinkage_kappa}.")

        self.alpha = float(alpha)
        self.loss_bound = float(loss_bound)
        self.grid_size = int(grid_size)
        self.shrinkage_kappa = float(shrinkage_kappa)

        self._global_controller = ConformalRiskController(
            alpha=self.alpha,
            max_loss_bound=self.loss_bound,
            grid_size=self.grid_size,
        )
        self._summary: MondrianCalibrationSummary | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether the Mondrian controller has been calibrated."""
        return self._summary is not None

    @property
    def summary(self) -> MondrianCalibrationSummary:
        """Calibrated Mondrian summary across all groups."""
        if self._summary is None:
            raise RuntimeError("MondrianConformalController has not been calibrated yet.")
        return self._summary

    def calibrate(
        self,
        probe_probs: list[float] | np.ndarray,
        cheap_losses: list[float] | np.ndarray,
        groups: list[str],
        mid_losses: list[float] | np.ndarray | None = None,
    ) -> MondrianCalibrationSummary:
        """Calibrates group-conditional thresholds (lambda_{1,g}, lambda_{2,g}) per group."""
        probs = np.asarray(probe_probs, dtype=np.float64).ravel()
        c_losses = np.asarray(cheap_losses, dtype=np.float64).ravel()
        if probs.shape[0] != c_losses.shape[0] or probs.shape[0] != len(groups):
            raise ValueError("probe_probs, cheap_losses, and groups must have identical length.")
        if probs.shape[0] < 2:
            raise ValueError("At least 2 calibration samples are required.")

        m_losses: np.ndarray | None = None
        if mid_losses is not None:
            m_losses = np.asarray(mid_losses, dtype=np.float64).ravel()
            if m_losses.shape[0] != probs.shape[0]:
                raise ValueError("mid_losses must match length of probe_probs.")

        # 1. Calibrate pooled global fallback controller
        global_sum = self._global_controller.calibrate(
            probe_probs=probs,
            cheap_losses=c_losses,
            mid_losses=m_losses,
        )
        g_low = global_sum.low_threshold
        g_high = global_sum.high_threshold

        # 2. Partition by Mondrian group and calibrate each partition
        unique_groups = sorted({str(g).strip() for g in groups if str(g).strip()})
        if not unique_groups:
            raise ValueError("groups list must contain at least one non-empty group name.")

        group_arr = np.array([str(g).strip() for g in groups])
        group_calibrations: dict[str, MondrianGroupCalibration] = {}
        empirical_risks: list[float] = []

        for grp in unique_groups:
            mask = group_arr == grp
            n_g = int(np.sum(mask))
            if n_g == 0:
                continue

            grp_ctrl = ConformalRiskController(
                alpha=self.alpha,
                max_loss_bound=self.loss_bound,
                grid_size=self.grid_size,
            )
            grp_sum = grp_ctrl.calibrate(
                probe_probs=probs[mask],
                cheap_losses=c_losses[mask],
                mid_losses=m_losses[mask] if m_losses is not None else None,
            )

            # Conservative Empirical Bayes shrinkage toward pooled global threshold
            w_g = float(n_g / (n_g + self.shrinkage_kappa))
            shrunk_low = min(
                grp_sum.low_threshold,
                w_g * grp_sum.low_threshold + (1.0 - w_g) * g_low,
            )
            shrunk_high = max(
                shrunk_low,
                min(
                    grp_sum.high_threshold,
                    w_g * grp_sum.high_threshold + (1.0 - w_g) * g_high,
                ),
            )

            group_calibrations[grp] = MondrianGroupCalibration(
                group=grp,
                num_samples=n_g,
                low_threshold=float(shrunk_low),
                high_threshold=float(shrunk_high),
                empirical_risk_low=grp_sum.empirical_risk_low,
                empirical_risk_high=grp_sum.empirical_risk_high,
                upper_bound_low=grp_sum.upper_risk_bound_low,
                upper_bound_high=grp_sum.upper_risk_bound_high,
            )
            empirical_risks.append(grp_sum.empirical_risk_low)

        disparity = (
            float(max(empirical_risks) - min(empirical_risks)) if len(empirical_risks) >= 2 else 0.0
        )

        self._summary = MondrianCalibrationSummary(
            alpha=self.alpha,
            global_low_threshold=g_low,
            global_high_threshold=g_high,
            group_calibrations=group_calibrations,
            max_group_risk_disparity=disparity,
        )
        return self._summary

    def get_thresholds(self, group: str | None = None) -> tuple[float, float]:
        """Returns (low_threshold, high_threshold) for `group`, or global fallback if unseen."""
        if self._summary is None:
            raise RuntimeError("MondrianConformalController has not been calibrated yet.")
        if group is not None and group in self._summary.group_calibrations:
            g_cal = self._summary.group_calibrations[group]
            return g_cal.low_threshold, g_cal.high_threshold
        return self._summary.global_low_threshold, self._summary.global_high_threshold

    def select_tier_index(
        self,
        prob_complex: float,
        group: str | None = None,
        num_tiers: int = 3,
        max_uncertainty: float | None = None,
    ) -> tuple[int, bool, float, float]:
        """Selects tier index in {0, ..., num_tiers - 1} using group-conditional thresholds."""
        low_t, high_t = self.get_thresholds(group)
        uncertainty = compute_epistemic_uncertainty(prob_complex)

        if num_tiers >= 3:
            if prob_complex < low_t:
                tier_idx = 0
            elif prob_complex < high_t:
                tier_idx = 1
            else:
                tier_idx = 2
            max_idx = 2
        else:
            tier_idx = 1 if prob_complex >= high_t else 0
            max_idx = 1

        abstained = False
        if max_uncertainty is not None and uncertainty >= max_uncertainty:
            if tier_idx < max_idx:
                tier_idx += 1
                abstained = True

        return tier_idx, abstained, low_t, high_t


__all__ = [
    "MondrianCalibrationSummary",
    "MondrianConformalController",
    "MondrianGroupCalibration",
]
