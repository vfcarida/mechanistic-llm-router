"""Distribution-Free Conformal Risk Control for Calibrated Multi-Tier Routing.

Implements split conformal risk control (Angelopoulos et al., ICLR 2024, arXiv:2208.02814)
and calibrated epistemic uncertainty quantification (Quach et al., ICLR 2024, arXiv:2306.10193)
for residual-stream activation probes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConformalCalibrationSummary:
    """Summary of finite-sample conformal risk calibration."""

    alpha: float
    low_threshold: float
    high_threshold: float
    empirical_risk_low: float
    upper_risk_bound_low: float
    empirical_risk_high: float
    upper_risk_bound_high: float
    calibration_size: int


def compute_epistemic_uncertainty(prob: float) -> float:
    """Computes normalized binary Shannon entropy H_2(p) in [0.0, 1.0].

    Quantifies epistemic uncertainty of a binary decision boundary probability:
    - p -> 0.0 or p -> 1.0 yields uncertainty -> 0.0 (high confidence).
    - p = 0.5 yields uncertainty = 1.0 (maximum boundary ambiguity).

    Args:
        prob: Predicted probability in [0.0, 1.0].

    Returns:
        Normalized Shannon entropy in [0.0, 1.0].
    """
    p = float(np.clip(prob, 1e-12, 1.0 - 1e-12))
    entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)
    return float(np.clip(entropy, 0.0, 1.0))


class ConformalRiskController:
    """Distribution-free Conformal Risk Controller for multi-tier LLM routing.

    Given calibration query probabilities p_i in [0, 1] (where higher p_i indicates
    higher task complexity) and bounded monotone quality losses l_i(m) in [0, B]
    relative to the frontier oracle, calibrates threshold(s) lambda_hat(alpha) such that:

        E[Loss(Y, M_lambda_hat(X))] <= alpha

    via the finite-sample Conformal Risk Control bound (Angelopoulos et al., 2024):

        (n / (n + 1)) * R_hat_n(lambda) + B / (n + 1) <= alpha

    Also supports Adaptive Conformal Inference (ACI; Gibbs & Candes, NeurIPS 2021)
    to dynamically adjust routing thresholds online under non-stationary distribution shift.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        max_loss_bound: float = 1.0,
        grid_size: int = 201,
        learning_rate: float = 0.05,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Conformal risk level alpha must be in (0, 1), got {alpha}.")
        if max_loss_bound <= 0.0:
            raise ValueError(f"max_loss_bound B must be > 0, got {max_loss_bound}.")
        if grid_size < 11:
            raise ValueError(f"grid_size must be >= 11, got {grid_size}.")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}.")

        self.alpha = float(alpha)
        self.max_loss_bound = float(max_loss_bound)
        self.grid_size = int(grid_size)
        self.learning_rate = float(learning_rate)

        self._low_threshold: float | None = None
        self._high_threshold: float | None = None
        self._summary: ConformalCalibrationSummary | None = None

        # Online adaptive state (Gibbs & Candes ACI)
        self._effective_alpha: float = float(alpha)
        self._online_steps: int = 0
        self._cumulative_loss_low: float = 0.0
        self._cumulative_loss_high: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        """Whether the controller has been calibrated on a validation split."""
        return self._low_threshold is not None and self._high_threshold is not None

    @property
    def low_threshold(self) -> float:
        """Calibrated probability threshold below which the cheapest tier is risk-safe."""
        if self._low_threshold is None:
            raise ValueError("ConformalRiskController has not been calibrated yet.")
        return self._low_threshold

    @property
    def high_threshold(self) -> float:
        """Calibrated probability threshold above which the frontier oracle is required."""
        if self._high_threshold is None:
            raise ValueError("ConformalRiskController has not been calibrated yet.")
        return self._high_threshold

    @property
    def summary(self) -> ConformalCalibrationSummary:
        """Detailed calibration summary statistics."""
        if self._summary is None:
            raise ValueError("ConformalRiskController has not been calibrated yet.")
        return self._summary

    @property
    def effective_alpha(self) -> float:
        """Current online adaptive conformal risk level alpha_t."""
        return self._effective_alpha

    @property
    def online_steps(self) -> int:
        """Total number of online feedback updates processed."""
        return self._online_steps

    @property
    def online_empirical_risk(self) -> tuple[float, float]:
        """Running empirical risk (risk_low, risk_high) across online feedback updates."""
        if self._online_steps == 0:
            if self._summary is not None:
                return (self._summary.empirical_risk_low, self._summary.empirical_risk_high)
            return (0.0, 0.0)
        return (
            self._cumulative_loss_low / self._online_steps,
            self._cumulative_loss_high / self._online_steps,
        )

    def _calibrate_single_threshold(
        self,
        probs: np.ndarray,
        losses_if_delegated: np.ndarray,
        target_alpha: float,
    ) -> tuple[float, float, float]:
        """Finds supremum threshold lambda in [0, 1] satisfying finite-sample risk control.

        When p_i < lambda, the query is delegated to the cheaper model incurring
        loss `losses_if_delegated[i]`; when p_i >= lambda, the query escalates
        incurring 0.0 delegation loss.
        """
        n = len(probs)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # If B / (n + 1) > target_alpha, even zero empirical risk cannot satisfy the bound,
        # so lambda = 0.0 (always escalate) is the unique safe threshold.
        if self.max_loss_bound / (n + 1.0) > target_alpha:
            return 0.0, 0.0, self.max_loss_bound / (n + 1.0)

        grid = np.linspace(0.0, 1.0, self.grid_size)
        best_lambda = 0.0
        best_emp_risk = 0.0
        best_bound = self.max_loss_bound / (n + 1.0)

        for lam in grid:
            delegated_mask = probs < lam
            incurred_losses = np.where(delegated_mask, losses_if_delegated, 0.0)
            emp_risk = float(np.mean(incurred_losses))
            upper_bound = (n / (n + 1.0)) * emp_risk + (self.max_loss_bound / (n + 1.0))

            if upper_bound <= target_alpha:
                best_lambda = float(lam)
                best_emp_risk = emp_risk
                best_bound = float(upper_bound)
            else:
                # Since delegation loss is non-decreasing in lambda, stop at first violation
                break

        return best_lambda, best_emp_risk, best_bound

    def calibrate(
        self,
        probe_probs: list[float] | np.ndarray,
        cheap_losses: list[float] | np.ndarray,
        mid_losses: list[float] | np.ndarray | None = None,
    ) -> ConformalCalibrationSummary:
        """Calibrates binary or 3-tier routing thresholds under finite-sample risk control.

        Args:
            probe_probs: Predicted complexity probabilities p_i in [0, 1] on calibration set.
            cheap_losses: Non-negative quality loss in [0, B] when routing to cheap tier
                relative to the frontier oracle (e.g. max(0, acc_oracle - acc_cheap)).
            mid_losses: Optional non-negative quality loss in [0, B] when routing to mid tier
                relative to the frontier oracle. If None, binary routing is calibrated
                (low_threshold == high_threshold).

        Returns:
            ConformalCalibrationSummary with calibrated thresholds and risk bounds.
        """
        probs_arr = np.asarray(probe_probs, dtype=np.float64)
        cheap_arr = np.clip(np.asarray(cheap_losses, dtype=np.float64), 0.0, self.max_loss_bound)

        if probs_arr.ndim != 1 or len(probs_arr) == 0:
            raise ValueError("probe_probs must be a non-empty 1D array.")
        if cheap_arr.shape != probs_arr.shape:
            raise ValueError("cheap_losses must have the same shape as probe_probs.")

        low_lam, emp_low, bound_low = self._calibrate_single_threshold(
            probs=probs_arr,
            losses_if_delegated=cheap_arr,
            target_alpha=self.alpha,
        )

        if mid_losses is not None:
            mid_arr = np.clip(np.asarray(mid_losses, dtype=np.float64), 0.0, self.max_loss_bound)
            if mid_arr.shape != probs_arr.shape:
                raise ValueError("mid_losses must have the same shape as probe_probs.")
            high_lam, emp_high, bound_high = self._calibrate_single_threshold(
                probs=probs_arr,
                losses_if_delegated=mid_arr,
                target_alpha=self.alpha,
            )
            # Enforce monotonicity: high_threshold >= low_threshold
            if high_lam < low_lam:
                high_lam = low_lam
        else:
            high_lam, emp_high, bound_high = low_lam, emp_low, bound_low

        self._low_threshold = low_lam
        self._high_threshold = high_lam
        self._effective_alpha = self.alpha
        self._online_steps = 0
        self._cumulative_loss_low = 0.0
        self._cumulative_loss_high = 0.0
        self._summary = ConformalCalibrationSummary(
            alpha=self.alpha,
            low_threshold=low_lam,
            high_threshold=high_lam,
            empirical_risk_low=emp_low,
            upper_risk_bound_low=bound_low,
            empirical_risk_high=emp_high,
            upper_risk_bound_high=bound_high,
            calibration_size=len(probs_arr),
        )
        return self._summary

    def calibrate_multi_objective(
        self,
        probe_probs: list[float] | np.ndarray,
        quality_losses_cheap: list[float] | np.ndarray,
        sla_losses_cheap: list[float] | np.ndarray,
        alpha_quality: float | None = None,
        alpha_sla: float = 0.10,
        quality_losses_mid: list[float] | np.ndarray | None = None,
        sla_losses_mid: list[float] | np.ndarray | None = None,
    ) -> ConformalCalibrationSummary:
        """Calibrates joint multi-objective thresholds satisfying both quality and SLA bounds.

        By monotonicity of delegation risk in lambda, taking the intersection threshold
        lambda_joint = min(lambda_quality(alpha_quality), lambda_sla(alpha_sla))
        guarantees both finite-sample conformal risk bounds simultaneously:
            E[L_quality(lambda_joint)] <= alpha_quality  AND  E[L_sla(lambda_joint)] <= alpha_sla.
        """
        target_alpha_q = float(alpha_quality) if alpha_quality is not None else self.alpha
        target_alpha_s = float(alpha_sla)
        if not (0.0 < target_alpha_q < 1.0) or not (0.0 < target_alpha_s < 1.0):
            raise ValueError("alpha_quality and alpha_sla must both lie in (0, 1).")

        probs_arr = np.asarray(probe_probs, dtype=np.float64)
        q_cheap = np.clip(
            np.asarray(quality_losses_cheap, dtype=np.float64), 0.0, self.max_loss_bound
        )
        s_cheap = np.clip(np.asarray(sla_losses_cheap, dtype=np.float64), 0.0, self.max_loss_bound)
        if probs_arr.ndim != 1 or len(probs_arr) == 0:
            raise ValueError("probe_probs must be a non-empty 1D array.")
        if q_cheap.shape != probs_arr.shape or s_cheap.shape != probs_arr.shape:
            raise ValueError("Loss arrays must have the same shape as probe_probs.")

        lam_q_low, emp_q_low, bound_q_low = self._calibrate_single_threshold(
            probs=probs_arr, losses_if_delegated=q_cheap, target_alpha=target_alpha_q
        )
        lam_s_low, emp_s_low, bound_s_low = self._calibrate_single_threshold(
            probs=probs_arr, losses_if_delegated=s_cheap, target_alpha=target_alpha_s
        )
        low_lam = min(lam_q_low, lam_s_low)
        emp_low = emp_q_low if lam_q_low <= lam_s_low else emp_s_low
        bound_low = bound_q_low if lam_q_low <= lam_s_low else bound_s_low

        if quality_losses_mid is not None and sla_losses_mid is not None:
            q_mid = np.clip(
                np.asarray(quality_losses_mid, dtype=np.float64), 0.0, self.max_loss_bound
            )
            s_mid = np.clip(np.asarray(sla_losses_mid, dtype=np.float64), 0.0, self.max_loss_bound)
            lam_q_high, emp_q_high, bound_q_high = self._calibrate_single_threshold(
                probs=probs_arr, losses_if_delegated=q_mid, target_alpha=target_alpha_q
            )
            lam_s_high, emp_s_high, bound_s_high = self._calibrate_single_threshold(
                probs=probs_arr, losses_if_delegated=s_mid, target_alpha=target_alpha_s
            )
            high_lam = max(low_lam, min(lam_q_high, lam_s_high))
            emp_high = emp_q_high if lam_q_high <= lam_s_high else emp_s_high
            bound_high = bound_q_high if lam_q_high <= lam_s_high else bound_s_high
        else:
            high_lam, emp_high, bound_high = low_lam, emp_low, bound_low

        self.alpha = min(target_alpha_q, target_alpha_s)
        self._low_threshold = low_lam
        self._high_threshold = high_lam
        self._effective_alpha = self.alpha
        self._online_steps = 0
        self._cumulative_loss_low = 0.0
        self._cumulative_loss_high = 0.0
        self._summary = ConformalCalibrationSummary(
            alpha=self.alpha,
            low_threshold=low_lam,
            high_threshold=high_lam,
            empirical_risk_low=emp_low,
            upper_risk_bound_low=bound_low,
            empirical_risk_high=emp_high,
            upper_risk_bound_high=bound_high,
            calibration_size=len(probs_arr),
        )
        return self._summary

    def update_online(
        self,
        incurred_loss_low: float,
        incurred_loss_high: float | None = None,
        step_size: float | None = None,
    ) -> tuple[float, float]:
        """Updates routing thresholds online via Adaptive Conformal Inference (Gibbs & Candes).

        When incurred quality loss exceeds target risk budget `alpha`, tightens (lowers)
        the delegation threshold so fewer complex queries are routed to cheaper tiers.
        When incurred loss is below `alpha`, gradually relaxes (raises) the threshold.

        Args:
            incurred_loss_low: Observed quality loss in [0, B] for low-tier delegation.
            incurred_loss_high: Optional observed quality loss in [0, B] for mid-tier delegation.
                Defaults to `incurred_loss_low` if not provided.
            step_size: Optional override for online step size eta in (0, 1].

        Returns:
            Tuple of updated (low_threshold, high_threshold).
        """
        if not self.is_calibrated or self._low_threshold is None or self._high_threshold is None:
            raise ValueError("ConformalRiskController must be calibrated before online updates.")

        eta = float(step_size) if step_size is not None else self.learning_rate
        if not (0.0 < eta <= 1.0):
            raise ValueError(f"step_size must be in (0, 1], got {eta}.")

        loss_low = float(np.clip(incurred_loss_low, 0.0, self.max_loss_bound))
        loss_high = float(
            np.clip(
                incurred_loss_high if incurred_loss_high is not None else incurred_loss_low,
                0.0,
                self.max_loss_bound,
            )
        )

        # Normalize error signal relative to max_loss_bound
        grad_low = (loss_low - self.alpha) / self.max_loss_bound
        grad_high = (loss_high - self.alpha) / self.max_loss_bound

        new_low = float(np.clip(self._low_threshold - eta * grad_low, 0.0, 1.0))
        new_high = float(np.clip(self._high_threshold - eta * grad_high, new_low, 1.0))

        self._low_threshold = new_low
        self._high_threshold = new_high
        self._effective_alpha = float(
            np.clip(self._effective_alpha - eta * grad_low, 1e-4, 1.0 - 1e-4)
        )
        self._online_steps += 1
        self._cumulative_loss_low += loss_low
        self._cumulative_loss_high += loss_high

        return self._low_threshold, self._high_threshold

    def select_tier_index(
        self,
        prob_complex: float,
        num_tiers: int = 3,
        max_uncertainty: float | None = None,
    ) -> tuple[int, bool]:
        """Selects target tier index (0=cheap, 1=mid, 2=strong) and flags uncertainty abstention.

        Args:
            prob_complex: Probe probability that query requires strong model.
            num_tiers: Total number of ordered model tiers (2 or 3).
            max_uncertainty: Optional maximum allowed epistemic uncertainty H_2(p). If
                exceeded, escalates one tier (up to highest tier) and sets abstained=True.

        Returns:
            Tuple of (selected_tier_index, abstention_escalated).
        """
        low_t = self.low_threshold
        high_t = self.high_threshold

        if num_tiers <= 1:
            return 0, False

        if num_tiers == 2:
            base_idx = 0 if prob_complex < low_t else 1
        else:
            if prob_complex < low_t:
                base_idx = 0
            elif prob_complex < high_t:
                base_idx = 1
            else:
                base_idx = num_tiers - 1

        abstained = False
        if max_uncertainty is not None:
            uncertainty = compute_epistemic_uncertainty(prob_complex)
            if uncertainty >= max_uncertainty and base_idx < (num_tiers - 1):
                base_idx = min(base_idx + 1, num_tiers - 1)
                abstained = True

        return base_idx, abstained

    def save(self, filepath: str | Path) -> None:
        """Persists calibrated conformal controller state securely to a .npz archive."""
        if not self.is_calibrated or self._summary is None:
            raise ValueError("Cannot save an uncalibrated ConformalRiskController.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            params=np.array(
                [
                    self.alpha,
                    self.max_loss_bound,
                    float(self.grid_size),
                    self.low_threshold,
                    self.high_threshold,
                    self._summary.empirical_risk_low,
                    self._summary.upper_risk_bound_low,
                    self._summary.empirical_risk_high,
                    self._summary.upper_risk_bound_high,
                    float(self._summary.calibration_size),
                    self.learning_rate,
                    self._effective_alpha,
                    float(self._online_steps),
                    self._cumulative_loss_low,
                    self._cumulative_loss_high,
                ],
                dtype=np.float64,
            ),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> ConformalRiskController:
        """Loads a calibrated ConformalRiskController from a compressed .npz archive."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Conformal calibration archive not found at: {path}")

        with np.load(str(path)) as data:
            p = data["params"]
            alpha = float(p[0])
            max_loss_bound = float(p[1])
            grid_size = int(p[2])
            low_t = float(p[3])
            high_t = float(p[4])
            emp_low = float(p[5])
            bound_low = float(p[6])
            emp_high = float(p[7])
            bound_high = float(p[8])
            cal_size = int(p[9])
            learning_rate = float(p[10]) if len(p) > 10 else 0.05
            effective_alpha = float(p[11]) if len(p) > 11 else alpha
            online_steps = int(p[12]) if len(p) > 12 else 0
            cum_low = float(p[13]) if len(p) > 13 else 0.0
            cum_high = float(p[14]) if len(p) > 14 else 0.0

        controller = cls(
            alpha=alpha,
            max_loss_bound=max_loss_bound,
            grid_size=grid_size,
            learning_rate=learning_rate,
        )
        controller._low_threshold = low_t
        controller._high_threshold = high_t
        controller._effective_alpha = effective_alpha
        controller._online_steps = online_steps
        controller._cumulative_loss_low = cum_low
        controller._cumulative_loss_high = cum_high
        controller._summary = ConformalCalibrationSummary(
            alpha=alpha,
            low_threshold=low_t,
            high_threshold=high_t,
            empirical_risk_low=emp_low,
            upper_risk_bound_low=bound_low,
            empirical_risk_high=emp_high,
            upper_risk_bound_high=bound_high,
            calibration_size=cal_size,
        )
        return controller


__all__ = [
    "ConformalCalibrationSummary",
    "ConformalRiskController",
    "compute_epistemic_uncertainty",
]
