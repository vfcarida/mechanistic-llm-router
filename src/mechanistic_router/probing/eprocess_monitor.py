"""Martingale E-Process & Anytime-Valid Sequential SLA Risk Monitor.

Provides time-uniform, non-asymptotic sequential hypothesis testing for routing
SLA compliance H_0: E[L_t | F_{t-1}] <= alpha under continuous monitoring (optional
stopping) without Type-I error inflation:
1. Betting Martingale Capital Process (Waudby-Smith & Ramdas, JRSS-B 2024):
   Constructs nonnegative test supermartingales
       E_t = prod_{i=1}^t (1 + lambda_i * (L_i - alpha))
   where lambda_i in [0, c / alpha] is F_{i-1}-measurable (predictable).
2. Approximate GRAPA / Kelly Predictable Betting Strategy:
   Adapts betting fractions online from past empirical excess loss and variance.
3. Ville's Maximal Inequality & Anytime-Valid P-Values (Ramdas et al., 2023):
   Guarantees P_{H_0}(exists t >= 1 : E_t >= 1 / delta) <= delta and maintains the
   running sequential p-value p_t = min_{0 <= s <= t} (1 / E_s).
4. Intersection E-Process & Conformal Circuit Breaker:
   Combines low-tier and high-tier E-processes via convex averaging
   E_t^{joint} = 0.5 * (E_t^{low} + E_t^{high}) and triggers an automatic SLA
   circuit breaker when E_t >= 1 / delta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EProcessStatus:
    """Snapshot of the anytime-valid sequential betting martingale SLA monitor."""

    alpha: float
    significance_delta: float
    rejection_threshold: float
    num_observations: int
    e_value_low: float
    e_value_high: float
    joint_e_value: float
    running_max_e_value: float
    anytime_p_value: float
    empirical_risk_low: float
    empirical_risk_high: float
    predictable_bet_low: float
    predictable_bet_high: float
    sla_violated: bool
    circuit_breaker_tripped: bool

    def to_dict(self) -> dict[str, Any]:
        """Serializes E-process monitor status to a JSON-compatible dictionary."""
        return asdict(self)


class BettingMartingaleMonitor:
    """Anytime-valid sequential SLA risk monitor using game-theoretic betting martingales."""

    def __init__(
        self,
        alpha: float = 0.10,
        significance_delta: float = 0.05,
        max_bet_multiplier: float = 0.75,
        min_exploration_bet: float = 0.05,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not (0.0 < significance_delta < 1.0):
            raise ValueError(f"significance_delta must be in (0, 1), got {significance_delta}.")
        if not (0.0 < max_bet_multiplier < 1.0):
            raise ValueError(
                "max_bet_multiplier must be in (0, 1) to ensure positivity, "
                f"got {max_bet_multiplier}."
            )
        if min_exploration_bet < 0.0:
            raise ValueError(f"min_exploration_bet must be >= 0, got {min_exploration_bet}.")

        self.alpha = float(alpha)
        self.significance_delta = float(significance_delta)
        self.max_bet_multiplier = float(max_bet_multiplier)
        self.min_exploration_bet = float(min_exploration_bet)

        self._losses_low: list[float] = []
        self._losses_high: list[float] = []
        self._e_low: float = 1.0
        self._e_high: float = 1.0
        self._running_max_joint: float = 1.0
        self._circuit_breaker_tripped: bool = False

    @property
    def rejection_threshold(self) -> float:
        """Ville's inequality rejection threshold 1 / delta."""
        return float(1.0 / self.significance_delta)

    def _compute_predictable_bet(self, history: list[float]) -> float:
        """Computes F_{t-1}-measurable aGRAPA betting fraction in [0, c / alpha]."""
        max_lambda = self.max_bet_multiplier / self.alpha
        if not history:
            return float(min(self.min_exploration_bet / self.alpha, max_lambda))

        arr = np.asarray(history, dtype=np.float64)
        n = float(arr.size)
        # Regularized plug-in mean and variance from prior history
        mu_hat = float((self.alpha + np.sum(arr)) / (1.0 + n))
        var_hat = float((self.alpha * (1.0 - self.alpha) + np.sum((arr - mu_hat) ** 2)) / (1.0 + n))
        excess = mu_hat - self.alpha
        if excess <= 0.0:
            # Conservative exploration bet when empirical risk is below alpha
            base_bet = self.min_exploration_bet / self.alpha
            return float(np.clip(base_bet * 0.25, 0.0, max_lambda))

        raw_lambda = excess / (var_hat + excess * excess + 1e-6)
        min_active = (
            (self.min_exploration_bet / self.alpha) if self.min_exploration_bet > 0 else 0.0
        )
        return float(np.clip(max(raw_lambda, min_active), 0.0, max_lambda))

    def update(
        self,
        incurred_loss_low: float,
        incurred_loss_high: float | None = None,
    ) -> EProcessStatus:
        """Updates the betting martingale with newly observed bounded routing losses in [0, 1]."""
        loss_low = float(np.clip(incurred_loss_low, 0.0, 1.0))
        loss_high = (
            float(np.clip(incurred_loss_high, 0.0, 1.0))
            if incurred_loss_high is not None
            else loss_low
        )

        # Predictable bets computed strictly prior to incorporating current observation
        bet_low = self._compute_predictable_bet(self._losses_low)
        bet_high = self._compute_predictable_bet(self._losses_high)

        mult_low = max(1e-6, 1.0 + bet_low * (loss_low - self.alpha))
        mult_high = max(1e-6, 1.0 + bet_high * (loss_high - self.alpha))

        self._e_low = float(min(1e12, self._e_low * mult_low))
        self._e_high = float(min(1e12, self._e_high * mult_high))

        self._losses_low.append(loss_low)
        self._losses_high.append(loss_high)

        joint_e = 0.5 * (self._e_low + self._e_high)
        max_component_e = max(self._e_low, self._e_high, joint_e)
        self._running_max_joint = max(self._running_max_joint, max_component_e)

        if self._running_max_joint >= self.rejection_threshold:
            self._circuit_breaker_tripped = True

        return self.status()

    def status(self) -> EProcessStatus:
        """Returns the current anytime-valid E-process telemetry snapshot."""
        n_obs = len(self._losses_low)
        emp_low = float(np.mean(self._losses_low)) if n_obs > 0 else 0.0
        emp_high = float(np.mean(self._losses_high)) if n_obs > 0 else 0.0
        bet_low = self._compute_predictable_bet(self._losses_low)
        bet_high = self._compute_predictable_bet(self._losses_high)

        joint_e = 0.5 * (self._e_low + self._e_high)
        anytime_p = float(np.clip(1.0 / max(1.0, self._running_max_joint), 0.0, 1.0))
        violated = bool(max(self._e_low, self._e_high, joint_e) >= self.rejection_threshold)

        return EProcessStatus(
            alpha=self.alpha,
            significance_delta=self.significance_delta,
            rejection_threshold=round(self.rejection_threshold, 6),
            num_observations=n_obs,
            e_value_low=round(self._e_low, 6),
            e_value_high=round(self._e_high, 6),
            joint_e_value=round(joint_e, 6),
            running_max_e_value=round(self._running_max_joint, 6),
            anytime_p_value=round(anytime_p, 6),
            empirical_risk_low=round(emp_low, 6),
            empirical_risk_high=round(emp_high, 6),
            predictable_bet_low=round(bet_low, 6),
            predictable_bet_high=round(bet_high, 6),
            sla_violated=violated,
            circuit_breaker_tripped=self._circuit_breaker_tripped,
        )

    def reset(self) -> EProcessStatus:
        """Resets the capital process and circuit breaker after recalibration."""
        self._losses_low.clear()
        self._losses_high.clear()
        self._e_low = 1.0
        self._e_high = 1.0
        self._running_max_joint = 1.0
        self._circuit_breaker_tripped = False
        return self.status()


__all__ = [
    "BettingMartingaleMonitor",
    "EProcessStatus",
]
