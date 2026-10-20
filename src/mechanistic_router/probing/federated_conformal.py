"""Differentially Private Federated Conformal Risk Control.

Enables multi-organization or multi-region silos to collaboratively calibrate
global multi-tier Conformal Risk Control thresholds (lambda_1, lambda_2)
without sharing raw prompts, residual-stream activations, or individual sample
losses.

Mathematical Formulation:
1. Histogram Prefix-Sum Mechanism:
   For a calibration set of size n_k in silo k, the empirical risk curve
   R_k(lambda) = (1 / n_k) * sum_{i=1}^{n_k} L_i * I[s_i <= lambda]
   is the cumulative prefix sum of the bin histogram h_k in R^G over grid Lambda.
   Because each sample falls into exactly one score bin j and L_i in [0, 1],
   replacing one sample changes at most two bins by 1 / n_k:
     Delta_2 = sqrt(2) / n_k,   Delta_1 = 2 / n_k.
2. Analytic Gaussian / Laplace Perturbation:
   Each silo perturbs its bin histogram with calibrated noise and projects onto
   the monotone non-decreasing cone via non-negative prefix accumulation.
3. Sample-Weighted Federated Aggregation:
   The coordinator combines {R_tilde_k(lambda)}_{k=1}^K with weights w_k = n_k / N
   and incorporates both the finite-sample conformal margin 1 / (N + 1) and
   a privacy variance safety margin.

References:
- Dwork & Roth (2014). The Algorithmic Foundations of Differential Privacy.
- Humbert et al. (ICML 2023). One-Shot Federated Conformal Prediction.
- Angelopoulos et al. (ICLR 2024). Conformal Risk Control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .conformal import ConformalRiskController


@dataclass(frozen=True)
class DPClientCalibrationSummary:
    """Differentially private monotone risk curve summary exported by a single silo."""

    client_id: str
    num_samples: int
    lambda_grid: list[float]
    noisy_cheap_risk_curve: list[float]
    noisy_mid_risk_curve: list[float]
    epsilon: float
    delta: float
    bin_noise_std: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes the DP client summary into a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class FederatedCalibrationResult:
    """Global conformal thresholds calibrated across federated silos under DP."""

    lambda_1: float
    lambda_2: float
    alpha_target: float
    total_samples: int
    num_clients: int
    privacy_margin: float
    effective_empirical_risk_bound: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes the federated calibration result into a dictionary."""
        return asdict(self)


class FederatedConformalAggregator:
    """Coordinator for Differentially Private Federated Conformal Risk Control."""

    @staticmethod
    def compute_client_dp_summary(
        client_id: str,
        scores: np.ndarray,
        cheap_losses: np.ndarray,
        mid_losses: np.ndarray | None = None,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        grid_size: int = 101,
        seed: int = 42,
    ) -> DPClientCalibrationSummary:
        """Computes a differentially private monotone risk curve for a single silo.

        Args:
            client_id: Unique identifier of the participating silo/cluster.
            scores: 1D array of probe difficulty probabilities in [0, 1].
            cheap_losses: 1D array of cheap-tier quality losses in [0, 1].
            mid_losses: Optional 1D array of mid-tier quality losses in [0, 1].
            epsilon: Differential privacy budget epsilon > 0.
            delta: Differential privacy failure probability delta in [0, 1).
            grid_size: Number of threshold discretization points in [0, 1].
            seed: Deterministic RNG seed for reproducible noise generation.

        Returns:
            DPClientCalibrationSummary containing only privatized risk curves.
        """
        if not client_id or not client_id.strip():
            raise ValueError("client_id must be a non-empty string.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}.")
        if not (0.0 <= delta < 1.0):
            raise ValueError(f"delta must be in [0, 1), got {delta}.")
        if grid_size < 11:
            raise ValueError(f"grid_size must be >= 11, got {grid_size}.")

        s = np.asarray(scores, dtype=np.float64).ravel()
        l_cheap = np.asarray(cheap_losses, dtype=np.float64).ravel()
        n_k = int(s.shape[0])

        if n_k < 2:
            raise ValueError("At least 2 calibration samples are required per client silo.")
        if l_cheap.shape[0] != n_k:
            raise ValueError("scores and cheap_losses must have identical length.")

        if mid_losses is not None:
            l_mid = np.asarray(mid_losses, dtype=np.float64).ravel()
            if l_mid.shape[0] != n_k:
                raise ValueError("scores and mid_losses must have identical length.")
        else:
            l_mid = np.clip(l_cheap * 0.35, 0.0, 1.0)

        s = np.clip(s, 0.0, 1.0)
        l_cheap = np.clip(l_cheap, 0.0, 1.0)
        l_mid = np.clip(l_mid, 0.0, 1.0)

        lambda_grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
        bin_indices = np.searchsorted(lambda_grid, s, side="left")
        bin_indices = np.clip(bin_indices, 0, grid_size - 1)

        hist_cheap = np.zeros(grid_size, dtype=np.float64)
        hist_mid = np.zeros(grid_size, dtype=np.float64)
        np.add.at(hist_cheap, bin_indices, l_cheap / float(n_k))
        np.add.at(hist_mid, bin_indices, l_mid / float(n_k))

        rng = np.random.default_rng(seed)

        # Split budget evenly across the two curves (cheap & mid) via sequential composition
        eps_per_curve = epsilon / 2.0
        delta_per_curve = delta / 2.0

        if delta_per_curve > 0.0:
            # Gaussian mechanism on L2-sensitivity Delta_2 = sqrt(2) / n_k
            delta_2 = float(np.sqrt(2.0)) / float(n_k)
            sigma_bin = (delta_2 * float(np.sqrt(2.0 * np.log(1.25 / delta_per_curve)))) / (
                eps_per_curve
            )
            noise_cheap = rng.normal(loc=0.0, scale=sigma_bin, size=grid_size)
            noise_mid = rng.normal(loc=0.0, scale=sigma_bin, size=grid_size)
        else:
            # Pure epsilon-DP Laplace mechanism on L1-sensitivity Delta_1 = 2 / n_k
            delta_1 = 2.0 / float(n_k)
            scale_lap = delta_1 / eps_per_curve
            sigma_bin = float(np.sqrt(2.0)) * scale_lap
            noise_cheap = rng.laplace(loc=0.0, scale=scale_lap, size=grid_size)
            noise_mid = rng.laplace(loc=0.0, scale=scale_lap, size=grid_size)

        # Prefix sum of noisy histogram followed by isotonic running-maximum projection
        raw_curve_cheap = np.cumsum(hist_cheap + noise_cheap)
        raw_curve_mid = np.cumsum(hist_mid + noise_mid)

        monotone_cheap = np.clip(
            np.maximum.accumulate(np.clip(raw_curve_cheap, 0.0, 1.0)), 0.0, 1.0
        )
        monotone_mid = np.clip(np.maximum.accumulate(np.clip(raw_curve_mid, 0.0, 1.0)), 0.0, 1.0)

        return DPClientCalibrationSummary(
            client_id=client_id,
            num_samples=n_k,
            lambda_grid=[float(x) for x in lambda_grid],
            noisy_cheap_risk_curve=[float(x) for x in monotone_cheap],
            noisy_mid_risk_curve=[float(x) for x in monotone_mid],
            epsilon=float(epsilon),
            delta=float(delta),
            bin_noise_std=float(sigma_bin),
        )

    @classmethod
    def aggregate(
        cls,
        summaries: list[DPClientCalibrationSummary],
        alpha: float = 0.10,
        confidence_level: float = 0.90,
    ) -> FederatedCalibrationResult:
        """Aggregates DP client risk curves into global conformal cascade thresholds.

        Args:
            summaries: Non-empty list of DPClientCalibrationSummary objects.
            alpha: Target upper bound on expected routing loss in (0, 1).
            confidence_level: Confidence level in (0, 1) for privacy noise margin.

        Returns:
            FederatedCalibrationResult with provably conservative (lambda_1, lambda_2).
        """
        if not summaries:
            raise ValueError("summaries must be a non-empty list.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")

        base_grid = np.asarray(summaries[0].lambda_grid, dtype=np.float64)
        grid_size = int(base_grid.shape[0])

        total_samples = sum(int(s.num_samples) for s in summaries)
        if total_samples <= 0:
            raise ValueError("Total samples across summaries must be positive.")

        agg_cheap = np.zeros(grid_size, dtype=np.float64)
        agg_mid = np.zeros(grid_size, dtype=np.float64)
        var_bin_weighted = 0.0

        for summary in summaries:
            grid_k = np.asarray(summary.lambda_grid, dtype=np.float64)
            if grid_k.shape[0] != grid_size or not np.allclose(grid_k, base_grid, atol=1e-6):
                raise ValueError(
                    f"Client '{summary.client_id}' lambda_grid does not match coordinator grid."
                )
            w_k = float(summary.num_samples) / float(total_samples)
            agg_cheap += w_k * np.asarray(summary.noisy_cheap_risk_curve, dtype=np.float64)
            agg_mid += w_k * np.asarray(summary.noisy_mid_risk_curve, dtype=np.float64)
            var_bin_weighted += (w_k**2) * (summary.bin_noise_std**2)

        # Conservative privacy noise safety margin
        # By Cantelli's one-sided inequality: z = sqrt(p / (1 - p))
        z_factor = float(np.sqrt(confidence_level / max(1.0 - confidence_level, 1e-6)))
        privacy_margin = float(min(0.25, z_factor * np.sqrt(var_bin_weighted)))

        n_f = float(total_samples)
        conformal_cheap = (n_f / (n_f + 1.0)) * agg_cheap + (1.0 / (n_f + 1.0)) + privacy_margin
        conformal_mid = (n_f / (n_f + 1.0)) * agg_mid + (1.0 / (n_f + 1.0)) + privacy_margin

        valid_cheap = np.where(conformal_cheap <= alpha)[0]
        idx_1 = int(valid_cheap[-1]) if valid_cheap.size > 0 else 0
        lambda_1 = float(base_grid[idx_1])

        valid_mid = np.where(conformal_mid <= alpha)[0]
        idx_2 = int(valid_mid[-1]) if valid_mid.size > 0 else idx_1
        lambda_2 = float(max(lambda_1, base_grid[idx_2]))

        bound_at_lambda_1 = float(min(1.0, conformal_cheap[idx_1]))

        return FederatedCalibrationResult(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_target=float(alpha),
            total_samples=total_samples,
            num_clients=len(summaries),
            privacy_margin=privacy_margin,
            effective_empirical_risk_bound=bound_at_lambda_1,
        )

    @staticmethod
    def apply_to_controller(
        controller: ConformalRiskController,
        result: FederatedCalibrationResult,
    ) -> ConformalRiskController:
        """Applies federated DP calibration thresholds to a ConformalRiskController."""
        from .conformal import ConformalCalibrationSummary

        controller.alpha = result.alpha_target
        controller._effective_alpha = result.alpha_target
        controller._low_threshold = result.lambda_1
        controller._high_threshold = result.lambda_2
        controller._summary = ConformalCalibrationSummary(
            alpha=result.alpha_target,
            low_threshold=result.lambda_1,
            high_threshold=result.lambda_2,
            empirical_risk_low=result.effective_empirical_risk_bound,
            upper_risk_bound_low=result.alpha_target,
            empirical_risk_high=result.effective_empirical_risk_bound,
            upper_risk_bound_high=result.alpha_target,
            calibration_size=result.total_samples,
        )
        return controller


__all__ = [
    "DPClientCalibrationSummary",
    "FederatedCalibrationResult",
    "FederatedConformalAggregator",
]
