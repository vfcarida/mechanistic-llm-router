"""Local Jacobian Sensitivity & L2 Decision Robustness Certifier.

Provides analytical and randomized-smoothing robustness certificates for residual-stream
routing decisions T(h) against latent perturbations delta in R^D:
1. Exact Hyperplane & Second-Order Taylor L2 Certificates (Hein & Andriushchenko, NeurIPS 2017):
   - For LinearActivationProbe: computes the exact Euclidean distance to the nearest
     conformal logit threshold tau_*:
         R_{L2}^{exact}(h) = |z(h) - tau_*| / ||w||_2.
   - For LowRankBilinearProbe: incorporates both the local Jacobian nabla_h z(h) and the
     spectral norm of the bilinear Hessian L_H = ||nabla^2 z||_2:
         R_{L2}^{cert}(h) = (sqrt(||nabla z||_2^2 + 2 L_H |z(h) - tau_*|) - ||nabla z||_2) / L_H.
2. Randomized Gaussian Smoothing in Latent Space (Cohen et al., ICML 2019):
   Evaluates Monte Carlo stability under h_tilde ~ N(h, sigma^2 I_D), computes a
   one-sided Wilson/Hoeffding lower confidence bound p_A_lower, and certifies
       R_{smooth}(h) = sigma * Phi^{-1}(p_A_lower)  when p_A_lower > 0.5.
3. Worst-Case Adversarial Direction Extraction:
   Returns the unit direction delta^* / ||delta^*||_2 pointing toward the closest
   tier boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from .bilinear_probe import LowRankBilinearProbe
from .linear_probe import LinearActivationProbe

_STD_NORMAL = NormalDist(0.0, 1.0)


@dataclass(frozen=True)
class RobustnessCertificate:
    """Analytical and smoothed L2 robustness certificate for a routing decision."""

    prob_complex: float
    logit_value: float
    selected_tier: str
    nearest_boundary_threshold: float
    nearest_boundary_logit: float
    logit_margin: float
    jacobian_l2_norm: float
    hessian_spectral_bound: float
    certified_l2_radius: float
    relative_l2_radius: float
    smoothing_sigma: float
    empirical_stability_rate: float
    smoothed_certified_radius: float
    is_certified_stable: bool
    worst_case_perturbation_norm: float
    worst_case_direction_top_dims: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Serializes the robustness certificate to a JSON-compatible dictionary."""
        return asdict(self)


def _prob_to_logit(prob: float) -> float:
    """Converts a probability in (0, 1) to logit log(p / (1 - p))."""
    p_clipped = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(math.log(p_clipped / (1.0 - p_clipped)))


class LocalRobustnessCertifier:
    """Certifies L2 perturbation radii around residual-stream routing decisions."""

    def __init__(
        self,
        smoothing_sigma: float = 0.15,
        num_smoothing_samples: int = 128,
        confidence_alpha: float = 0.05,
        min_relative_radius: float = 0.02,
        seed: int = 42,
    ) -> None:
        if smoothing_sigma <= 0.0:
            raise ValueError(f"smoothing_sigma must be > 0, got {smoothing_sigma}.")
        if num_smoothing_samples < 8:
            raise ValueError(f"num_smoothing_samples must be >= 8, got {num_smoothing_samples}.")
        if not (0.0 < confidence_alpha < 0.5):
            raise ValueError(f"confidence_alpha must be in (0, 0.5), got {confidence_alpha}.")
        if min_relative_radius < 0.0:
            raise ValueError(f"min_relative_radius must be >= 0, got {min_relative_radius}.")

        self.smoothing_sigma = float(smoothing_sigma)
        self.num_smoothing_samples = int(num_smoothing_samples)
        self.confidence_alpha = float(confidence_alpha)
        self.min_relative_radius = float(min_relative_radius)
        self.seed = int(seed)

    @staticmethod
    def _select_tier(
        prob: float,
        low_threshold: float,
        high_threshold: float,
        tier_names: list[str],
    ) -> str:
        """Maps complexity probability to tier name given conformal thresholds."""
        if len(tier_names) >= 3:
            if prob < low_threshold:
                return tier_names[0]
            if prob < high_threshold:
                return tier_names[1]
            return tier_names[-1]
        return tier_names[-1] if prob >= high_threshold else tier_names[0]

    def certify(
        self,
        activation: np.ndarray,
        probe: LinearActivationProbe | LowRankBilinearProbe,
        low_threshold: float = 0.35,
        high_threshold: float = 0.70,
        tier_names: list[str] | None = None,
    ) -> RobustnessCertificate:
        """Computes analytical and randomized-smoothing L2 robustness certificates."""
        h_vec = np.asarray(activation, dtype=np.float64).ravel()
        if h_vec.size == 0:
            raise ValueError("activation vector must be non-empty.")
        if not (0.0 < low_threshold <= high_threshold < 1.0):
            raise ValueError("Thresholds must satisfy 0.0 < low_threshold <= high_threshold < 1.0.")

        tiers = (
            list(tier_names)
            if tier_names is not None and len(tier_names) >= 2
            else ["SLM-BERTau-Local", "LLM-Mid-Tier", "LLM-Frontier-Oracle"]
        )

        z_val = float(probe.decision_function(h_vec[None, :])[0])
        prob_val = float(probe.predict_proba(h_vec[None, :])[0])
        selected_tier = self._select_tier(prob_val, low_threshold, high_threshold, tiers)

        # Determine active decision boundary thresholds in logit space
        tau_low = _prob_to_logit(low_threshold)
        tau_high = _prob_to_logit(high_threshold)

        if len(tiers) >= 3:
            dist_low = abs(z_val - tau_low)
            dist_high = abs(z_val - tau_high)
            if dist_low <= dist_high:
                nearest_prob_thresh = float(low_threshold)
                nearest_logit_thresh = tau_low
                logit_margin = dist_low
            else:
                nearest_prob_thresh = float(high_threshold)
                nearest_logit_thresh = tau_high
                logit_margin = dist_high
        else:
            nearest_prob_thresh = float(high_threshold)
            nearest_logit_thresh = tau_high
            logit_margin = abs(z_val - tau_high)

        # Compute local Jacobian and Hessian spectral bound
        if isinstance(probe, LowRankBilinearProbe):
            grad_vec, hessian_bound = probe.local_gradient_and_hessian_bound(h_vec)
        else:
            grad_vec = np.asarray(probe.weights, dtype=np.float64).ravel()
            hessian_bound = 0.0

        grad_norm = float(np.linalg.norm(grad_vec))
        act_norm = max(float(np.linalg.norm(h_vec)), 1e-12)

        if grad_norm <= 1e-12 and hessian_bound <= 1e-12:
            cert_radius = float("inf") if logit_margin > 0 else 0.0
        elif hessian_bound <= 1e-12:
            # Exact hyperplane distance for LinearActivationProbe
            cert_radius = float(logit_margin / grad_norm)
        else:
            # Exact 2nd-order Taylor quadratic certificate for LowRankBilinearProbe
            discriminant = grad_norm * grad_norm + 2.0 * hessian_bound * logit_margin
            cert_radius = float((math.sqrt(discriminant) - grad_norm) / hessian_bound)

        rel_radius = float(cert_radius / act_norm) if math.isfinite(cert_radius) else 1e6

        # Worst-case perturbation direction pointing toward the nearest boundary
        sign_dir = 1.0 if nearest_logit_thresh >= z_val else -1.0
        if grad_norm > 1e-12:
            unit_dir = (sign_dir * grad_vec) / grad_norm
        else:
            unit_dir = np.zeros_like(h_vec)
        top_dims = [int(idx) for idx in np.argsort(np.abs(unit_dir))[::-1][: min(5, h_vec.size)]]

        # Randomized Gaussian Smoothing certificate (Cohen et al., ICML 2019)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(
            0.0,
            self.smoothing_sigma,
            size=(self.num_smoothing_samples, h_vec.size),
        )
        perturbed_batch = h_vec[None, :] + noise
        perturbed_probs = probe.predict_proba(perturbed_batch)

        matches = sum(
            1
            for p_sample in perturbed_probs
            if self._select_tier(float(p_sample), low_threshold, high_threshold, tiers)
            == selected_tier
        )
        emp_stability = float(matches / float(self.num_smoothing_samples))

        # One-sided Hoeffding lower confidence bound on p_A
        hoeffding_slack = math.sqrt(
            math.log(1.0 / self.confidence_alpha) / (2.0 * float(self.num_smoothing_samples))
        )
        p_a_lower = max(0.0, emp_stability - hoeffding_slack)

        if p_a_lower > 0.5:
            p_clamped = min(p_a_lower, 1.0 - 1e-9)
            smoothed_radius = float(self.smoothing_sigma * _STD_NORMAL.inv_cdf(p_clamped))
        else:
            smoothed_radius = 0.0

        is_stable = bool(
            rel_radius >= self.min_relative_radius
            and emp_stability >= (1.0 - self.confidence_alpha)
        )

        return RobustnessCertificate(
            prob_complex=round(prob_val, 6),
            logit_value=round(z_val, 6),
            selected_tier=selected_tier,
            nearest_boundary_threshold=round(nearest_prob_thresh, 6),
            nearest_boundary_logit=round(nearest_logit_thresh, 6),
            logit_margin=round(logit_margin, 6),
            jacobian_l2_norm=round(grad_norm, 6),
            hessian_spectral_bound=round(hessian_bound, 6),
            certified_l2_radius=round(cert_radius, 6) if math.isfinite(cert_radius) else 1e6,
            relative_l2_radius=round(rel_radius, 6),
            smoothing_sigma=self.smoothing_sigma,
            empirical_stability_rate=round(emp_stability, 6),
            smoothed_certified_radius=round(smoothed_radius, 6),
            is_certified_stable=is_stable,
            worst_case_perturbation_norm=round(cert_radius, 6)
            if math.isfinite(cert_radius)
            else 1e6,
            worst_case_direction_top_dims=top_dims,
        )


__all__ = [
    "LocalRobustnessCertifier",
    "RobustnessCertificate",
]
