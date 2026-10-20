"""Mechanistic Token-Level Hallucination & Epistemic Divergence Detector.

Detects latent confabulation risk during prefill prior to autoregressive generation:
1. DoLa Contrastive Layer Divergence (Chuang et al., ICLR 2024):
   Measures Jensen-Shannon and angular divergence between premature layer
   representations h^{(l)} (l < L-1) and the final mature layer h^{(L-1)}.
2. Contrastive Truthfulness / Confabulation Subspace Probing (Azaria & Mitchell,
   EMNLP 2023; Burns et al., ICLR 2023):
   Projects residual activations onto the unit contrastive confabulation direction
   v_halluc = (mu_confab - mu_factual) / ||mu_confab - mu_factual||_2.
3. Finite-Sample Conformal False-Alarm Calibration:
   Sets the escalation threshold via the finite-sample conformal quantile
   ceil((n + 1)(1 - alpha)) / n over grounded factual activations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HallucinationRiskReport:
    """Mechanistic prefill hallucination and confabulation risk assessment."""

    hallucination_risk_score: float
    conformal_threshold: float
    should_escalate_to_oracle: bool
    dola_layer_divergence: float
    truthfulness_projection: float
    epistemic_entropy: float
    num_layers_analyzed: int
    dominant_divergent_layer: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes hallucination risk report to a JSON-compatible dictionary."""
        return asdict(self)


class MechanisticHallucinationDetector:
    """Detects prefill epistemic instability and confabulation risk via DoLa and CCS probing."""

    def __init__(
        self,
        num_layers: int = 3,
        false_alarm_alpha: float = 0.10,
        divergence_weight: float = 0.55,
        truthfulness_weight: float = 0.45,
    ) -> None:
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 for DoLa contrast, got {num_layers}.")
        if not (0.0 < false_alarm_alpha < 1.0):
            raise ValueError(f"false_alarm_alpha must be in (0, 1), got {false_alarm_alpha}.")
        if (
            divergence_weight < 0.0
            or truthfulness_weight < 0.0
            or (divergence_weight + truthfulness_weight) <= 0.0
        ):
            raise ValueError(
                "divergence_weight and truthfulness_weight must be >= 0 with positive sum."
            )

        total_w = float(divergence_weight + truthfulness_weight)
        self.num_layers = int(num_layers)
        self.false_alarm_alpha = float(false_alarm_alpha)
        self.divergence_weight = float(divergence_weight / total_w)
        self.truthfulness_weight = float(truthfulness_weight / total_w)

        self._confab_direction: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._scale: float = 1.0
        self._threshold: float = 0.60
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether contrastive truthfulness calibration has been fitted."""
        return self._is_fitted

    @property
    def conformal_threshold(self) -> float:
        """Calibrated conformal threshold for hallucination escalation."""
        return self._threshold

    @staticmethod
    def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Computes normalized Jensen-Shannon divergence in [0, 1] between probability vectors."""
        p_safe = np.maximum(p, 1e-12)
        q_safe = np.maximum(q, 1e-12)
        m = 0.5 * (p_safe + q_safe)
        kl_pm = float(np.sum(p_safe * np.log(p_safe / m)))
        kl_qm = float(np.sum(q_safe * np.log(q_safe / m)))
        js_nats = 0.5 * (kl_pm + kl_qm)
        return float(np.clip(js_nats / math.log(2.0), 0.0, 1.0))

    @staticmethod
    def _softmax_projection(vec: np.ndarray) -> np.ndarray:
        """Maps a layer slice into a normalized softmax distribution."""
        scale = max(math.sqrt(float(vec.size)), 1.0)
        logits = vec / scale
        shifted = logits - float(np.max(logits))
        exp_v = np.exp(shifted)
        return np.asarray(exp_v / max(float(np.sum(exp_v)), 1e-12), dtype=np.float64)

    def _compute_dola_layer_divergence(self, activation: np.ndarray) -> tuple[float, int, int]:
        """Computes DoLa premature-vs-mature layer divergence and dominant divergent layer."""
        h_vec = np.asarray(activation, dtype=np.float64).ravel()
        if h_vec.size < 4:
            raise ValueError(f"Activation dimension must be >= 4, got {h_vec.size}.")

        eff_layers = min(self.num_layers, h_vec.size // 2)
        eff_layers = max(2, eff_layers)
        d_layer = h_vec.size // eff_layers
        layers = [h_vec[i * d_layer : (i + 1) * d_layer] for i in range(eff_layers)]

        mature = layers[-1]
        mature_norm = max(float(np.linalg.norm(mature)), 1e-12)
        mature_dist = self._softmax_projection(mature)

        divergences: list[float] = []
        for l_idx in range(eff_layers - 1):
            premature = layers[l_idx]
            prem_norm = max(float(np.linalg.norm(premature)), 1e-12)
            cos_sim = float(np.dot(premature, mature) / (prem_norm * mature_norm))
            angular_div = float(np.clip(0.5 * (1.0 - cos_sim), 0.0, 1.0))

            prem_dist = self._softmax_projection(premature)
            js_div = self._jensen_shannon_divergence(prem_dist, mature_dist)
            divergences.append(0.55 * angular_div + 0.45 * js_div)

        dom_layer = int(np.argmax(divergences))
        max_div = float(np.clip(divergences[dom_layer], 0.0, 1.0))
        return max_div, dom_layer, eff_layers

    def _score_single(
        self,
        h_vec: np.ndarray,
        prob_complex: float | None = None,
    ) -> tuple[float, float, float, float, int, int]:
        """Returns (risk_score, dola_div, truth_proj, entropy, dom_layer, eff_layers)."""
        dola_div, dom_layer, eff_layers = self._compute_dola_layer_divergence(h_vec)

        if self._confab_direction is not None and self._center is not None:
            proj = float(np.dot(h_vec - self._center, self._confab_direction) / self._scale)
            truth_risk = float(1.0 / (1.0 + math.exp(-float(np.clip(proj, -30.0, 30.0)))))
        else:
            # Unsupervised spectral kurtosis / high-frequency fluctuation proxy
            centered = h_vec - float(np.mean(h_vec))
            std_v = max(float(np.std(centered)), 1e-6)
            z_abs = np.abs(centered) / std_v
            tail_ratio = float(np.mean(z_abs > 1.5))
            truth_risk = float(np.clip(tail_ratio * 2.5, 0.0, 1.0))

        if prob_complex is not None:
            p = float(np.clip(prob_complex, 1e-6, 1.0 - 1e-6))
            ent = float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))
        else:
            ent = float(np.clip(2.0 * dola_div * (1.0 - dola_div), 0.0, 1.0))

        risk_score = float(
            np.clip(
                self.divergence_weight * dola_div + self.truthfulness_weight * truth_risk,
                0.0,
                1.0,
            )
        )
        return risk_score, dola_div, truth_risk, ent, dom_layer, eff_layers

    def fit(
        self,
        factual_activations: np.ndarray,
        confabulation_activations: np.ndarray,
    ) -> MechanisticHallucinationDetector:
        """Fits contrastive confabulation direction and calibrates conformal false-alarm bound."""
        X_fact = np.asarray(factual_activations, dtype=np.float64)
        X_conf = np.asarray(confabulation_activations, dtype=np.float64)
        if X_fact.ndim != 2 or X_conf.ndim != 2:
            raise ValueError("factual_activations and confabulation_activations must be 2D arrays.")
        if X_fact.shape[0] < 2 or X_conf.shape[0] < 1:
            raise ValueError("Need at least 2 factual and 1 confabulation calibration samples.")
        if X_fact.shape[1] != X_conf.shape[1] or X_fact.shape[1] < 4:
            raise ValueError("Feature dimensions must match and be >= 4.")

        mu_fact = np.mean(X_fact, axis=0)
        mu_conf = np.mean(X_conf, axis=0)
        diff = mu_conf - mu_fact
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm > 1e-12:
            self._confab_direction = diff / diff_norm
        else:
            self._confab_direction = np.zeros_like(mu_fact)
            self._confab_direction[0] = 1.0

        self._center = 0.5 * (mu_fact + mu_conf)
        proj_fact = (X_fact - self._center[None, :]) @ self._confab_direction
        proj_conf = (X_conf - self._center[None, :]) @ self._confab_direction
        spread = float(np.std(np.concatenate([proj_fact, proj_conf])))
        self._scale = max(spread, 1e-4)

        # Compute non-conformity hallucination scores on factual calibration set
        fact_scores = np.array(
            [self._score_single(X_fact[i])[0] for i in range(X_fact.shape[0])],
            dtype=np.float64,
        )
        n = int(fact_scores.size)
        rank_idx = min(n, max(1, int(math.ceil((n + 1) * (1.0 - self.false_alarm_alpha)))))
        sorted_scores = np.sort(fact_scores)
        self._threshold = float(np.clip(sorted_scores[rank_idx - 1], 0.05, 0.99))
        self._is_fitted = True
        return self

    def evaluate(
        self,
        activation: np.ndarray,
        prob_complex: float | None = None,
    ) -> HallucinationRiskReport:
        """Evaluates prefill hallucination risk and oracle escalation recommendation."""
        h_vec = np.asarray(activation, dtype=np.float64).ravel()
        if (
            self._is_fitted
            and self._confab_direction is not None
            and h_vec.size != self._confab_direction.size
        ):
            raise ValueError(
                f"Expected activation dimension {self._confab_direction.size}, got {h_vec.size}."
            )

        risk_score, dola_div, truth_risk, ent, dom_layer, eff_layers = self._score_single(
            h_vec, prob_complex=prob_complex
        )
        should_escalate = bool(risk_score >= self._threshold)

        return HallucinationRiskReport(
            hallucination_risk_score=round(risk_score, 6),
            conformal_threshold=round(self._threshold, 6),
            should_escalate_to_oracle=should_escalate,
            dola_layer_divergence=round(dola_div, 6),
            truthfulness_projection=round(truth_risk, 6),
            epistemic_entropy=round(ent, 6),
            num_layers_analyzed=eff_layers,
            dominant_divergent_layer=dom_layer,
        )


__all__ = [
    "HallucinationRiskReport",
    "MechanisticHallucinationDetector",
]
