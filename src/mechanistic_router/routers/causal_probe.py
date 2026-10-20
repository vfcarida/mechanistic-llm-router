"""CausalProbeRouter Strategy Implementation with Multi-Tier Cascade & Conformal Risk Control."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..config import DEFAULT_CONFIG, RouterConfig
from ..models.types import TargetModel
from ..probing.activation_extractor import PrefillActivationExtractor
from ..probing.conformal import (
    ConformalCalibrationSummary,
    ConformalRiskController,
    compute_epistemic_uncertainty,
)
from ..probing.linear_probe import LinearActivationProbe
from ..probing.ood_detector import MahalanobisOODDetector
from ..probing.sae_engine import SAEEngine
from ..probing.safety_guardrail import MechanisticSafetyGuardrail
from ..schemas.routing import ProbingSignals, RoutingDecision, RoutingRequest
from .base import AbstractRouter


class CausalProbeRouter(AbstractRouter):
    """Router strategy driven by residual-stream activations, linear probing, and conformal bounds.

    Extracts prefill hidden-state representations from a local open-weight model
    (such as SmolLM-135M) and classifies routing difficulty via a learned hyperplane
    in activation space. Supports both binary routing and calibrated multi-tier
    cascades (SLM -> Mid-Tier -> Frontier Oracle) with Conformal Risk Control
    (Angelopoulos et al., ICLR 2024), epistemic uncertainty abstention,
    Mahalanobis Out-of-Distribution (OOD) gating, and Sparse Autoencoder (`SAEEngine`)
    concept-bottleneck circuit telemetry.
    """

    def __init__(
        self,
        extractor: PrefillActivationExtractor,
        probe: LinearActivationProbe,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
        threshold: float = 0.5,
        mid_threshold: float | None = None,
        conformal_controller: ConformalRiskController | None = None,
        max_uncertainty: float | None = None,
        cost_weight: float | None = None,
        sae_engine: SAEEngine | None = None,
        ood_detector: MahalanobisOODDetector | None = None,
        safety_guardrail: MechanisticSafetyGuardrail | None = None,
    ) -> None:
        super().__init__(model_pool, config)
        self.extractor = extractor
        self.probe = probe
        self.threshold = threshold
        self.mid_threshold = mid_threshold
        self.conformal_controller = conformal_controller
        self.max_uncertainty = max_uncertainty
        self.cost_weight = cost_weight if cost_weight is not None else config.lambda_budget
        self.sae_engine = sae_engine
        self._runtime_sae: SAEEngine | None = sae_engine
        self.ood_detector = ood_detector
        self.safety_guardrail = safety_guardrail

        ordered_models = sorted(model_pool.values(), key=lambda m: (m.cost, m.base_accuracy))
        self.ordered_tier_names = [m.name for m in ordered_models]
        self.cheap_model = self.ordered_tier_names[0]
        self.strong_model = self.ordered_tier_names[-1]
        self.mid_model = (
            self.ordered_tier_names[len(self.ordered_tier_names) // 2]
            if len(self.ordered_tier_names) >= 3
            else self.cheap_model
        )

    def _get_or_init_sae(self, dim: int) -> SAEEngine:
        """Retrieves or lazily initializes a deterministic SAEEngine matching `dim`."""
        if self._runtime_sae is not None and self._runtime_sae.d_in == dim:
            return self._runtime_sae
        self._runtime_sae = SAEEngine(
            d_in=dim,
            d_sae=max(64, dim * 4),
            top_k=8,
            seed=self.config.seed,
        )
        return self._runtime_sae

    def save_probe(self, filepath: str | Path) -> None:
        """Saves the underlying linear probe weights to a compressed .npz archive."""
        self.probe.save(filepath)

    def load_probe(self, filepath: str | Path) -> None:
        """Loads linear probe weights from a compressed .npz archive into this router."""
        self.probe = LinearActivationProbe.load(filepath)

    @classmethod
    def from_saved_probe(
        cls,
        probe_path: str | Path,
        extractor: PrefillActivationExtractor,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
        threshold: float = 0.5,
        mid_threshold: float | None = None,
        conformal_controller: ConformalRiskController | None = None,
        max_uncertainty: float | None = None,
    ) -> CausalProbeRouter:
        """Instantiates CausalProbeRouter by loading fitted probe weights from disk."""
        probe = LinearActivationProbe.load(probe_path)
        return cls(
            extractor=extractor,
            probe=probe,
            model_pool=model_pool,
            config=config,
            threshold=threshold,
            mid_threshold=mid_threshold,
            conformal_controller=conformal_controller,
            max_uncertainty=max_uncertainty,
        )

    def fit(self, prompts: list[str], labels: list[int] | np.ndarray) -> CausalProbeRouter:
        """Extracts activations for prompts and fits underlying LinearActivationProbe."""
        activations = self.extractor.extract_batch(prompts)
        y = np.asarray(labels, dtype=int)
        self.probe.fit(activations, y)
        return self

    def fit_ood_detector(
        self,
        prompts: list[str],
        labels: list[int] | np.ndarray | None = None,
        shrinkage: float = 0.10,
        quantile_alpha: float = 0.05,
    ) -> MahalanobisOODDetector:
        """Fits a regularized Mahalanobis OOD detector on calibration prompts."""
        activations = self.extractor.extract_batch(prompts)
        detector = MahalanobisOODDetector(shrinkage=shrinkage, quantile_alpha=quantile_alpha)
        detector.fit(activations, labels=labels)
        self.ood_detector = detector
        return detector

    def fit_safety_guardrail(
        self,
        benign_prompts: list[str],
        harmful_prompts: list[str],
        subspace_rank: int = 3,
        false_positive_alpha: float = 0.05,
    ) -> MechanisticSafetyGuardrail:
        """Fits a MechanisticSafetyGuardrail on benign and harmful calibration prompts."""
        benign_acts = self.extractor.extract_batch(benign_prompts)
        harm_acts = self.extractor.extract_batch(harmful_prompts)
        guardrail = MechanisticSafetyGuardrail(
            subspace_rank=subspace_rank,
            false_positive_alpha=false_positive_alpha,
        )
        guardrail.fit(benign_acts, harm_acts)
        self.safety_guardrail = guardrail
        return guardrail

    def calibrate_conformal(
        self,
        prompts: list[str],
        cheap_losses: list[float] | np.ndarray,
        mid_losses: list[float] | np.ndarray | None = None,
        alpha: float = 0.10,
    ) -> ConformalCalibrationSummary:
        """Calibrates distribution-free Conformal Risk Control thresholds on a calibration split.

        Args:
            prompts: Validation prompt strings.
            cheap_losses: Quality degradation loss in [0, 1] when routing to cheap tier.
            mid_losses: Optional quality degradation loss in [0, 1] when routing to mid tier.
            alpha: Maximum tolerated expected quality loss risk level in (0, 1).

        Returns:
            ConformalCalibrationSummary containing calibrated thresholds and risk bounds.
        """
        activations = self.extractor.extract_batch(prompts)
        probs = self.probe.predict_proba(activations)
        controller = ConformalRiskController(alpha=alpha)
        summary = controller.calibrate(
            probe_probs=probs,
            cheap_losses=cheap_losses,
            mid_losses=mid_losses,
        )
        self.conformal_controller = controller
        self.threshold = summary.high_threshold
        if mid_losses is not None and len(self.ordered_tier_names) >= 3:
            self.mid_threshold = summary.low_threshold
        return summary

    def update_conformal_online(
        self,
        incurred_loss_low: float,
        incurred_loss_high: float | None = None,
        step_size: float | None = None,
    ) -> tuple[float, float]:
        """Adapts conformal risk thresholds online under non-stationary distribution shift.

        Args:
            incurred_loss_low: Observed quality degradation loss in [0, 1] for cheap tier.
            incurred_loss_high: Optional observed quality degradation loss in [0, 1] for mid tier.
            step_size: Optional online learning rate override in (0, 1].

        Returns:
            Tuple of updated (low_threshold, high_threshold).
        """
        if self.conformal_controller is None or not self.conformal_controller.is_calibrated:
            raise ValueError(
                "CausalProbeRouter has no calibrated ConformalRiskController to update online."
            )
        low_t, high_t = self.conformal_controller.update_online(
            incurred_loss_low=incurred_loss_low,
            incurred_loss_high=incurred_loss_high,
            step_size=step_size,
        )
        self.threshold = high_t
        if len(self.ordered_tier_names) >= 3:
            self.mid_threshold = low_t
        return low_t, high_t

    def _select_model_from_prob(self, prob_strong: float) -> tuple[str, bool, float, float]:
        """Selects target model tier given predicted complexity probability."""
        num_tiers = len(self.ordered_tier_names)
        uncertainty = compute_epistemic_uncertainty(prob_strong)

        if self.conformal_controller is not None and self.conformal_controller.is_calibrated:
            tier_idx, abstained = self.conformal_controller.select_tier_index(
                prob_complex=prob_strong,
                num_tiers=min(num_tiers, 3),
                max_uncertainty=self.max_uncertainty,
            )
            tiers = (
                [self.cheap_model, self.mid_model, self.strong_model]
                if num_tiers >= 3
                else [self.cheap_model, self.strong_model]
            )
            return (
                tiers[tier_idx],
                abstained,
                self.conformal_controller.low_threshold,
                self.conformal_controller.high_threshold,
            )

        if self.mid_threshold is not None and num_tiers >= 3:
            low_t = min(self.mid_threshold, self.threshold)
            high_t = max(self.mid_threshold, self.threshold)
            if prob_strong < low_t:
                tier_idx = 0
            elif prob_strong < high_t:
                tier_idx = 1
            else:
                tier_idx = 2
            tiers = [self.cheap_model, self.mid_model, self.strong_model]
        else:
            low_t = self.threshold
            high_t = self.threshold
            tier_idx = 1 if prob_strong >= self.threshold else 0
            tiers = [self.cheap_model, self.strong_model]

        abstained = False
        if self.max_uncertainty is not None and uncertainty >= self.max_uncertainty:
            if tier_idx < len(tiers) - 1:
                tier_idx += 1
                abstained = True

        return tiers[tier_idx], abstained, low_t, high_t

    async def route(self, request: RoutingRequest | str) -> RoutingDecision:
        """Evaluates prompt activation against linear probe decision boundary."""
        start_time = time.perf_counter()

        if isinstance(request, str):
            request = RoutingRequest(prompt=request)
        elif not isinstance(request, RoutingRequest):
            raise TypeError("request must be an instance of RoutingRequest or str")

        activation = np.asarray(self.extractor.extract_one(request.prompt), dtype=np.float32)
        prob_strong = float(self.probe.predict_proba(activation[None, :])[0])
        uncertainty = compute_epistemic_uncertainty(prob_strong)

        selected_model, abstained, low_t, high_t = self._select_model_from_prob(prob_strong)

        mahalanobis_dist: float | None = None
        is_ood: bool = False
        if self.ood_detector is not None and self.ood_detector.is_fitted:
            mahalanobis_dist = float(self.ood_detector.score(activation)[0])
            is_ood = bool(mahalanobis_dist > self.ood_detector.threshold)
            if is_ood and selected_model != self.strong_model:
                selected_model = self.strong_model
                abstained = True

        safety_is_unsafe: bool = False
        safety_score: float | None = None
        safety_subspace_energy: float | None = None
        if self.safety_guardrail is not None and self.safety_guardrail.is_fitted:
            safety_res = self.safety_guardrail.evaluate(activation)
            safety_is_unsafe = safety_res.is_unsafe
            safety_score = safety_res.safety_score
            safety_subspace_energy = safety_res.subspace_energy_ratio

        selected_target = self.model_pool[selected_model]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        margin_val: float = 0.0
        try:
            raw_margin = self.probe.decision_function(activation[None, :])
            margin_val = float(raw_margin[0])
        except Exception:
            margin_val = 0.0

        sae = self._get_or_init_sae(int(activation.shape[0]))
        sae_info = sae.extract_active_circuits(
            torch.from_numpy(activation).float(), threshold=0.005
        )
        active_sae_features: list[int] = list(sae_info.get("active_feature_indices", []))
        circuit_type: str = str(sae_info.get("circuit_type", "factual_retrieval"))

        conformal_alpha = (
            self.conformal_controller.alpha
            if self.conformal_controller is not None and self.conformal_controller.is_calibrated
            else None
        )
        effective_alpha = (
            self.conformal_controller.effective_alpha
            if self.conformal_controller is not None and self.conformal_controller.is_calibrated
            else None
        )
        online_steps = (
            self.conformal_controller.online_steps
            if self.conformal_controller is not None and self.conformal_controller.is_calibrated
            else 0
        )
        num_groups_val = getattr(self.probe, "num_groups", 1)
        group_importances = (
            [float(x) for x in self.probe.get_group_importances()]
            if isinstance(num_groups_val, int) and num_groups_val > 1
            else [1.0]
        )

        signals = {
            m_name: ProbingSignals(
                d_eff_mean=float(np.linalg.norm(activation)),
                fisher_j=prob_strong,
                fisher_j_norm=prob_strong,
                is_competent=(
                    self.model_pool[m_name].cost >= selected_target.cost
                    or (m_name == selected_model)
                ),
                sae_features=active_sae_features,
                extra_metadata={
                    "prob_strong": prob_strong,
                    "threshold": self.threshold,
                    "low_threshold": low_t,
                    "high_threshold": high_t,
                    "direction_norm": self.probe.direction_norm,
                    "margin": margin_val,
                    "epistemic_uncertainty": uncertainty,
                    "abstention_escalated": abstained,
                    "mahalanobis_distance": mahalanobis_dist,
                    "is_ood": is_ood,
                    "safety_is_unsafe": safety_is_unsafe,
                    "safety_score": safety_score,
                    "safety_subspace_energy": safety_subspace_energy,
                    "conformal_alpha": conformal_alpha,
                    "effective_alpha": effective_alpha,
                    "online_steps": online_steps,
                    "group_importances": group_importances,
                    "circuit_type": circuit_type,
                },
            )
            for m_name in self.model_pool
        }

        return RoutingDecision(
            selected_model=selected_model,
            strategy_used="CausalProbeRouter",
            estimated_cost_usd=selected_target.cost,
            latency_ms=latency_ms,
            signals=signals,
        )


__all__ = ["CausalProbeRouter"]
