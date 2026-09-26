"""CausalProbeRouter Strategy Implementation."""

from __future__ import annotations

import time

import numpy as np

from ..config import DEFAULT_CONFIG, RouterConfig
from ..models.types import TargetModel
from ..probing.activation_extractor import PrefillActivationExtractor
from ..probing.linear_probe import LinearActivationProbe
from ..schemas.routing import ProbingSignals, RoutingDecision, RoutingRequest
from .base import AbstractRouter


class CausalProbeRouter(AbstractRouter):
    """Router strategy driven by real residual-stream activations and a trained linear probe.

    Extracts prefill hidden-state representations from a local open-weight model
    (such as SmolLM-135M) and classifies routing difficulty via a direction vector
    in activation space.
    """

    def __init__(
        self,
        extractor: PrefillActivationExtractor,
        probe: LinearActivationProbe,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(model_pool, config)
        self.extractor = extractor
        self.probe = probe
        self.threshold = threshold

        self.cheap_model = min(model_pool.values(), key=lambda m: m.cost).name
        self.strong_model = max(model_pool.values(), key=lambda m: m.cost).name

    async def route(self, request: RoutingRequest | str) -> RoutingDecision:
        """Evaluates prompt activation against linear probe decision boundary."""
        start_time = time.perf_counter()

        if isinstance(request, str):
            request = RoutingRequest(prompt=request)
        elif not isinstance(request, RoutingRequest):
            raise TypeError("request must be an instance of RoutingRequest or str")

        activation = self.extractor.extract_one(request.prompt)
        prob_strong = float(self.probe.predict_proba(activation[None, :])[0])

        if prob_strong >= self.threshold:
            selected_model = self.strong_model
        else:
            selected_model = self.cheap_model

        selected_target = self.model_pool[selected_model]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        signals = {
            m_name: ProbingSignals(
                d_eff_mean=float(np.linalg.norm(activation)),
                fisher_j=prob_strong,
                fisher_j_norm=prob_strong,
                is_competent=(m_name == self.strong_model or prob_strong < self.threshold),
                sae_features=[],
                extra_metadata={
                    "prob_strong": prob_strong,
                    "threshold": self.threshold,
                    "direction_norm": self.probe.direction_norm,
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
