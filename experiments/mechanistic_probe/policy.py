"""Causal Probe Policy integrating with MLR evaluation harness."""

from __future__ import annotations

import numpy as np

from mechanistic_router.evaluation.baselines import MODEL_POOL, BasePolicy, TargetModel
from mechanistic_router.schemas.eval import EvalCase

from .activation_extractor import PrefillActivationExtractor
from .probe import LinearActivationProbe


class CausalProbePolicy(BasePolicy):
    """Routing policy driven by real residual-stream activations and a trained probe."""

    name: str = "CausalProbeRouter (SmolLM-135M)"

    def __init__(
        self,
        extractor: PrefillActivationExtractor,
        probe: LinearActivationProbe,
        model_pool: dict[str, TargetModel] = MODEL_POOL,
        threshold: float = 0.5,
    ) -> None:
        self.extractor = extractor
        self.probe = probe
        self.model_pool = model_pool
        self.threshold = threshold

        self.cheap_model = min(model_pool.values(), key=lambda m: m.cost).name
        self.strong_model = max(model_pool.values(), key=lambda m: m.cost).name

    def fit_probe(self, train_cases: list[EvalCase]) -> None:
        """Extracts activations on train split and fits the probe without eval leakage."""
        if not train_cases:
            return

        prompts = [c.prompt for c in train_cases]
        X = self.extractor.extract_batch(prompts)

        # Label: 0 if cheap model succeeds, 1 if strong model needed
        y: list[int] = []
        for c in train_cases:
            cheap_acc = c.per_model_outcome.get(self.cheap_model, 0.0)
            y.append(0 if cheap_acc >= 0.85 else 1)

        y_arr = np.array(y, dtype=int)
        self.probe.fit(X, y_arr)

    def tune(self, dev_cases: list[EvalCase]) -> None:
        """Tunes the routing probability threshold on dev split."""
        if not dev_cases:
            return

        prompts = [c.prompt for c in dev_cases]
        X_dev = self.extractor.extract_batch(prompts)
        probs = self.probe.predict_proba(X_dev)

        best_threshold = 0.5
        best_score = -1e9

        for thresh in np.linspace(0.1, 0.9, 17):
            total_acc = 0.0
            total_cost = 0.0
            for idx, case in enumerate(dev_cases):
                model = self.strong_model if probs[idx] >= thresh else self.cheap_model
                total_acc += case.per_model_outcome.get(model, 0.0)
                total_cost += case.price_table.get(model, self.model_pool[model].cost)

            mean_acc = total_acc / len(dev_cases)
            mean_cost = total_cost / len(dev_cases)
            score = mean_acc - 0.2 * mean_cost

            if score > best_score:
                best_score = score
                best_threshold = float(thresh)

        self.threshold = best_threshold

    def select_model(self, prompt: str) -> str:
        """Selects target model using only prompt text (zero leakage)."""
        activation = self.extractor.extract_one(prompt)
        prob = float(self.probe.predict_proba(activation[None, :])[0])
        if prob >= self.threshold:
            return self.strong_model
        return self.cheap_model

    def get_probe_direction(self) -> np.ndarray:
        """Returns the unit directional vector of the underlying probe."""
        return self.probe.direction_vector
