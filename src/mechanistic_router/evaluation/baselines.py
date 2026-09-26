"""Baseline and Candidate Router Policies for Pareto Benchmarking."""

import asyncio
import concurrent.futures
import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import DEFAULT_CONFIG, RouterConfig
from ..core.encoder import SharedTrunkEncoder
from ..models.pool import MODEL_POOL
from ..models.types import TargetModel
from ..routers.causal_probe import CausalProbeRouter
from ..routers.cost_performance import CostPerformanceRouter
from ..routers.mechanistic import MechanisticRouter
from ..routers.semantic import SemanticRouter
from ..schemas.eval import EvalCase
from ..schemas.routing import RoutingRequest


class BasePolicy(ABC):
    """Abstract interface for routing policies evaluated on the Pareto benchmark."""

    name: str = "BasePolicy"

    def fit(self, train_cases: list[EvalCase]) -> None:
        """Optional training phase strictly on the train split."""
        return None

    def tune(self, dev_cases: list[EvalCase]) -> None:
        """Optional threshold or hyperparameter tuning strictly on the dev split."""
        return None

    @abstractmethod
    def select_model(self, prompt: str) -> str:
        """Synchronously selects target model given prompt text only."""
        raise NotImplementedError

    async def route(self, prompt: str) -> str:
        """Asynchronously selects target model given prompt text only."""
        return self.select_model(prompt)


class AlwaysCheapPolicy(BasePolicy):
    """Trivial baseline: always routes to the lowest-cost model candidate."""

    name: str = "AlwaysCheap (SLM-Only)"

    def __init__(self, model_pool: dict[str, TargetModel] = MODEL_POOL):
        self.model_pool = model_pool
        self.cheapest_model = min(model_pool.values(), key=lambda m: m.cost).name

    def select_model(self, prompt: str) -> str:
        return self.cheapest_model


class AlwaysStrongPolicy(BasePolicy):
    """Trivial baseline: always routes to the highest-capability frontier model candidate."""

    name: str = "AlwaysStrong (Oracle-Only)"

    def __init__(self, model_pool: dict[str, TargetModel] = MODEL_POOL):
        self.model_pool = model_pool
        self.strongest_model = max(
            model_pool.values(), key=lambda m: (m.base_accuracy, m.cost)
        ).name

    def select_model(self, prompt: str) -> str:
        return self.strongest_model


class RandomPolicy(BasePolicy):
    """Uniform random baseline policy across candidate models."""

    name: str = "Random"

    def __init__(self, model_pool: dict[str, TargetModel] = MODEL_POOL, seed: int = 42):
        self.model_pool = model_pool
        self.models = sorted(list(model_pool.keys()))
        self.rng = random.Random(seed)

    def select_model(self, prompt: str) -> str:
        return self.rng.choice(self.models)


class LengthThresholdPolicy(BasePolicy):
    """Heuristic baseline: routes to cheap if prompt length < threshold, else strong."""

    name: str = "LengthThreshold"

    def __init__(
        self,
        model_pool: dict[str, TargetModel] = MODEL_POOL,
        default_threshold_words: int = 15,
    ):
        self.model_pool = model_pool
        self.threshold = default_threshold_words
        self.cheap_model = min(model_pool.values(), key=lambda m: m.cost).name
        self.strong_model = max(model_pool.values(), key=lambda m: (m.base_accuracy, m.cost)).name

    def tune(self, dev_cases: list[EvalCase]) -> None:
        """Tune threshold on dev split to maximize quality at acceptable cost."""
        if not dev_cases:
            return

        candidate_thresholds = [5, 10, 15, 20, 25, 30]
        best_threshold = self.threshold
        best_score = -1.0

        for thresh in candidate_thresholds:
            total_acc = 0.0
            total_cost = 0.0
            for case in dev_cases:
                model = self.cheap_model if len(case.prompt.split()) < thresh else self.strong_model
                total_acc += case.per_model_outcome.get(model, 0.0)
                total_cost += case.price_table.get(model, self.model_pool[model].cost)

            mean_acc = total_acc / len(dev_cases)
            mean_cost = total_cost / len(dev_cases)
            # Composite utility metric: quality penalizing cost
            score = mean_acc - 0.2 * mean_cost
            if score > best_score:
                best_score = score
                best_threshold = thresh

        self.threshold = best_threshold

    def select_model(self, prompt: str) -> str:
        if len(prompt.split()) < self.threshold:
            return self.cheap_model
        return self.strong_model


class LearnedLogisticPolicy(BasePolicy):
    """Learned routing baseline (RouteLLM S-MLR-1 style): TF-IDF + Logistic Regression."""

    name: str = "LearnedLogistic (RouteLLM-Style)"

    def __init__(self, model_pool: dict[str, TargetModel] = MODEL_POOL, seed: int = 42):
        self.model_pool = model_pool
        self.seed = seed
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=300)
        self.classifier: LogisticRegression | None = None
        self.fallback_model = min(model_pool.values(), key=lambda m: m.cost).name

    def fit(self, train_cases: list[EvalCase]) -> None:
        """Fits TF-IDF and Logistic Regression on the train split only."""
        if not train_cases:
            return

        texts: list[str] = []
        labels: list[str] = []

        cheap_model = min(self.model_pool.values(), key=lambda m: m.cost).name
        mid_models = [
            m.name
            for m in self.model_pool.values()
            if m.cost > self.model_pool[cheap_model].cost and m.cost < 1.0
        ]
        mid_model = mid_models[0] if mid_models else cheap_model
        strong_model = max(self.model_pool.values(), key=lambda m: m.cost).name

        for case in train_cases:
            texts.append(case.prompt)
            cheap_acc = case.per_model_outcome.get(cheap_model, 0.0)
            mid_acc = case.per_model_outcome.get(mid_model, 0.0)

            # Determine cost-effective competent target
            if cheap_acc >= 0.85:
                labels.append(cheap_model)
            elif mid_acc >= 0.85:
                labels.append(mid_model)
            else:
                labels.append(strong_model)

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            # Not enough distinct classes to train logistic regression
            self.fallback_model = labels[0] if labels else cheap_model
            return

        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        clf = LogisticRegression(max_iter=500, random_state=self.seed)
        clf.fit(X, y)
        self.classifier = clf

    def select_model(self, prompt: str) -> str:
        if self.classifier is None:
            return self.fallback_model
        X = self.vectorizer.transform([prompt])
        pred = self.classifier.predict(X)[0]
        return str(pred)


class MechanisticPolicy(BasePolicy):
    """Adapter wrapping the canonical MechanisticRouter with encoder prefill probing."""

    name: str = "MechanisticRouter (Prefill-Probing)"

    def __init__(
        self,
        model_pool: dict[str, TargetModel] = MODEL_POOL,
        config: RouterConfig = DEFAULT_CONFIG,
        encoder: SharedTrunkEncoder | None = None,
    ):
        self.model_pool = model_pool
        self.config = config
        self.encoder = encoder or SharedTrunkEncoder(config)
        self.router = MechanisticRouter(self.encoder, model_pool, config)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def select_model(self, prompt: str) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            decision = self._executor.submit(asyncio.run, self.router.route(prompt)).result()
        else:
            decision = asyncio.run(self.router.route(prompt))

        return decision.selected_model

    async def route(self, prompt: str) -> str:
        decision = await self.router.route(prompt)
        return decision.selected_model


class CostPerformancePolicy(BasePolicy):
    """Adapter wrapping CostPerformanceRouter."""

    name: str = "CostPerformanceRouter"

    def __init__(
        self,
        model_pool: dict[str, TargetModel] = MODEL_POOL,
        config: RouterConfig = DEFAULT_CONFIG,
    ):
        self.model_pool = model_pool
        self.config = config
        self.router = CostPerformanceRouter(model_pool, config)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def select_model(self, prompt: str) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        req = RoutingRequest(prompt=prompt)
        if loop and loop.is_running():
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            decision = self._executor.submit(asyncio.run, self.router.route(req)).result()
        else:
            decision = asyncio.run(self.router.route(req))

        return decision.selected_model

    async def route(self, prompt: str) -> str:
        req = RoutingRequest(prompt=prompt)
        decision = await self.router.route(req)
        return decision.selected_model


class SemanticPolicy(BasePolicy):
    """Adapter wrapping SemanticRouter."""

    name: str = "SemanticRouter"

    def __init__(
        self,
        model_pool: dict[str, TargetModel] = MODEL_POOL,
        config: RouterConfig = DEFAULT_CONFIG,
    ):
        self.model_pool = model_pool
        self.config = config
        self.router = SemanticRouter(model_pool, config)
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def select_model(self, prompt: str) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        req = RoutingRequest(prompt=prompt)
        if loop and loop.is_running():
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            decision = self._executor.submit(asyncio.run, self.router.route(req)).result()
        else:
            decision = asyncio.run(self.router.route(req))

        return decision.selected_model

    async def route(self, prompt: str) -> str:
        req = RoutingRequest(prompt=prompt)
        decision = await self.router.route(req)
        return decision.selected_model


class CausalProbePolicy(BasePolicy):
    """Adapter wrapping CausalProbeRouter."""

    name: str = "CausalProbeRouter"

    def __init__(
        self,
        router: CausalProbeRouter | None = None,
        model_pool: dict[str, TargetModel] = MODEL_POOL,
    ):
        self.model_pool = model_pool
        self.router = router
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def select_model(self, prompt: str) -> str:
        if self.router is None:
            return min(self.model_pool.values(), key=lambda m: m.cost).name

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        req = RoutingRequest(prompt=prompt)
        if loop and loop.is_running():
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            decision = self._executor.submit(asyncio.run, self.router.route(req)).result()
        else:
            decision = asyncio.run(self.router.route(req))

        return decision.selected_model

    async def route(self, prompt: str) -> str:
        if self.router is None:
            return min(self.model_pool.values(), key=lambda m: m.cost).name
        req = RoutingRequest(prompt=prompt)
        decision = await self.router.route(req)
        return decision.selected_model


__all__ = [
    "BasePolicy",
    "AlwaysCheapPolicy",
    "AlwaysStrongPolicy",
    "RandomPolicy",
    "LengthThresholdPolicy",
    "LearnedLogisticPolicy",
    "MechanisticPolicy",
    "CostPerformancePolicy",
    "SemanticPolicy",
    "CausalProbePolicy",
]
