"""SemanticRouter Strategy Implementation."""

import math
import time
import numpy as np
from ..config import RouterConfig
from ..models.types import TargetModel, TaskComplexity
from ..schemas.routing import ProbingSignals, RoutingDecision, RoutingRequest
from .base import AbstractRouter


class SemanticRouter(AbstractRouter):
    """Router strategy executing intent-based routing via vector embeddings and cosine similarity.

    Maps incoming prompt text into a normalized embedding representation, comparing against
    reference centroid vectors for routine queries, moderate tasks, and complex reasoning.
    """

    def __init__(self, model_pool: dict[str, TargetModel], config: RouterConfig):
        super().__init__(model_pool, config)
        # Precomputed synthetic reference centroids for task complexity intents
        rng = np.random.RandomState(config.seed)
        dim = config.embedding_dim
        self.intent_centroids = {
            TaskComplexity.ROUTINE: self._normalize_vec(rng.randn(dim)),
            TaskComplexity.MODERATE: self._normalize_vec(rng.randn(dim)),
            TaskComplexity.COMPLEX: self._normalize_vec(rng.randn(dim)),
        }

    def _normalize_vec(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    def _embed_prompt(self, prompt: str) -> np.ndarray:
        """Deterministic prompt embedding simulation using UTF-8 byte hashes."""
        dim = self.config.embedding_dim
        vec = np.zeros(dim, dtype=np.float32)
        words = prompt.split()
        for i, word in enumerate(words):
            val = sum(bytearray(word, "utf-8"))
            idx = (val + i) % dim
            vec[idx] += math.sin(val) + math.cos(i)
        return self._normalize_vec(vec)

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """Computes cosine similarity between prompt embedding and intent centroids to route."""
        start_time = time.perf_counter()
        prompt_vec = self._embed_prompt(request.prompt)

        # Compute cosine similarity with each complexity intent centroid
        similarities = {
            complexity: float(np.dot(prompt_vec, centroid))
            for complexity, centroid in self.intent_centroids.items()
        }

        # Inferred complexity tier from highest semantic similarity match
        inferred_complexity = max(similarities, key=similarities.get)  # type: ignore[arg-type]

        signals: dict[str, ProbingSignals] = {}
        candidate_scores: dict[str, float] = {}

        complexity_order = [TaskComplexity.ROUTINE, TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
        inferred_idx = complexity_order.index(inferred_complexity)

        for name, model in self.model_pool.items():
            ceiling_idx = complexity_order.index(model.complexity_ceiling)
            is_competent = inferred_idx <= ceiling_idx
            sim_val = max(0.0, (similarities[inferred_complexity] + 1.0) / 2.0)

            signals[name] = ProbingSignals(
                d_eff_mean=float(inferred_idx + 1.0),
                fisher_j=sim_val * 2.0,
                fisher_j_norm=sim_val,
                is_competent=is_competent,
                extra_metadata={"inferred_complexity": inferred_complexity.value, "similarity": sim_val},
            )

            if is_competent:
                candidate_scores[name] = (1.0 / model.cost) * sim_val
            else:
                candidate_scores[name] = 0.001 * sim_val

        winning_model = max(candidate_scores, key=candidate_scores.get)  # type: ignore[arg-type]
        latency = self._measure_latency(start_time)
        selected = self.model_pool[winning_model]

        return RoutingDecision(
            selected_model=winning_model,
            strategy_used="SemanticRouter",
            estimated_cost_usd=selected.cost,
            latency_ms=latency,
            signals=signals,
        )
