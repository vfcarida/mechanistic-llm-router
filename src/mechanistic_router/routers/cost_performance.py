"""CostPerformanceRouter Strategy Implementation."""

import time
from ..config import RouterConfig
from ..models.pool import get_model_accuracy
from ..models.types import TargetModel, TaskComplexity
from ..schemas.routing import ProbingSignals, RoutingDecision, RoutingRequest
from .base import AbstractRouter


class CostPerformanceRouter(AbstractRouter):
    """Router strategy executing traditional LMSYS RouteLLM cost-performance matrix heuristics.

    Evaluates prompt complexity thresholds against static model ceiling tiers and unit costs,
    selecting the cheapest candidate meeting required performance accuracy floors.
    """

    def __init__(self, model_pool: dict[str, TargetModel], config: RouterConfig):
        super().__init__(model_pool, config)

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """Determines target route by filtering competent candidates and picking minimal cost."""
        start_time = time.perf_counter()
        signals: dict[str, ProbingSignals] = {}

        # Complexity order ranking
        complexity_order = [TaskComplexity.ROUTINE, TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
        req_idx = complexity_order.index(request.task_complexity)

        candidate_scores: dict[str, float] = {}

        for name, model in self.model_pool.items():
            ceiling_idx = complexity_order.index(model.complexity_ceiling)
            accuracy = get_model_accuracy(model, request.task_complexity)
            
            # Competent if request complexity is below or equal to ceiling
            is_competent = req_idx <= ceiling_idx

            signals[name] = ProbingSignals(
                d_eff_mean=float(req_idx + 1.0),
                fisher_j=1.0 if is_competent else 0.1,
                fisher_j_norm=1.0 if is_competent else 0.1,
                is_competent=is_competent,
                extra_metadata={"cost": model.cost, "accuracy": accuracy},
            )

            if is_competent:
                # Rank primarily by lower cost
                candidate_scores[name] = 1.0 / model.cost
            else:
                candidate_scores[name] = 0.001 * accuracy

        if any(signals[m].is_competent for m in self.model_pool):
            # Select cheapest competent model
            winning_model = max(candidate_scores, key=candidate_scores.get)  # type: ignore[arg-type]
        else:
            # Fallback to highest accuracy model (frontier oracle)
            winning_model = max(
                self.model_pool.keys(), key=lambda m: self.model_pool[m].base_accuracy
            )

        latency = self._measure_latency(start_time)
        selected = self.model_pool[winning_model]

        return RoutingDecision(
            selected_model=winning_model,
            strategy_used="CostPerformanceRouter",
            estimated_cost_usd=selected.cost,
            latency_ms=latency,
            signals=signals,
        )
