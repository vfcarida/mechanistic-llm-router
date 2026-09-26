"""Abstract Base Class for Routing Strategies (Strategy Pattern)."""

import time
from abc import ABC, abstractmethod

from ..config import RouterConfig
from ..models.types import TargetModel
from ..schemas.routing import RoutingDecision, RoutingRequest


class AbstractRouter(ABC):
    """Abstract Base Class defining the contract for all router strategies.

    Follows the SOLID Strategy Pattern to decouple router heuristic implementations
    from callers and execution dispatchers.
    """

    def __init__(self, model_pool: dict[str, TargetModel], config: RouterConfig):
        """Initializes router strategy with model pool and configuration.

        Args:
            model_pool: Mapping of model names to TargetModel metadata objects.
            config: Global configuration object.
        """
        self.model_pool = model_pool
        self.config = config

    @abstractmethod
    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """Evaluates incoming request and determines the optimal target model path.

        Args:
            request: Standard RoutingRequest containing prompt and optional parameters.

        Returns:
            RoutingDecision containing winning model, strategy name, cost estimate, and latency.
        """
        pass

    def _measure_latency(self, start_time: float) -> float:
        """Helper method calculating execution latency in milliseconds."""
        return (time.perf_counter() - start_time) * 1000.0
