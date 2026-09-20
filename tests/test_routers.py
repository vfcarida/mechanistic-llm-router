"""Unit Tests for Strategy Pattern Routers."""

import pytest
from mechanistic_router.config import RouterConfig
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.types import TargetModel
from mechanistic_router.routers.cost_performance import CostPerformanceRouter
from mechanistic_router.routers.mechanistic import MechanisticRouter
from mechanistic_router.routers.semantic import SemanticRouter
from mechanistic_router.schemas.routing import RoutingRequest


@pytest.mark.asyncio
async def test_cost_performance_router(
    model_pool: dict[str, TargetModel], default_config: RouterConfig
) -> None:
    """Test CostPerformanceRouter strategy route decisions without label leakage."""
    router = CostPerformanceRouter(model_pool, default_config)

    # Routine query -> should select cheapest SLM
    req_routine = RoutingRequest(prompt="Check credit card balance")
    decision_routine = await router.route(req_routine)
    assert decision_routine.selected_model == "SLM-BERTau-Local"
    assert decision_routine.strategy_used == "CostPerformanceRouter"

    # Complex query -> should select Frontier Oracle
    req_complex = RoutingRequest(prompt="Analyze DTI LTV risk")
    decision_complex = await router.route(req_complex)
    assert decision_complex.selected_model == "LLM-Frontier-Oracle"


@pytest.mark.asyncio
async def test_semantic_router(
    model_pool: dict[str, TargetModel], default_config: RouterConfig
) -> None:
    """Test SemanticRouter vector similarity strategy."""
    router = SemanticRouter(model_pool, default_config)

    req = RoutingRequest(prompt="What is my balance?")
    decision = await router.route(req)
    assert decision.strategy_used == "SemanticRouter"
    assert decision.selected_model in model_pool
    assert decision.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_mechanistic_router(
    mock_encoder: SharedTrunkEncoder,
    model_pool: dict[str, TargetModel],
    default_config: RouterConfig,
) -> None:
    """Test MechanisticRouter strategy probing signals and decisions."""
    router = MechanisticRouter(mock_encoder, model_pool, default_config)

    req_routine = RoutingRequest(prompt="Qual o valor da minha fatura?")
    decision_routine = await router.route(req_routine)
    assert decision_routine.strategy_used == "MechanisticRouter"
    assert decision_routine.selected_model in model_pool
    assert "SLM-BERTau-Local" in decision_routine.signals
    assert decision_routine.signals["SLM-BERTau-Local"].is_competent is True
