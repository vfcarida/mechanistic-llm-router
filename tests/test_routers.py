"""Unit Tests for Strategy Pattern Routers."""

import pytest

from mechanistic_router.config import RouterConfig
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.types import TargetModel, TaskComplexity
from mechanistic_router.routers.cost_performance import CostPerformanceRouter
from mechanistic_router.routers.heuristics import estimate_complexity
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
    assert decision_routine.signals["SLM-BERTau-Local"].is_competent is True


@pytest.mark.asyncio
async def test_semantic_router_custom_embedding(
    model_pool: dict[str, TargetModel], default_config: RouterConfig
) -> None:
    """Test SemanticRouter with custom user-provided embedding function."""
    import numpy as np

    dim = default_config.embedding_dim

    def custom_embed(text: str) -> np.ndarray:
        vec = np.ones(dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    router = SemanticRouter(model_pool, default_config, embedding_fn=custom_embed)
    req = RoutingRequest(prompt="Custom embedding test query")
    decision = await router.route(req)
    assert decision.strategy_used == "SemanticRouter"
    assert decision.selected_model in model_pool


def test_estimate_complexity_boundary_cases() -> None:
    """Test estimate_complexity heuristics across edge cases and boundary conditions."""
    # Empty or whitespace-only prompts
    assert estimate_complexity("") == TaskComplexity.ROUTINE
    assert estimate_complexity("   \n\t   ") == TaskComplexity.ROUTINE

    # Short routine prompt
    assert estimate_complexity("Saldo em conta") == TaskComplexity.ROUTINE
    assert estimate_complexity("Hello world") == TaskComplexity.ROUTINE

    # Moderate markers or length (> 10 words)
    assert (
        estimate_complexity("Gostaria de ver o detalhamento das transações")
        == TaskComplexity.MODERATE
    )
    assert estimate_complexity("Explain difference between these two") == TaskComplexity.MODERATE

    # 11 words with no complex markers -> MODERATE due to word count > 10
    eleven_words = "one two three four five six seven eight nine ten eleven"
    assert estimate_complexity(eleven_words) == TaskComplexity.MODERATE

    # Complex markers (English and Portuguese)
    assert estimate_complexity("Calculate DTI and LTV for loan") == TaskComplexity.COMPLEX
    assert (
        estimate_complexity("Análise de portfólio e diversificação de risco")
        == TaskComplexity.COMPLEX
    )

    # Long prompts (> 30 words) -> COMPLEX regardless of keywords
    long_words = "word " * 35
    assert estimate_complexity(long_words) == TaskComplexity.COMPLEX

    # LRU cache verification: repeated calls hit cache
    initial_hits = estimate_complexity.cache_info().hits
    _ = estimate_complexity("Cached prompt test string")
    _ = estimate_complexity("Cached prompt test string")
    assert estimate_complexity.cache_info().hits > initial_hits
