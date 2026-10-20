"""Unit Tests for Legacy Core Router Interface Compatibility."""

import pytest

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.routers.mechanistic import MechanisticRouter


@pytest.fixture
def legacy_router() -> MechanisticRouter:
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    return MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)


@pytest.mark.asyncio
async def test_router_routine_task(legacy_router: MechanisticRouter) -> None:
    """Verify routine prompt routes without explicit label parameter."""
    prompt = "What is my current balance?"
    decision = await legacy_router.route(prompt)
    assert decision.selected_model == "SLM-BERTau-Local"
    assert decision.signals["SLM-BERTau-Local"].is_competent is True


@pytest.mark.asyncio
async def test_router_moderate_task(legacy_router: MechanisticRouter) -> None:
    """Verify moderate prompt routes without explicit label parameter."""
    prompt = "Why was my credit card declined last night?"
    decision = await legacy_router.route(prompt)
    assert decision.selected_model == "LLM-Mid-Tier"
    assert decision.signals["LLM-Mid-Tier"].is_competent is True


@pytest.mark.asyncio
async def test_router_complex_task(legacy_router: MechanisticRouter) -> None:
    """Verify complex prompt routes without explicit label parameter."""
    prompt = "Detailed analysis of DTI and LTV for commercial loan approval."
    decision = await legacy_router.route(prompt)
    assert decision.selected_model == "LLM-Frontier-Oracle"
    assert decision.signals["LLM-Frontier-Oracle"].is_competent is True


@pytest.mark.asyncio
async def test_lambda_elasticity() -> None:
    """Test setting lambda_budget = 0.0 (100% accuracy weight)."""
    rich_config = DEFAULT_CONFIG.model_copy(update={"lambda_budget": 0.0})
    encoder = SharedTrunkEncoder(rich_config)
    router_rich = MechanisticRouter(encoder, MODEL_POOL, rich_config)

    prompt = "Hi"
    decision = await router_rich.route(prompt)
    assert decision.selected_model == "LLM-Frontier-Oracle"


@pytest.mark.asyncio
async def test_empty_prompt_handling(legacy_router: MechanisticRouter) -> None:
    """Ensure empty prompt handling does not fail."""
    prompt = "   "
    decision = await legacy_router.route(prompt)
    assert decision.selected_model in MODEL_POOL


@pytest.mark.asyncio
async def test_router_strict_validation(legacy_router: MechanisticRouter) -> None:
    """Verify type error raises on invalid arguments."""
    with pytest.raises(TypeError, match="request must be an instance of RoutingRequest or str"):
        await legacy_router.route(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="encoder must be an instance of SharedTrunkEncoder"):
        MechanisticRouter("not_an_encoder", MODEL_POOL, DEFAULT_CONFIG)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_effective_dimensionality_is_dynamic(legacy_router: MechanisticRouter) -> None:
    """Verify d_eff dynamic behavior without label leakage."""
    prompt_short = "Balance."
    decision_short = await legacy_router.route(prompt_short)
    d_eff_short = decision_short.signals["SLM-BERTau-Local"].d_eff_mean

    prompt_long = (
        "Provide a comprehensive analysis of my credit profile considering "
        "DTI, LTV, revolving credit utilization history, and 12-month "
        "repayment capacity projection."
    )
    decision_long = await legacy_router.route(prompt_long)
    d_eff_long = decision_long.signals["SLM-BERTau-Local"].d_eff_mean

    assert d_eff_short != d_eff_long
    assert d_eff_long > 1.0


def test_core_router_deprecation_warning() -> None:
    """Verify importing mechanistic_router.core.router emits DeprecationWarning."""
    import importlib

    with pytest.deprecated_call():
        import mechanistic_router.core.router as legacy_router_mod

        importlib.reload(legacy_router_mod)


def test_estimate_complexity_lru_cache() -> None:
    """Verify estimate_complexity caches results with LRU cache."""
    from mechanistic_router.routers.heuristics import estimate_complexity

    estimate_complexity.cache_clear()
    info_before = estimate_complexity.cache_info()
    assert info_before.hits == 0

    prompt = "What is my current checking account balance?"
    res1 = estimate_complexity(prompt)
    res2 = estimate_complexity(prompt)

    info_after = estimate_complexity.cache_info()
    assert res1 == res2
    assert info_after.hits >= 1


def test_prompt_to_tensor_truncation(legacy_router: MechanisticRouter) -> None:
    """Verify that _prompt_to_tensor truncates long prompts to exactly 50 tokens."""
    # 100-word prompt -> must truncate to 50 tokens
    words_100 = "word " * 100
    tensor_100 = legacy_router._prompt_to_tensor(words_100)
    assert tensor_100.shape == (1, 50)

    # Short prompt with 5 words -> shape (1, 5)
    words_5 = "one two three four five"
    tensor_5 = legacy_router._prompt_to_tensor(words_5)
    assert tensor_5.shape == (1, 5)

    # Empty prompt returns shape (1, 1) with fallback token [0]
    tensor_empty = legacy_router._prompt_to_tensor("")
    assert tensor_empty.shape == (1, 1)
    assert tensor_empty.item() == 0
