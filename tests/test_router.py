"""Unit Tests for Legacy Core Router Interface Compatibility."""

import pytest

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.core.router import MechanisticRouter
from mechanistic_router.models.pool import MODEL_POOL


@pytest.fixture
def legacy_router() -> MechanisticRouter:
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    return MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)


@pytest.mark.asyncio
async def test_router_routine_task(legacy_router: MechanisticRouter) -> None:
    """Verify routine prompt routes without explicit label parameter."""
    prompt = "Qual o meu saldo atual?"
    decision = await legacy_router.route(prompt)
    assert decision.selected_model == "SLM-BERTau-Local"
    assert decision.signals["SLM-BERTau-Local"].is_competent is True


@pytest.mark.asyncio
async def test_router_moderate_task(legacy_router: MechanisticRouter) -> None:
    """Verify moderate prompt routes without explicit label parameter."""
    prompt = "Por que meu cartão de crédito foi recusado ontem à noite?"
    decision = await legacy_router.route(prompt)
    assert decision.selected_model == "LLM-Mid-Tier"
    assert decision.signals["LLM-Mid-Tier"].is_competent is True


@pytest.mark.asyncio
async def test_router_complex_task(legacy_router: MechanisticRouter) -> None:
    """Verify complex prompt routes without explicit label parameter."""
    prompt = "Análise detalhada de DTI e LTV para aprovação de crédito rural."
    decision = await legacy_router.route(prompt)
    assert decision.selected_model == "LLM-Frontier-Oracle"
    assert decision.signals["LLM-Frontier-Oracle"].is_competent is True


@pytest.mark.asyncio
async def test_lambda_elasticity() -> None:
    """Test setting lambda_budget = 0.0 (100% accuracy weight)."""
    rich_config = DEFAULT_CONFIG.model_copy(update={"lambda_budget": 0.0})
    encoder = SharedTrunkEncoder(rich_config)
    router_rich = MechanisticRouter(encoder, MODEL_POOL, rich_config)

    prompt = "Oi"
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
    prompt_short = "Saldo."
    decision_short = await legacy_router.route(prompt_short)
    d_eff_short = decision_short.signals["SLM-BERTau-Local"].d_eff_mean

    prompt_long = (
        "Preciso de uma análise completa do meu perfil de crédito considerando "
        "DTI, LTV, histórico de utilização de crédito rotativo e projeção de "
        "capacidade de pagamento para os próximos 12 meses."
    )
    decision_long = await legacy_router.route(prompt_long)
    d_eff_long = decision_long.signals["SLM-BERTau-Local"].d_eff_mean

    assert d_eff_short != d_eff_long
    assert d_eff_long > 1.0


def test_core_router_deprecation_warning() -> None:
    """Verify importing mechanistic_router.core.router emits DeprecationWarning."""
    import importlib
    import mechanistic_router.core.router as legacy_router_mod

    with pytest.deprecated_call():
        importlib.reload(legacy_router_mod)

