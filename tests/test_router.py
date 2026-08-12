"""Unit Tests for Legacy Core Router Interface Compatibility."""

import pytest
from mechanistic_router.config import DEFAULT_CONFIG, RouterConfig
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.core.router import MechanisticRouter
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TaskComplexity


@pytest.fixture
def legacy_router() -> MechanisticRouter:
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    return MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)


def test_router_routine_task(legacy_router: MechanisticRouter) -> None:
    """Verify routine tasks route to local SLM."""
    prompt = "Qual o meu saldo atual?"
    selected, details = legacy_router.route(prompt, TaskComplexity.ROUTINE)
    assert selected == "SLM-BERTau-Local"
    assert details["SLM-BERTau-Local"]["is_competent"] == 1.0


def test_router_moderate_task(legacy_router: MechanisticRouter) -> None:
    """Verify moderate tasks route to mid-tier LLM."""
    prompt = "Por que meu cartão de crédito foi recusado ontem à noite?"
    selected, details = legacy_router.route(prompt, TaskComplexity.MODERATE)
    assert selected == "LLM-Mid-Tier"
    assert details["SLM-BERTau-Local"]["is_competent"] == 0.0
    assert details["LLM-Mid-Tier"]["is_competent"] == 1.0


def test_router_complex_task(legacy_router: MechanisticRouter) -> None:
    """Verify complex tasks route to frontier oracle."""
    prompt = "Análise detalhada de DTI e LTV para aprovação de crédito rural."
    selected, details = legacy_router.route(prompt, TaskComplexity.COMPLEX)
    assert selected == "LLM-Frontier-Oracle"
    assert details["SLM-BERTau-Local"]["is_competent"] == 0.0
    assert details["LLM-Mid-Tier"]["is_competent"] == 0.0
    assert details["LLM-Frontier-Oracle"]["is_competent"] == 1.0


def test_lambda_elasticity() -> None:
    """Test setting lambda_budget = 0.0 (100% accuracy weight)."""
    rich_config = DEFAULT_CONFIG.model_copy(update={"lambda_budget": 0.0})
    encoder = SharedTrunkEncoder(rich_config)
    router_rich = MechanisticRouter(encoder, MODEL_POOL, rich_config)

    prompt = "Oi"
    selected, _ = router_rich.route(prompt, TaskComplexity.ROUTINE)
    assert selected == "LLM-Frontier-Oracle"


def test_empty_prompt_handling(legacy_router: MechanisticRouter) -> None:
    """Ensure empty prompt handling does not fail."""
    prompt = "   "
    selected, details = legacy_router.route(prompt, TaskComplexity.ROUTINE)
    assert selected in MODEL_POOL


def test_router_strict_validation(legacy_router: MechanisticRouter) -> None:
    """Verify type error raises on invalid arguments."""
    with pytest.raises(TypeError, match="prompt_text must be a string"):
        legacy_router.route(123, TaskComplexity.ROUTINE)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="complexity must be of type TaskComplexity"):
        legacy_router.route("Qual o meu saldo?", "ROUTINE")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="encoder must be an instance of SharedTrunkEncoder"):
        MechanisticRouter("not_an_encoder", MODEL_POOL, DEFAULT_CONFIG)  # type: ignore[arg-type]


def test_effective_dimensionality_is_dynamic(legacy_router: MechanisticRouter) -> None:
    """Verify d_eff dynamic behavior."""
    prompt_short = "Saldo."
    _, details_short = legacy_router.route(prompt_short, TaskComplexity.ROUTINE)
    d_eff_short = details_short["SLM-BERTau-Local"]["d_eff_mean"]

    prompt_long = (
        "Preciso de uma análise completa do meu perfil de crédito considerando "
        "DTI, LTV, histórico de utilização de crédito rotativo e projeção de "
        "capacidade de pagamento para os próximos 12 meses."
    )
    _, details_long = legacy_router.route(prompt_long, TaskComplexity.COMPLEX)
    d_eff_long = details_long["SLM-BERTau-Local"]["d_eff_mean"]

    assert d_eff_short != d_eff_long
    assert d_eff_long > d_eff_short
    assert d_eff_long > 1.0
