import torch
import pytest
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.core.router import MechanisticRouter
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TaskComplexity

@pytest.fixture
def router():
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    return MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)

def test_router_routine_task(router):
    """Verifica se tarefas rotineiras são roteadas para o SLM."""
    prompt = "Qual o meu saldo atual?"
    selected, details = router.route(prompt, TaskComplexity.ROUTINE)
    assert selected == "SLM-BERTau-Local"
    assert details["SLM-BERTau-Local"]["is_competent"] == 1.0

def test_router_moderate_task(router):
    """Verifica se tarefas moderadas são roteadas para o Mid-Tier."""
    prompt = "Por que meu cartão de crédito foi recusado ontem à noite?"
    selected, details = router.route(prompt, TaskComplexity.MODERATE)
    assert selected == "LLM-Mid-Tier"
    
    # SLM deve ser considerado incompetente para tarefas moderadas
    assert details["SLM-BERTau-Local"]["is_competent"] == 0.0
    # Mid-Tier deve ser competente
    assert details["LLM-Mid-Tier"]["is_competent"] == 1.0

def test_router_complex_task(router):
    """Verifica se tarefas complexas são roteadas para o Frontier."""
    prompt = "Análise detalhada de DTI e LTV para aprovação de crédito rural."
    selected, details = router.route(prompt, TaskComplexity.COMPLEX)
    assert selected == "LLM-Frontier-Oracle"
    
    # Ambos SLM e Mid-Tier devem ser incompetentes
    assert details["SLM-BERTau-Local"]["is_competent"] == 0.0
    assert details["LLM-Mid-Tier"]["is_competent"] == 0.0
    # Apenas o Frontier é competente
    assert details["LLM-Frontier-Oracle"]["is_competent"] == 1.0
