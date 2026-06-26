import torch
import pytest
import dataclasses
from mechanistic_router.config import DEFAULT_CONFIG, RouterConfig
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.core.router import MechanisticRouter
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TaskComplexity

@pytest.fixture
def router():
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    return MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)

def test_router_routine_task(router):
    """Verifica se tarefas rotineiras são roteadas para o SLM (Portão Aberto)."""
    prompt = "Qual o meu saldo atual?"
    selected, details = router.route(prompt, TaskComplexity.ROUTINE)
    assert selected == "SLM-BERTau-Local"
    assert details["SLM-BERTau-Local"]["is_competent"] == 1.0

def test_router_moderate_task(router):
    """Verifica se tarefas moderadas são roteadas para o Mid-Tier."""
    prompt = "Por que meu cartão de crédito foi recusado ontem à noite?"
    selected, details = router.route(prompt, TaskComplexity.MODERATE)
    assert selected == "LLM-Mid-Tier"
    
    # SLM deve ser considerado incompetente
    assert details["SLM-BERTau-Local"]["is_competent"] == 0.0
    # Mid-Tier deve ser competente
    assert details["LLM-Mid-Tier"]["is_competent"] == 1.0

def test_router_complex_task(router):
    """Verifica se tarefas complexas escapam para o Frontier (Emergency Oracle)."""
    prompt = "Análise detalhada de DTI e LTV para aprovação de crédito rural."
    selected, details = router.route(prompt, TaskComplexity.COMPLEX)
    assert selected == "LLM-Frontier-Oracle"
    
    # Ambos SLM e Mid-Tier são esmagados no portão
    assert details["SLM-BERTau-Local"]["is_competent"] == 0.0
    assert details["LLM-Mid-Tier"]["is_competent"] == 0.0
    # Apenas o Frontier sobrevive
    assert details["LLM-Frontier-Oracle"]["is_competent"] == 1.0

def test_lambda_elasticity():
    """Testa se forçar um Lambda = 0 (0% custo, 100% acurácia) vicia a rota.
    
    Se ignoramos o custo completamente, o Frontier deve ganhar até as 
    tarefas mais estúpidas e banais por ter a acurácia base levemente maior.
    """
    rich_config = dataclasses.replace(DEFAULT_CONFIG, lambda_budget=0.0)
    encoder = SharedTrunkEncoder(rich_config)
    router_rich = MechanisticRouter(encoder, MODEL_POOL, rich_config)

    prompt = "Oi"
    # Mesmo sendo Rotineiro, o router rico ignora o SLM e manda pro Frontier Oracle
    selected, _ = router_rich.route(prompt, TaskComplexity.ROUTINE)
    assert selected == "LLM-Frontier-Oracle"

def test_empty_prompt_handling(router):
    """Garante que o edge-case de strings vazias ou só espaços no Roteador não quebre."""
    prompt = "   "
    selected, details = router.route(prompt, TaskComplexity.ROUTINE)
    # Por segurança, deve conseguir gerar uma predição topológica com token <pad>
    assert selected in MODEL_POOL
