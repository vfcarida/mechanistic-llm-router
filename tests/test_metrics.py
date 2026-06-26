import pytest
from mechanistic_router.utils.metrics import normalized_inverse_cost, normalized_accuracy

def test_inverse_cost_normalization():
    """Testa se a normalização inversa penaliza modelos caros e favorece baratos."""
    c_min = 0.02
    c_max = 1.50
    
    # Modelo mais barato deve bater cravado no 1.0 (melhor custo)
    assert normalized_inverse_cost(0.02, c_min, c_max) == 1.0
    
    # Modelo mais caro deve bater cravado no 0.0 (pior custo)
    assert normalized_inverse_cost(1.50, c_min, c_max) == 0.0
    
    # Modelo mediano deve estar contido entre (0, 1) com viés para o barato
    mid = normalized_inverse_cost(0.25, c_min, c_max)
    assert 0.0 < mid < 1.0
    
    # Testar falha de custo zero ou negativo
    with pytest.raises(ValueError):
        normalized_inverse_cost(0.0, c_min, c_max)

def test_inverse_cost_single_price_pool():
    """Pools sem variação de preço não devem quebrar (div_by_zero)."""
    assert normalized_inverse_cost(1.0, 1.0, 1.0) == 0.5

def test_normalized_accuracy():
    """Valida projeção de acurácia com clamp no domínio [0,1]."""
    assert normalized_accuracy(0.90, 0.50, 1.0) == pytest.approx(0.80)
    
    # Testar clamp acima de 1 (se a acurácia vier maior que o teto)
    assert normalized_accuracy(1.2, 0.50, 1.0) == 1.0
    
    # Testar clamp abaixo de 0
    assert normalized_accuracy(0.2, 0.50, 1.0) == 0.0
    
    # Single point floor/ceil fallback
    assert normalized_accuracy(0.8, 0.8, 0.8) == 1.0
