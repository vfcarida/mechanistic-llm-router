import dataclasses
from typing import Final

@dataclasses.dataclass(frozen=True)
class RouterConfig:
    """Configurações globais e hiperparâmetros do Mechanistic Router.

    Esta classe imutável centraliza a parametrização matemática e estrutural
    utilizada durante a simulação do prefill e a tomada de decisão do roteador.

    Attributes:
        seed (int): Semente para garantia de reprodutibilidade estatística.
        hidden_dim (int): Dimensão do espaço latente projetado pelo Encoder.
        num_prefill_layers (int): Número de camadas simuladas durante o prefill.
        lambda_budget (float): Peso orçamentário no score final. Valores maiores
            priorizam severamente o custo, valores menores priorizam a acurácia.
            Geralmente mantido entre 0.6 e 0.8 para obter "Cost-Optimality".
        deff_complexity_threshold (float): Limiar esperado para a Dimensionalidade Efetiva.
        fisher_alpha (float): Fator de decaimento/penalidade aplicado em scores de modelos incompetentes.
        fisher_j_threshold (float): Valor mínimo de J (normalizado) para que o modelo
            atravesse o Portão de Competência. Abaixo disso, os clusters de falha
            e sucesso se misturam e o modelo é reprovado mecanisticamente.
    """
    seed: int = 42
    hidden_dim: int = 128
    num_prefill_layers: int = 6
    lambda_budget: float = 0.68

    # Parâmetros para cálculo de Dimensionalidade Efetiva
    deff_complexity_threshold: float = 3.5

    # Parâmetros para Gating de Fisher
    fisher_alpha: float = 0.30
    fisher_j_threshold: float = 0.30

# Instância de configuração global padrão injetada na inicialização
DEFAULT_CONFIG: Final[RouterConfig] = RouterConfig()
