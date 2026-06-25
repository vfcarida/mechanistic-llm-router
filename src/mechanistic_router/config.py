import dataclasses
from typing import Final

@dataclasses.dataclass(frozen=True)
class RouterConfig:
    """Configurações globais e hiperparâmetros do Mechanistic Router."""
    seed: int = 42
    hidden_dim: int = 128
    num_prefill_layers: int = 6
    lambda_budget: float = 0.68

    # Parâmetros para cálculo de Dimensionalidade Efetiva
    deff_complexity_threshold: float = 3.5

    # Parâmetros para Gating de Fisher
    fisher_alpha: float = 0.30
    fisher_j_threshold: float = 0.30

# Instância de configuração global padrão
DEFAULT_CONFIG: Final = RouterConfig()
