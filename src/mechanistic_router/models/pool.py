from typing import Final
from .types import TargetModel, TaskComplexity

# Pool de modelos fictícios inspirados no domínio BERTaú (financeiro).
MODEL_POOL: Final[dict[str, TargetModel]] = {
    "SLM-BERTau-Local": TargetModel(
        name="SLM-BERTau-Local",
        cost=0.02,            # US$ 0.02 por chamada
        base_accuracy=0.91,   # Excelente em tarefas rotineiras
        complexity_ceiling=TaskComplexity.ROUTINE,
    ),
    "LLM-Mid-Tier": TargetModel(
        name="LLM-Mid-Tier",
        cost=0.25,            # US$ 0.25 por chamada
        base_accuracy=0.88,   # Boa generalização
        complexity_ceiling=TaskComplexity.MODERATE,
    ),
    "LLM-Frontier-Oracle": TargetModel(
        name="LLM-Frontier-Oracle",
        cost=1.50,            # US$ 1.50 por chamada – custo altíssimo
        base_accuracy=0.97,   # Precisão quase perfeita
        complexity_ceiling=TaskComplexity.COMPLEX,
    ),
}

def get_model_accuracy(model: TargetModel, complexity: TaskComplexity) -> float:
    """Calcula a acurácia efetiva de um modelo dado a complexidade do prompt.

    Se a complexidade excede o ceiling do modelo, a acurácia sofre degradação
    proporcional à distância entre a complexidade exigida e o ceiling.
    """
    complexity_order = [TaskComplexity.ROUTINE, TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
    task_idx = complexity_order.index(complexity)
    ceiling_idx = complexity_order.index(model.complexity_ceiling)

    if task_idx <= ceiling_idx:
        # Modelo está dentro de sua zona de competência
        return model.base_accuracy
    else:
        # Degradação: cada nível acima do ceiling reduz ~15% de acurácia
        degradation = (task_idx - ceiling_idx) * 0.15
        return max(model.base_accuracy - degradation, 0.40)
