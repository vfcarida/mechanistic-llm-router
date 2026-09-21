from typing import Final

from .types import TargetModel, TaskComplexity

# Mock target model pool inspired by benchmark domains.
MODEL_POOL: Final[dict[str, TargetModel]] = {
    "SLM-BERTau-Local": TargetModel(
        name="SLM-BERTau-Local",
        cost=0.02,  # USD 0.02 per query
        base_accuracy=0.91,  # Strong on routine retrieval and memorization
        complexity_ceiling=TaskComplexity.ROUTINE,
    ),
    "LLM-Mid-Tier": TargetModel(
        name="LLM-Mid-Tier",
        cost=0.25,  # USD 0.25 per query
        base_accuracy=0.88,  # Balanced reasoning and generalization
        complexity_ceiling=TaskComplexity.MODERATE,
    ),
    "LLM-Frontier-Oracle": TargetModel(
        name="LLM-Frontier-Oracle",
        cost=1.50,  # USD 1.50 per query - high-capacity frontier tier
        base_accuracy=0.97,  # Near-perfect reasoning accuracy
        complexity_ceiling=TaskComplexity.COMPLEX,
    ),
}


def get_model_accuracy(model: TargetModel, complexity: TaskComplexity) -> float:
    """Calculate the effective accuracy of a model given the prompt complexity.

    If task complexity exceeds the model ceiling, accuracy suffers degradation
    proportional to the distance between the required complexity and the ceiling.
    """
    complexity_order = [TaskComplexity.ROUTINE, TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
    task_idx = complexity_order.index(complexity)
    ceiling_idx = complexity_order.index(model.complexity_ceiling)

    if task_idx <= ceiling_idx:
        # Model is within its competence ceiling
        return model.base_accuracy
    else:
        # Degradation: each tier above ceiling reduces accuracy by ~15%
        degradation = (task_idx - ceiling_idx) * 0.15
        return max(model.base_accuracy - degradation, 0.40)
