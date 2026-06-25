import dataclasses
import enum

class TaskComplexity(enum.Enum):
    """Classificação mecanística de complexidade da tarefa."""
    ROUTINE = "routine"            # Consultas diretas, FAQ
    MODERATE = "moderate"          # Requer raciocínio moderado
    COMPLEX = "complex"            # Raciocínio multi-step, análise profunda


@dataclasses.dataclass(frozen=True)
class TargetModel:
    """Representa um modelo-alvo no pool de roteamento."""
    name: str
    cost: float
    base_accuracy: float
    complexity_ceiling: TaskComplexity
