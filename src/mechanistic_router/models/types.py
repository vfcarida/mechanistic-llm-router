import dataclasses
import enum

class TaskComplexity(enum.Enum):
    """Classificação mecanística de complexidade da tarefa.
    
    A complexidade não é determinada por heurísticas semânticas clássicas
    (ex: contagem de tokens ou palavras-chave), mas simulada via perfil de
    ativação latente. O ground-truth nesta simulação é categorizado
    nestes 3 níveis lógicos de processamento.
    """
    ROUTINE = "routine"            # Consultas diretas e memorização (ex: FAQ, Saldos)
    MODERATE = "moderate"          # Raciocínio inferencial de passo único
    COMPLEX = "complex"            # Raciocínio multi-step, alta complexidade cognitiva


@dataclasses.dataclass(frozen=True)
class TargetModel:
    """Representa a topologia e o custo de um modelo-alvo no pool de roteamento.

    Attributes:
        name (str): Identificador único comercial ou técnico do modelo.
        cost (float): Custo por requisição/inferência (US$).
        base_accuracy (float): Acurácia natural do modelo operando dentro do
            seu limite seguro de complexidade (zona de conforto).
        complexity_ceiling (TaskComplexity): O nível máximo de complexidade que o
            modelo consegue resolver antes que o "Fisher J" despenque e as
            alucinações/falhas superem os sucessos.
    """
    name: str
    cost: float
    base_accuracy: float
    complexity_ceiling: TaskComplexity
