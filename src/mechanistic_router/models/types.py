import dataclasses
import enum


class TaskComplexity(enum.Enum):
    """Task complexity categorization for routing decisions.

    Complexity is not determined by classical shallow semantic heuristics
    (e.g., token count or regex keyword matching), but probed via latent
    activation representations or ground-truth evaluation tiers.
    """

    ROUTINE = "routine"  # Direct queries and lookup/memorization
    MODERATE = "moderate"  # Single-step inferential reasoning
    COMPLEX = "complex"  # Multi-step reasoning and high cognitive load


@dataclasses.dataclass(frozen=True)
class TargetModel:
    """Represents the topology and cost structure of a target model in the pool.

    Attributes:
        name (str): Unique identifier or provider API name for the model.
        cost (float): Cost per request/inference in USD.
        base_accuracy (float): Baseline capability benchmark accuracy when operating
            within the model's competence ceiling.
        complexity_ceiling (TaskComplexity): Maximum task complexity tier the model
            can reliably handle before performance degrades.
    """

    name: str
    cost: float
    base_accuracy: float
    complexity_ceiling: TaskComplexity
