"""Evaluation Data Schemas for Ground Truth and Performance Benchmarking."""

from pydantic import BaseModel, Field

from ..models.types import TaskComplexity

ComplexityTier = TaskComplexity


class EvalCase(BaseModel):
    """Schema representing an evaluation dataset record containing ground-truth targets.

    Attributes:
        prompt: Raw user query evaluated by routers.
        per_model_outcome: Ground-truth capability/accuracy score per candidate model.
        price_table: Pricing table mapping model candidate identifier to USD cost per query.
        reference_tier: Ground-truth reference complexity tier (eval-only, never visible to router).
    """

    prompt: str = Field(..., description="The user prompt text evaluated by routers.")
    per_model_outcome: dict[str, float] = Field(
        ...,
        description="Ground-truth capability/accuracy score per candidate model for this query.",
    )
    price_table: dict[str, float] = Field(
        ..., description="Pricing table mapping model candidate identifier to USD cost per query."
    )
    reference_tier: ComplexityTier | None = Field(
        default=None,
        description="Ground-truth reference complexity tier (eval-only, never visible to router).",
    )
