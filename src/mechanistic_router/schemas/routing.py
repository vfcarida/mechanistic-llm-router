"""Pydantic v2 Routing Data Schemas."""

from typing import Any

from pydantic import BaseModel, Field

from ..models.types import TaskComplexity


class ModelCandidate(BaseModel):
    """Schema representing a candidate LLM endpoint target."""

    name: str = Field(..., description="Unique model identifier or API name.")
    cost_per_query: float = Field(..., gt=0.0, description="Nominal USD cost per request.")
    base_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Baseline capability benchmark accuracy."
    )
    complexity_ceiling: TaskComplexity = Field(
        ..., description="Maximum task complexity ceiling for competent execution."
    )


class ProbingSignals(BaseModel):
    """Diagnostic signals extracted from mechanistic prefill probing or vector matching."""

    d_eff_mean: float = Field(..., description="Mean Effective Dimensionality (entropy spectrum).")
    fisher_j: float = Field(..., description="Raw Fisher Discriminant separability measure.")
    fisher_j_norm: float = Field(
        ..., description="Normalized Fisher J competence score in range [0, 1]."
    )
    is_competent: bool = Field(..., description="Whether the model crossed the competence gate.")
    sae_features: list[int] = Field(default_factory=list, description="Active SAE feature indices.")
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional debug signals."
    )


class RoutingRequest(BaseModel):
    """Incoming prompt payload for router evaluation."""

    prompt: str = Field(..., min_length=1, description="Raw prompt query text.")
    user_id: str | None = Field(default=None, description="Optional caller user identifier.")
    max_budget_usd: float | None = Field(
        default=None, description="Optional hard constraint on cost."
    )


class RoutingDecision(BaseModel):
    """Final decision output from the router strategy."""

    selected_model: str = Field(..., description="Target model name selected for execution.")
    strategy_used: str = Field(..., description="Name of the router strategy applied.")
    estimated_cost_usd: float = Field(..., description="Estimated cost for selected route.")
    latency_ms: float = Field(default=0.0, description="Router execution decision latency in ms.")
    signals: dict[str, ProbingSignals] = Field(
        default_factory=dict,
        description="Topological probing signals per evaluated model candidate.",
    )
