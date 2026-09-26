"""Global Configuration Module using Pydantic v2 Settings."""

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterConfig(BaseSettings):
    """Immutable global configuration settings and hyperparameters for the Mechanistic Router.

    Attributes:
        seed: Random seed for statistical reproducibility across runs.
        hidden_dim: Dimensionality of the latent representation space.
        num_prefill_layers: Number of simulated layers during prefill phase.
        lambda_budget: Cost weight balance in final scoring (0.0 to 1.0).
        deff_complexity_threshold: Expected threshold for Effective Dimensionality (d_eff).
        fisher_alpha: Decay penalty applied to incompetent model scores.
        fisher_j_threshold: Minimum normalized Fisher J score required to pass Competence Gate.
        embedding_dim: Embedding dimension for semantic routing.
        similarity_threshold: Cosine similarity threshold for semantic cache and router matching.
    """

    seed: int = Field(default=42, description="Random seed for statistical reproducibility.")
    hidden_dim: int = Field(default=128, description="Latent space dimension size.")
    num_prefill_layers: int = Field(default=6, description="Number of prefill layers simulated.")
    lambda_budget: float = Field(default=0.68, description="Elastic budget cost weight factor.")

    deff_complexity_threshold: float = Field(
        default=3.5, description="Effective dimensionality complexity threshold."
    )
    fisher_alpha: float = Field(
        default=0.30, description="Penalty decay for incompetent candidates."
    )
    fisher_j_threshold: float = Field(
        default=0.30, description="Minimum Fisher J score for competence gating."
    )

    embedding_dim: int = Field(
        default=384, description="Vector embedding dimension for semantic router."
    )
    similarity_threshold: float = Field(
        default=0.85, description="Cosine similarity threshold for semantic matching."
    )

    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        case_sensitive=False,
        frozen=True,
    )


DEFAULT_CONFIG: Final[RouterConfig] = RouterConfig()
