"""Global Configuration Module using Pydantic v2 Settings."""

from typing import Any, Final

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterConfig(BaseSettings):
    """Immutable global configuration settings and hyperparameters for the Mechanistic Router.

    Attributes:
        seed: Random seed for statistical reproducibility across runs.
        hidden_dim: Dimensionality of the latent representation space.
        vocab_size: Vocabulary size for prompt tokenization.
        num_prefill_layers: Number of simulated layers during prefill phase (alias: num_layers).
        lambda_budget: Cost weight balance in final scoring 0.0 to 1.0 (alias: cost_weight).
        deff_complexity_threshold: Expected threshold for Effective Dimensionality (d_eff).
        fisher_alpha: Decay penalty applied to incompetent model scores.
        fisher_j_threshold: Minimum normalized Fisher J score required to pass Competence Gate.
        conformal_alpha: Target maximum risk bound alpha for Conformal Risk Control.
        embedding_dim: Embedding dimension for semantic routing.
        similarity_threshold: Cosine similarity threshold for semantic cache and router matching.
        max_prompt_chars: Maximum character length allowed for incoming user prompts.
        rate_limit_capacity: Token bucket burst capacity per client key.
        rate_limit_refill_rate: Token refill rate per second per client key.
        host: Gateway HTTP server network host binding.
        port: Gateway HTTP server binding port.
        api_key: Optional authentication secret key for gateway endpoints.
    """

    seed: int = Field(default=42, description="Random seed for statistical reproducibility.")
    hidden_dim: int = Field(default=128, description="Latent space dimension size.")
    vocab_size: int = Field(
        default=10000, description="Vocabulary size for simulated prompt tokenization."
    )
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
    conformal_alpha: float = Field(
        default=0.10, description="Target risk bound alpha for Conformal Risk Control."
    )

    embedding_dim: int = Field(
        default=384, description="Vector embedding dimension for semantic router."
    )
    similarity_threshold: float = Field(
        default=0.85, description="Cosine similarity threshold for semantic matching."
    )
    max_prompt_chars: int = Field(
        default=10000, description="Maximum character length allowed for incoming prompts."
    )
    rate_limit_capacity: float = Field(
        default=10.0, description="Token bucket burst capacity per client key."
    )
    rate_limit_refill_rate: float = Field(
        default=5.0, description="Token refill rate per second per client key."
    )
    host: str = Field(default="0.0.0.0", description="Gateway HTTP server network host binding.")
    port: int = Field(default=8000, description="Gateway HTTP server binding port.")
    api_key: str | None = Field(
        default=None, description="Authentication secret key for gateway endpoints."
    )

    model_config = SettingsConfigDict(
        env_prefix="ROUTER_",
        case_sensitive=False,
        frozen=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        """Maps documented alias fields (cost_weight, num_layers) to canonical fields."""
        if isinstance(data, dict):
            normalized = dict(data)
            if "cost_weight" in normalized:
                val = normalized.pop("cost_weight")
                normalized.setdefault("lambda_budget", val)
            if "num_layers" in normalized:
                val = normalized.pop("num_layers")
                normalized.setdefault("num_prefill_layers", val)
            return normalized
        return data

    @property
    def cost_weight(self) -> float:
        """Alias for lambda_budget balancing normalized cost versus accuracy."""
        return self.lambda_budget

    @property
    def num_layers(self) -> int:
        """Alias for num_prefill_layers in the encoder trunk."""
        return self.num_prefill_layers


DEFAULT_CONFIG: Final[RouterConfig] = RouterConfig()
