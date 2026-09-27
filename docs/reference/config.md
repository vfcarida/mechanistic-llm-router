# Configuration Reference

The Mechanistic LLM Router uses Pydantic Settings (`RouterConfig`) for centralized, type-safe configuration. Settings are populated from environment variables with the `ROUTER_` prefix or passed programmatically at instantiation.

---

## Configuration Settings

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_dim` | `int` | `128` | Latent activation vector dimension for simulated encoder embeddings. |
| `vocab_size` | `int` | `10000` | Vocabulary size for simulated prompt tokenization. |
| `num_layers` | `int` | `4` | Number of transformer layers in the simulated encoder trunk. |
| `cost_weight` | `float` | `0.5` | Weight ($\lambda$) balancing normalized cost vs. accuracy: $\text{score} = \lambda \cdot \text{cost\_inv} + (1 - \lambda) \cdot \text{acc}$. |
| `fisher_alpha` | `float` | `0.1` | Penalty multiplier applied to models deemed non-competent by Fisher separability gating. |
| `api_key` | `str \| None` | `None` | Authentication secret key for gateway endpoints. If unset, requests return HTTP 503. |
| `port` | `int` | `8000` | Gateway HTTP server binding port. |
| `host` | `str` | `"0.0.0.0"` | Gateway HTTP server network host binding. |
| `max_prompt_chars` | `int` | `10000` | Maximum character length allowed for incoming user prompts before HTTP 413. |
| `rate_limit_capacity` | `float` | `100.0` | Token bucket maximum burst capacity per API key. |
| `rate_limit_refill_rate` | `float` | `10.0` | Token refill rate per second per API key. |

---

## Environment Variables

All settings can be configured via environment variables:

```bash
# Gateway Server
export ROUTER_PORT=8000
export ROUTER_HOST=0.0.0.0
export ROUTER_API_KEY="prod-router-secret-key"

# Guardrails & Concurrency
export ROUTER_MAX_PROMPT_CHARS=8192
export ROUTER_RATE_LIMIT_CAPACITY=50.0
export ROUTER_RATE_LIMIT_REFILL_RATE=5.0

# Mathematical Routing Weights
export ROUTER_COST_WEIGHT=0.6
export ROUTER_FISHER_ALPHA=0.15
```

---

## Programmatic Usage

```python
from mechanistic_router.config import RouterConfig

# Custom configuration instance
config = RouterConfig(
    cost_weight=0.7,
    fisher_alpha=0.05,
    max_prompt_chars=4096,
)

# Export configuration parameters as dictionary
config_dict = config.model_dump()
```
