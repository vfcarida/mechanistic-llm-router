# Quickstart Guide

This quickstart demonstrates how to use `mechanistic-llm-router` both programmatically as a Python library and as a drop-in OpenAI-compatible API gateway.

---

## 1. Python Library Usage

You can use the router directly inside your Python application:

```python
import asyncio
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.routers.mechanistic import MechanisticRouter
from mechanistic_router.schemas.routing import RoutingRequest


async def main():
    # 1. Initialize encoder and mechanistic router
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)

    # 2. Construct routing request (strictly isolated from ground-truth labels)
    request = RoutingRequest(
        prompt="Analyze the fiscal implications of sovereign debt restructuring under inflation."
    )

    # 3. Evaluate prompt and route to optimal candidate model
    decision = await router.route(request)

    print(f"Selected Model: {decision.selected_model}")
    print(f"Strategy Used:  {decision.strategy_used}")
    print(f"Estimated Cost: ${decision.estimated_cost_usd:.6f}")
    if decision.probing_signals:
        print(f"Avg d_eff:      {decision.probing_signals.effective_dimensionality:.3f}")
        print(f"Fisher Score:   {decision.probing_signals.fisher_separability:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 2. Running the API Gateway

Launch the FastAPI gateway server using Uvicorn:

```bash
export ROUTER_API_KEY="my-secret-key"
uvicorn mechanistic_router.gateway.server:app --host 0.0.0.0 --port 8000
```

On Windows (PowerShell):

```powershell
$env:ROUTER_API_KEY="my-secret-key"
uvicorn mechanistic_router.gateway.server:app --host 0.0.0.0 --port 8000
```

---

## 3. Drop-in OpenAI Compatibility

Query the `/v1/chat/completions` endpoint using standard cURL or the official `openai` Python SDK:

### Using cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{
    "model": "mechanistic-auto",
    "messages": [
      {"role": "user", "content": "What is the capital of Brazil?"}
    ],
    "temperature": 0.7
  }'
```

Response includes `router_strategy` and `cost_saved_usd` headers and body fields:

```json
{
  "id": "chatcmpl-mocked",
  "object": "chat.completion",
  "created": 1758832000,
  "model": "SLM-BERTau-Local",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of Brazil is Brasília."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 8,
    "total_tokens": 15
  },
  "router_strategy": "MechanisticRouter -> SLM-BERTau-Local",
  "cost_saved_usd": 1.48
}
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key",
)

response = client.chat.completions.create(
    model="mechanistic-auto",
    messages=[
        {"role": "user", "content": "Write a quick Python script to calculate Fibonacci numbers."}
    ],
)

print(response.choices[0].message.content)
```

---

## 4. Selecting Routing Strategies Dynamically

You can choose the router policy on a per-request basis using either:

1. **HTTP Header**: `X-Router-Strategy: semantic` or `X-Router-Strategy: cost-performance`
2. **Model Name**: Specifying `"model": "semantic-auto"` or `"model": "cost-performance-auto"` in the payload
