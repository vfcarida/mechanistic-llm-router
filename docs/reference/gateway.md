# Gateway REST API Reference

The Mechanistic LLM Router ships with a FastAPI-powered HTTP gateway implementing OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.

---

## Authentication & Headers

All API calls must provide authorization via bearer token or header:

```http
Authorization: Bearer <ROUTER_API_KEY>
X-API-Key: <ROUTER_API_KEY>
```

### Strategy Selection Header

Callers can explicitly select the routing strategy dynamically per request using the `X-Router-Strategy` header:

```http
X-Router-Strategy: causal-probe
```

Supported strategy values:
- `mechanistic` / `simulated`: Simulated probing with SVD & Fisher gating.
- `causal-probe`: Real prefill hidden-state probing with `LinearActivationProbe`.
- `semantic`: Centroid cosine similarity routing.
- `cost-performance`: Static Pareto trade-off routing.

---

## Endpoints

### 1. Healthcheck

- **Path**: `GET /health` or `GET /healthz`
- **Auth**: None required

#### Response (`200 OK`)

```json
{
  "status": "ok",
  "service": "mechanistic-llm-router-gateway",
  "version": "0.1.0"
}
```

---

### 2. List Models

- **Path**: `GET /v1/models`
- **Auth**: Required

#### Response (`200 OK`)

```json
{
  "object": "list",
  "data": [
    {
      "id": "mechanistic-auto",
      "object": "model",
      "owned_by": "mechanistic-router"
    },
    {
      "id": "causal-probe-auto",
      "object": "model",
      "owned_by": "mechanistic-router"
    },
    {
      "id": "SLM-BERTau-Local",
      "object": "model",
      "owned_by": "system"
    },
    {
      "id": "LLM-Frontier-Oracle",
      "object": "model",
      "owned_by": "system"
    }
  ]
}
```

---

### 3. Chat Completions

- **Path**: `POST /v1/chat/completions`
- **Auth**: Required
- **Content-Type**: `application/json`

#### Request Body

```json
{
  "model": "mechanistic-auto",
  "messages": [
    {
      "role": "user",
      "content": "Perform a complete risk analysis on my DTI ratio and credit projection."
    }
  ],
  "temperature": 0.7,
  "stream": false
}
```

#### Response (`200 OK`)

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1727400000,
  "model": "LLM-Frontier-Oracle",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Risk analysis projection details..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 64,
    "total_tokens": 82
  }
}
```

### Streaming Responses

When `stream=true`, the gateway yields Server-Sent Events (`text/event-stream`):

```http
data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"delta":{"content":"Risk"}}]}

data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","choices":[{"delta":{"content":" analysis"}}]}

data: [DONE]
```

---

## Guardrails & Status Codes

| Status Code | Error Code | Cause |
|---|---|---|
| `401 Unauthorized` | `invalid_api_key` | Missing, incorrect, or empty API key. |
| `413 Payload Too Large` | `prompt_too_large` | Prompt character count exceeds `ROUTER_MAX_PROMPT_CHARS`. |
| `429 Too Many Requests` | `rate_limit_exceeded` | Client token bucket exhausted. |
| `502 Bad Gateway` | `provider_dispatch_error` | Downstream LLM provider dispatch failure. |
| `503 Service Unavailable` | `unconfigured_api_key` | Server started without `ROUTER_API_KEY` configured. |
