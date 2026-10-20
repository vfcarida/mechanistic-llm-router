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
  "version": "0.3.0"
}
```

---

### 2. List Models

- **Path**: `GET /v1/models`
- **Auth**: Required

---

### 3. Chat Completions & Embeddings (OpenAI-Compatible)

- **`POST /v1/chat/completions`**: Supports `mechanistic-auto`, `causal-probe-auto`, `speculative-cascade-auto`, `linucb-auto`, `semantic-auto`, and `cost-performance-auto`, plus headers `X-Router-Strategy`, `X-Speculative-Cascade: true`, `X-Mechanistic-Cache: true`, and `X-Safety-Guardrail: true`.
- **`POST /v1/embeddings`**: Returns normalized prefill residual-stream embeddings together with white-box Sparse Autoencoder (`SAEEngine`) active circuit indices and $d_{\text{eff}}$ metadata.

---

### 4. White-Box Mechanistic & Conformal Governance Endpoints

| Method & Path | Purpose |
|---|---|
| `POST /v1/routing/inspect` | Zero-cost white-box prefill inspection returning complexity probabilities, $d_{\text{eff}}$, SAE circuits, OOD distance, and trajectory drift without dispatching to an LLM. |
| `POST /v1/routing/feedback` | Online Adaptive Conformal Inference (ACI) and Martingale E-Process feedback update from observed tier losses. |
| `POST /v1/routing/conformal-sync` | Distributed multi-replica CRDT snapshot synchronization for Conformal Risk Control state. |
| `POST /v1/routing/counterfactual` | Contrastive Activation Addition (CAA) and closed-form minimal steering magnitude $\gamma^*$ required to flip routing tier decisions. |
| `POST /v1/routing/batch-optimize` | Primal-Dual & Knapsack batch portfolio optimization under a hard FinOps USD budget constraint. |
| `POST /v1/routing/dag-plan` | Multi-agent DAG topological routing with critical-path conformal $\alpha$-spending and exact Shapley value attribution. |
| `POST /v1/routing/safety-evaluate` | Mechanistic jailbreak subspace evaluation and orthogonal harm subspace ablation ($h_{\text{clean}} = h - U_{\text{sub}} U_{\text{sub}}^\top h$). |
| `POST /v1/routing/early-exit` | Layer-depth adaptive CALM early-exit evaluation with conformal disagreement bounds. |
| `POST /v1/routing/kv-cache-plan` | Heavy-Hitter Oracle (H2O) and Attention Sink KV-cache budget allocation with analytical softmax perturbation bounds. |
| `POST /v1/routing/robustness-certificate` | Local Jacobian $\ell_2$, 2nd-order Taylor bilinear, and Cohen et al. Gaussian randomized smoothing robustness certification. |
| `POST /v1/routing/off-policy-evaluate` | Counterfactual IPS, SNIPS, and Doubly Robust (DR) threshold evaluation with Kish's ESS and safe promotion gating. |
| `POST /v1/routing/hallucination-check` | DoLa premature-vs-mature layer divergence and confabulation subspace detection with conformal false-alarm control. |
| `POST /v1/routing/multimodal-inspect` | Vision-Language cross-attention probing, $d_{\text{eff}}^{\text{vision}}$, and FastV/PruMerge visual patch token pruning. |
| `POST /v1/routing/wasserstein-align` | Closed-form Bures-Wasserstein Gaussian Optimal Transport subspace alignment and zero-shot `LinearActivationProbe` pullback. |
| `GET /v1/routing/sla-eprocess` | Anytime-valid betting martingale E-process status ($E_t$, $p_t = \min(1, 1/\sup_{s \le t} E_s)$, and Ville circuit-breaker state). |
| `GET /v1/routing/drift-status` | Live sliding-window multi-bandwidth RBF Kernel $\text{MMD}^2$ permutation test against reference calibration activations. |
| `GET /v1/routing/cache-stats` | Semantic-Mechanistic Activation Cache hit rate, circuit mismatch rejections, and OOD safety blocks. |
| `GET /v1/routing/audit-ledger` | Cryptographic SHA-256 hash-chained routing audit ledger verification for EU AI Act Art. 12 compliance. |

---

## Guardrails & Status Codes

| Status Code | Error Code | Cause |
|---|---|---|
| `401 Unauthorized` | `invalid_api_key` | Missing, incorrect, or empty API key. |
| `413 Payload Too Large` | `prompt_too_large` | Prompt character count exceeds `ROUTER_MAX_PROMPT_CHARS`. |
| `429 Too Many Requests` | `rate_limit_exceeded` | Client token bucket exhausted. |
| `502 Bad Gateway` | `provider_dispatch_error` | Downstream LLM provider dispatch failure. |
| `503 Service Unavailable` | `unconfigured_api_key` | Server started without `ROUTER_API_KEY` configured. |

