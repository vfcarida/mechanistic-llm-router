# Testing Guide

The Mechanistic LLM Router maintains a strict, zero-regression testing discipline. Every commit is validated against offline test suites, leakage isolation checks, mathematical edge cases, and static analysis gates.

---

## Running Quality Gates

```bash
# 1. Run all unit and integration tests
pytest

# 2. Run with code coverage reporting
pytest --cov=mechanistic_router --cov-report=term-missing

# 3. Static type checking across all library modules
mypy src/

# 4. Code quality & linting
ruff check .

# 5. Code formatting validation
ruff format --check .
```

---

## Test Suites Overview

| Test Module | Coverage Scope |
|---|---|
| `test_causal_probe_router.py` | Linear probe persistence (`.save()`/`.load()`), pure-NumPy inference, `from_saved_probe` factory, and simulated router aliasing. |
| `test_signals.py` | Mathematical correctness of SVD Effective Dimensionality ($d_{eff}$), NaN fallbacks, Fisher separability, and Pareto convex hull edge cases. |
| `test_no_leakage.py` | Source-tree AST scans ensuring zero ground-truth label leakage into `RoutingRequest` or router policies. |
| `test_gateway.py` | FastAPI gateway routing, SSE streaming responses, OTel metric increments, and healthcheck endpoints. |
| `test_gateway_limits.py` | Authentication enforcement, `ROUTER_MAX_PROMPT_CHARS` payload rejection, and async `asyncio.gather` concurrent rate limit isolation. |
| `test_sae_engine.py` | Top-K sparsity enforcement, reconstruction loss, custom circuit mapping, and boundary conditions. |
| `test_benchmark.py` | Statistical evaluation harness, paired bootstrap confidence intervals (vectorized 2D NumPy), and Go/No-Go decision branches. |
| `test_run_benchmark.py` | Benchmark evaluation CLI `--all-policies` execution and report generation. |
| `test_transformer_activation_encoder.py` | HuggingFace `AutoModel` activation extraction over transformer layers. |

---

## Zero Label Leakage Invariant

The core scientific integrity of this repository requires that routing decisions **never** have access to ground-truth dataset labels:

- `RoutingRequest` contains solely user prompt text and optional model hints.
- `EvalCase` holds benchmark ground-truth labels and is strictly decoupled from router inputs.
- Enforced by automated AST scans in `tests/test_no_leakage.py`.
