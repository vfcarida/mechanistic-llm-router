# Shared Conventions — Mechanistic LLM Router (MLR)

This document establishes development conventions, testing protocols, execution boundaries, and architectural standards for the `mechanistic-llm-router` repository across tasks MLR-T01 through MLR-T05.

---

## 1. Environment & Dependencies

- **Python Version**: Python `>=3.10` (tested on Python `3.12.10`).
- **Dependency Management**: Standard `pyproject.toml` with `setuptools` build backend.
- **Local Isolation**: All local executions use a disposable `.venv`.
- **Resource Constraints**: Avoid large network downloads (e.g. multi-gigabyte PyTorch/CUDA binaries) during CI/test preflights; utilize system packages or CPU builds where appropriate.

---

## 2. Leakage Isolation Principles (MLR-F9)

- **Decoupling Integrity**: The router must NEVER receive ground truth evaluation labels (`TaskComplexity` or class targets) as inputs to its routing or feature-extraction decisions.
- **Production Routing Boundary**: In runtime production requests (`RoutingRequest` / `/v1/chat/completions`), the router only observes the user prompt, optional user constraints (budget, latency SLA), and upstream context—never the downstream evaluation label.
- **Synthesized Probing Signals**: Any mock or simulation activation generators must decouple topological signals (e.g., token entropy, layer trajectories) from label cheating.
- **Test Integrity**: Unit and integration tests must not assert route selections based on hard-coded ground truth complexity flags passed into `route()`.

---

## 3. Code Style & Static Analysis Quality Gates

All contributions must pass the following quality gates:

1. **Ruff (Linter & Formatter)**:
   ```bash
   ruff check .
   ruff format --check .
   ```
   - Target line length: `100`.
   - Python target: `py310`.
   - Selected rule sets: `["E", "F", "W", "I", "B", "UP"]`.

2. **MyPy (Static Type Checking)**:
   ```bash
   mypy src/
   ```
   - Strict typing enforced: `disallow_untyped_defs = true`, `warn_return_any = true`.

3. **Pytest (Unit & Integration Testing with Coverage)**:
   ```bash
   pytest -v --cov=mechanistic_router --cov-report=term-missing --cov-report=xml
   ```
   - All tests must pass offline.
   - External model APIs (OpenAI, Anthropic, etc.) must be mocked or handled via LiteLLM mock/fallback handlers without incurring live network billing.

---

## 4. Execution Boundaries

- **Offline Only**: No outbound HTTP requests to paid LLM inference APIs during automated testing.
- **Source Preservation**: Preflight and audit tasks must never modify files under `src/` without explicit approval and designated task scope.
- **Git Hygiene**: No unstaged drift, no pushing to remote or force-pushes without coordination.
