# Changelog

All notable changes to the **Mechanistic LLM Router** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `CONTRIBUTING.md` developer guide covering quality gates, coding conventions, and PR workflows.
- `SECURITY.md` security policy with responsible vulnerability disclosure instructions.
- `CODE_OF_CONDUCT.md` adhering to the Contributor Covenant v2.1 standard.
- GitHub issue templates (`bug_report.md`, `feature_request.md`) and pull request template (`PULL_REQUEST_TEMPLATE.md`).
- `CHANGELOG.md` for tracking project evolution and release notes.
- Added `CostPerformancePolicy`, `SemanticPolicy`, and `CausalProbePolicy` to `evaluation.baselines` for evaluating all router strategies on Pareto benchmarks.
- Supported `causal-probe` strategy and `causal-probe-auto` model in FastAPI gateway `/v1/chat/completions` and `/v1/models`.
- Upgraded GitHub Actions CI workflow to test matrix (`["3.11", "3.12"]`) using `actions/checkout@v4` and `actions/setup-python@v5`.
- Added `scikit-learn>=1.3.0` to dev dependencies in `pyproject.toml`.
- Concrete `TransformerActivationEncoder` implementing `AbstractEncoder` over real HuggingFace transformer models (`AutoModel`, `AutoTokenizer`) for layer-wise prefill activation extraction.
- Enhanced `SAEEngine` with Top-K activation sparsity (Gao et al. 2024), standard MSE reconstruction + L1 sparsity loss (`compute_loss`), and custom semantic `circuit_feature_map` matching.
- OpenTelemetry GenAI semantic convention alignment (`P2-C`) in `RouterMetrics` with dual recording of `gen_ai.routing.duration`, `gen_ai.client.operation.duration`, `gen_ai.cost.saved`, and `gen_ai.routing.requests`.
- LRU caching (`@functools.lru_cache(maxsize=4096)`) on `estimate_complexity` for $O(1)$ evaluation of repeated prompts (`PERF-03`).
- Pluggable custom embedding function (`embedding_fn`) and centroids support in `SemanticRouter`.
- Streaming response support (`stream=True`) in `LiteLLMDispatcher.dispatch_stream` and FastAPI `/v1/chat/completions` with SSE (`text/event-stream`).
- Multi-turn conversation context extraction helper (`extract_routing_prompt`) in `gateway/server.py` to route based on complete dialogue flow.
- Abstract base encoder interface `AbstractEncoder` for plugging external HuggingFace and local transformer backends.
- Vectorized 2D NumPy implementation of `compute_paired_bootstrap_ci`, accelerating paired bootstrap intervals by 10–50x.
- Support for `X-Router-Strategy` header and `request.model` in the FastAPI gateway for per-request router policy selection.
- Dedicated RouterBench download script (`scripts/download_routerbench.py`) and benchmarking support (`--dataset routerbench`).
- Comprehensive documentation: `docs/getting-started/installation.md`, `docs/getting-started/quickstart.md`, `docs/guides/causal-probe.md`, `docs/guides/routerbench.md`, enriched `docs/index.md`, and updated `mkdocs.yml` navigation.

### Changed
- Refactored `server.py` gateway to use FastAPI dependency injection (`Depends`) and `lifespan` application state.
- Enhanced `TokenBucketRateLimiter` with fine-grained per-key concurrency locking instead of a single global lock.
- Optimized `MechanisticPolicy.select_model()` in `evaluation/baselines.py` to reuse a shared persistent `ThreadPoolExecutor`.
- Moved `create_financial_dataset` directly into `mechanistic_router.evaluation.dataset` to decouple the runtime package from `tests/`.
- Updated MyPy target to `python_version = "3.12"` to ensure full compatibility with modern NumPy type stubs.
- Moved `import hashlib` to top level in `src/mechanistic_router/routers/mechanistic.py`.

### Security
- Fixed SEC-01: Removed hardcoded default API key requirement, returning HTTP 503 if `ROUTER_API_KEY` is unconfigured.
- Added SEC-02: Validated `ROUTER_MAX_PROMPT_CHARS` environment variable as a strictly positive integer with fallback.
- Added SEC-03: Security audit warning logging with truncated SHA-256 client key hash upon rate limit exhaustion.

### Fixed
- Fixed 152 Ruff lint errors across the repository and normalized formatting across all modules.
- Fixed BUG-01: Runtime evaluation failure caused by importing `tests.fixtures.mock_dataset` from inside `src/`.
- Fixed BUG-02: Removed unreachable exception raising in `LiteLLMDispatcher.dispatch` retry loop and correctly chained upstream exceptions.
- Fixed BUG-03: Added warning logging when Prometheus HTTP metrics exporter fails to bind on a busy port.
- Fixed SEC-01: Removed hardcoded default API key requirement, returning HTTP 503 if `ROUTER_API_KEY` is unconfigured.

---

## [0.1.0] - 2026-09-20

### Added
- Initial public release of `mechanistic-llm-router`.
- Shared-trunk encoder prefill simulation (`SharedTrunkEncoder`).
- Mechanistic interpretability metrics:
  - Effective dimensionality ($d_{\text{eff}}$ via SVD entropy).
  - Fisher discriminant ratio ($J$) for class separability.
  - Sparse Autoencoder (SAE) cognitive circuit analysis.
- Multi-tier model routing strategies:
  - `MechanisticRouter` (multi-objective latency-cost-quality optimization).
  - `SemanticRouter` (deterministic vector embedding baseline).
  - `CostPerformanceRouter` (heuristic cost/accuracy balance).
- FastAPI drop-in OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/models`, `/healthz`).
- LiteLLM universal provider dispatch with exponential backoff retries.
- OpenTelemetry instrumentation with Prometheus metrics endpoint.
- Paired bootstrap confidence intervals and Pareto frontier evaluation harness (`BenchmarkHarness`).
- Causal interpretability spike experiment with `SmolLM-135M` (`experiments/mechanistic_probe/`).
