# Changelog

All notable changes to the **Mechanistic LLM Router** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-10-19

### Added
- **Cross-Lingual & Cross-Family Bures-Wasserstein Subspace Alignment:** Added `BuresWassersteinAligner` and `WassersteinAlignmentTelemetry` in `mechanistic_router.probing.wasserstein_alignment` implementing closed-form Gaussian Optimal Transport ($W_2$ distance and Monge linear map), zero-shot `LinearActivationProbe` pullback (`pullback_probe`), and Álvarez-Esteban fixed-point Wasserstein barycenters (`compute_barycenter`), exposed via `POST /v1/routing/wasserstein-align`.
- **Multi-Modal Vision-Language Cross-Attention & Patch-Token Probing:** Added `MultiModalActivationProbe` and `MultiModalProbeTelemetry` in `mechanistic_router.probing.multimodal_probe` computing text-to-vision cross-attention, visual effective dimensionality ($d_{\text{eff}}^{\text{vision}}$), and FastV/PruMerge visual patch pruning and similarity-weighted merging.
- **Doubly Robust & SNIPS Counterfactual Off-Policy Evaluator:** Added `DoublyRobustPolicyEvaluator`, `LoggedRoutingSample`, and `OffPolicyEvaluationResult` in `mechanistic_router.evaluation.off_policy_evaluator` implementing Direct Method (DM), clipped IPS, Self-Normalized IPS (SNIPS), and Doubly Robust (DR) counterfactual risk estimation with Kish's Effective Sample Size (ESS) and a safe promotion gate.
- **Mechanistic Hallucination & DoLa Layer Divergence Detector:** Added `MechanisticHallucinationDetector` and `HallucinationRiskReport` in `mechanistic_router.probing.hallucination_detector` combining DoLa premature-vs-mature layer Jensen-Shannon/angular divergence and contrastive confabulation subspace probing with split conformal false-alarm calibration.
- **Off-Policy, Hallucination Check & Multi-Modal Gateway Endpoints:** Added `POST /v1/routing/off-policy-evaluate`, `POST /v1/routing/hallucination-check`, and `POST /v1/routing/multimodal-inspect` in `mechanistic_router.gateway.server` and `mechanistic_router.schemas.routing`.
- **Mechanistic KV-Cache Budget & Heavy-Hitter Token Eviction Policy:** Added `MechanisticKVCachePolicy` and `KVCacheAllocationPlan` in `mechanistic_router.probing.kv_cache_policy` combining Attention Sink preservation (Xiao et al., ICLR 2024), Heavy-Hitter Oracle ($H_2O$) selection (Zhang et al., NeurIPS 2023), complexity-modulated retention budgets, and analytical $\ell_1$ softmax perturbation error bounds.
- **Martingale E-Process & Anytime-Valid Sequential SLA Monitor:** Added `BettingMartingaleMonitor` and `EProcessStatus` in `mechanistic_router.probing.eprocess_monitor` implementing predictable aGRAPA/Kelly betting supermartingales with Ville's maximal inequality guarantees, sequential anytime-valid $p$-values, and automatic SLA circuit breaking.
- **Local Jacobian Sensitivity & $\ell_2$ Decision Robustness Certifier:** Added `LocalRobustnessCertifier` and `RobustnessCertificate` in `mechanistic_router.probing.robustness_certifier` computing exact hyperplane $\ell_2$ radii for `LinearActivationProbe`, second-order Taylor quadratic certificates for `LowRankBilinearProbe`, and randomized Gaussian smoothing radii (Cohen et al., ICML 2019).
- **KV-Cache Plan, Robustness Certificate & SLA E-Process Gateway Endpoints:** Added `POST /v1/routing/kv-cache-plan`, `POST /v1/routing/robustness-certificate`, and `GET /v1/routing/sla-eprocess` (automatically updated on `POST /v1/routing/feedback`) in `mechanistic_router.gateway.server` and `mechanistic_router.schemas.routing`.
- **Multi-Agent DAG Topological Routing & Shapley Attribution:** Added `TopologicalDAGRouter`, `AgentDAGNode`, `AgentNodeRoutingAllocation`, and `AgentDAGRoutingPlan` in `mechanistic_router.routers.dag_router` implementing Kahn topological validation, downstream path criticality, critical-path conformal $\alpha$-spending, and exact closed-form cooperative Shapley value attribution, exposed via `POST /v1/routing/dag-plan`.
- **Low-Rank Bilinear Cross-Layer Interaction Probe:** Added `LowRankBilinearProbe` and `BilinearInteractionTelemetry` in `mechanistic_router.probing.bilinear_probe` modeling rank-$r$ multiplicative cross-layer interactions $h^{(1)\top} (U V^\top) h^{(2)}$ initialized via class-conditional cross-covariance SVD.
- **Mondrian Group-Conditional Conformal Risk Control:** Added `MondrianConformalController`, `MondrianGroupCalibration`, and `MondrianCalibrationSummary` in `mechanistic_router.probing.mondrian_conformal` enforcing finite-sample group-conditional SLA guarantees across domains and SAE circuits with conservative Empirical Bayes shrinkage.
- **SecOps Subspace Ablation & Early-Exit Gateway Endpoints:** Added `POST /v1/routing/safety-evaluate` and `POST /v1/routing/early-exit` in `mechanistic_router.gateway.server` and `mechanistic_router.schemas.routing`.
- **Mechanistic Safety & Jailbreak Subspace Rejection Gate:** Added `MechanisticSafetyGuardrail` and `SafetyEvaluationResult` in `mechanistic_router.probing.safety_guardrail` extracting the unit refusal/harm direction $\hat{v}_{\text{harm}}$ (Arditi et al., NeurIPS 2024) and SVD principal jailbreak subspace $U_{\text{sub}}$ with split conformal false-positive calibration and orthogonal subspace ablation (`ablate_harm_subspace`), integrated into `CausalProbeRouter` and `/v1/chat/completions` (`x-safety-guardrail: BLOCKED`).
- **Layer-Depth Adaptive Early-Exit Probe:** Added `LayerwiseEarlyExitProbe` and `EarlyExitDecision` in `mechanistic_router.probing.early_exit_probe` implementing sequential depth-adaptive linear probing with Shannon entropy and inter-layer angular stability halting gates plus finite-sample conformal disagreement calibration (`calibrate_halting_threshold`).
- **Primal-Dual & Knapsack Batch Portfolio Router:** Added `BatchPortfolioRouter`, `PrimalDualBudgetController`, `BatchPortfolioResult`, and `BatchAllocationItem` in `mechanistic_router.routers.portfolio_optimizer` plus `POST /v1/routing/batch-optimize` in `mechanistic_router.gateway.server` for global FinOps budget-constrained portfolio allocation and online Dual Mirror Descent shadow-price adaptation.
- **Live Drift Status & OpenAI-Compatible `/v1/embeddings` Gateway:** Wired a sliding production activation window into `PopulationDriftMonitor` via `GET /v1/routing/drift-status` (with `auto_tighten=true` support) and added `POST /v1/embeddings` returning L2-normalized residual-stream embeddings with Top-K SAE circuit and complexity telemetry.
- **Differentially Private Federated Conformal Calibration:** Added `FederatedConformalAggregator`, `DPClientCalibrationSummary`, and `FederatedCalibrationResult` in `mechanistic_router.probing.federated_conformal` enabling multi-silo conformal threshold calibration under $(\epsilon, \delta)$-Gaussian or pure $\epsilon$-Laplace Differential Privacy via prefix-sum histogram perturbation.
- **Attention-Head Routing Entropy & Induction Circuit Probing:** Added `AttentionCircuitProbe` and `AttentionProbeSignals` in `mechanistic_router.probing.attention_probe` quantifying normalized per-head Shannon entropy, attention sink ratio, previous-token diagonal mass, and Olsson prefix-matching induction-head scores, integrated into `MechanisticRouter`.
- **Semantic-Mechanistic Activation Cache with SAE Circuit Safety:** Added `MechanisticActivationCache` in `mechanistic_router.gateway.activation_cache`, `X-Mechanistic-Cache` header support in `/v1/chat/completions`, and `GET /v1/routing/cache-stats` enforcing residual-stream cosine similarity, Top-K SAE circuit agreement, and epistemic uncertainty / OOD safety gates.
- **Population Kernel MMD Drift Monitor & Counterfactual Explainability Endpoint:** Added `PopulationDriftMonitor` and `DriftTestResult` in `mechanistic_router.probing.drift_monitor` implementing a multi-bandwidth RBF kernel $\text{MMD}^2$ permutation test with automatic conformal threshold tightening, plus `POST /v1/routing/counterfactual` in `mechanistic_router.gateway.server`.
- **Mahalanobis Out-of-Distribution Detection Gate:** Added `MahalanobisOODDetector` in `mechanistic_router.probing.ood_detector` implementing class-conditional shrinkage Mahalanobis distance with finite-sample conformal quantile calibration, integrated into `CausalProbeRouter` (`fit_ood_detector`, `mahalanobis_distance`, `is_ood`) for automatic OOD escalation to the frontier oracle.
- **Multi-Turn Conversation Trajectory Drift Probing:** Added `ConversationTrajectoryAnalyzer` and `TrajectoryMetrics` in `mechanistic_router.probing.trajectory_probe` quantifying inter-turn cosine drift, norm growth ratio, and trajectory SVD Effective Dimensionality ($d_{\text{eff}}$) across multi-turn dialogues.
- **Counterfactual Activation Steering & SAE Feature Clamping:** Added `ActivationSteeringEngine` and `SteeringCounterfactualResult` in `mechanistic_router.probing.activation_steering` supporting Contrastive Activation Addition (CAA), exact closed-form minimum boundary flip $\gamma^*$, and error-preserving Sparse Autoencoder feature clamping (`clamp_sae_features`).
- **Tamper-Evident SHA-256 Hash-Chained Routing Audit Ledger:** Added `RoutingAuditLedger` and `RoutingAuditEntry` in `mechanistic_router.observability.audit_log` plus `GET /v1/routing/audit-ledger` in `mechanistic_router.gateway.server` for cryptographic AI TRiSM auditability.
- **Distributed Multi-Replica Conformal State Synchronization:** Added `ConformalStateSnapshot`, `DistributedConformalCoordinator`, `InMemorySharedConformalBackend`, `AtomicFileConformalBackend`, and commutative/associative CRDT state merger `merge_conformal_snapshots()` in `mechanistic_router.probing.distributed_conformal`, plus `GET /v1/routing/conformal-state` and `POST /v1/routing/conformal-sync` in `mechanistic_router.gateway.server`.
- **End-to-End Speculative Cascade Integration in Gateway:** Wired `SpeculativeCascadeMonitor` directly into `/v1/chat/completions` (`model="speculative-cascade-auto"` or header `X-Speculative-Cascade: true`), automatically escalating hedged or high-uncertainty completions from cheaper tiers to `LLM-Frontier-Oracle` and emitting `x-speculative-handoff` and `x-initial-model` headers.
- **Contextual Bandit Routing via Sherman-Morrison LinUCB:** Added `LinUCBActivationRouter` in `mechanistic_router.routers.linucb_probe` implementing Disjoint Ridge Upper Confidence Bound routing over prefill residual-stream activations with $O(D^2)$ rank-1 Sherman-Morrison precision inverse updates (`update_feedback`), `.npz` persistence (`save`/`load`), and `linucb-auto` Gateway strategy support.
- **Unified CLI & Test Coverage Elevation:** Added `mechanistic_router.cli` (`mechanistic-router calibrate | inspect | benchmark`) registered under `[project.scripts]` in `pyproject.toml`, eliminated legacy deprecation warnings in test imports, and expanded the automated test suite (`tests/test_round4_horizon.py`, `tests/test_round5_horizon.py`, `tests/test_round6_horizon.py`) to achieve $\ge 97\%$ statement coverage.

---

## [0.2.0] - 2026-10-19

### Added
- **Cross-Family Universal Activation Alignment:** Added `OrthogonalProcrustesAligner` in `mechanistic_router.probing.alignment` solving SVD Orthogonal Procrustes alignment across heterogeneous encoder backbones and enabling zero-label probe pullback (`align_probe()`).
- **Mid-Generation Speculative Cascade Hand-Off:** Added `SpeculativeCascadeMonitor` and `SpeculativeCascadeResult` in `mechanistic_router.routers.speculative_cascade` monitoring streaming prefix token NLL, epistemic failure markers, and repetition degeneration to trigger seamless mid-stream escalation to the frontier oracle.
- **Multi-Objective Conformal Risk Control:** Added `calibrate_multi_objective()` to `ConformalRiskController` in `mechanistic_router.probing.conformal` to guarantee simultaneous finite-sample risk bounds across quality degradation ($\alpha_{\text{quality}}$) and secondary SLA/latency constraints ($\alpha_{\text{sla}}$).
- **Subspace Concept-Bottleneck SAE Routing:** Integrated live `SAEEngine` active circuit extraction (`sae_features` and `circuit_type`) directly into `CausalProbeRouter.route()`.
- **Multi-Layer Residual Stream & Group-Lasso Sparsity:** Extended `PrefillActivationExtractor` with `layer_indices: list[int] | None` and `layer_fusion: Literal["concat", "mean"]` for multi-layer representation fusion, and upgraded `LinearActivationProbe` with structured $\ell_{2,1}$ Group-Lasso block coordinate shrinkage (`num_groups`, `group_lasso_lambda`, `apply_group_lasso_shrinkage()`, `get_group_importances()`).
- **Multi-Domain RouterBench & MMLU-Pro Calibration Pack:** Added `mechanistic_router.evaluation.calibration_pack` featuring a 5-domain benchmark corpus (`math_gsm8k`, `code_mbpp`, `reasoning_mmlu_pro`, `finance_enterprise`, `rag_synthesis`), `DeterministicMultiLayerExtractor`, `build_and_save_calibration_pack()`, `load_calibrated_causal_router()`, and pre-calibrated `.npz` checkpoints in `data/checkpoints/`.
- **Adaptive Online Conformal Risk Control:** Added Gibbs & Candès Adaptive Conformal Inference (`update_online`, `learning_rate`, `effective_alpha`, `online_steps`, `online_empirical_risk`) to `ConformalRiskController` and `update_conformal_online()` to `CausalProbeRouter`.
- **White-Box Diagnostic Gateway & Mechanistic OpenTelemetry:** Added `POST /v1/routing/inspect` (zero-cost white-box routing inspection) and `POST /v1/routing/feedback` (online ACI threshold adaptation) to `mechanistic_router.gateway.server`, and added `gen_ai.mechanistic.d_eff`, `gen_ai.mechanistic.fisher_j`, `gen_ai.mechanistic.epistemic_uncertainty`, and `gen_ai.mechanistic.abstention_escalations` to `RouterMetrics`.
- Distribution-free `ConformalRiskController`, `ConformalCalibrationSummary`, and `compute_epistemic_uncertainty` in `mechanistic_router.probing.conformal` implementing finite-sample Conformal Risk Control (Angelopoulos et al., ICLR 2024) and binary Shannon entropy abstention escalation.
- Multi-tier cascade routing (`SLM-BERTau-Local` -> `LLM-Mid-Tier` -> `LLM-Frontier-Oracle`) and `calibrate_conformal()` method on `CausalProbeRouter`.
- Promoted `CausalAblationHarness` and `CausalValidationResult` into core `mechanistic_router.probing.causal_validator` for counterfactual directional ablation against random orthogonal controls.
- Wired live `SAEEngine.extract_active_circuits()` into `MechanisticRouter.route()` with deterministic seed support, replacing static placeholder feature IDs with real Top-K sparse autoencoder circuit extraction.
- Unified `RouterConfig` with `vocab_size`, `max_prompt_chars`, `host`, `port`, `api_key`, `conformal_alpha`, and bidirectional aliases (`cost_weight` <-> `lambda_budget`, `num_layers` <-> `num_prefill_layers`).
- Added canonical `ROADMAP.md`, `LICENSE` (MIT), and `CITATION.cff`.
- Added unit and property test suites in `tests/test_conformal.py`, `tests/test_config.py`, and `tests/test_next_horizon.py`.
- Property-based invariant test suite (`test_compute_convex_hull_property_invariants`) in `tests/test_signals.py` asserting strict cost monotonicity, non-decreasing accuracy, and convex slope diminution across 100 randomized topological distributions.
- Application factory function `create_app(config: RouterConfig = DEFAULT_CONFIG) -> FastAPI` and modular `APIRouter` in `mechanistic_router.gateway.server`, enabling isolated gateway instances with custom configurations for production and multi-tenant testing.
- Configurable `rate_limit_capacity` and `rate_limit_refill_rate` fields on `RouterConfig` supporting environment variable overrides (`ROUTER_RATE_LIMIT_CAPACITY`, `ROUTER_RATE_LIMIT_REFILL_RATE`).
- Integrated `encoder: TransformerActivationEncoder | None = None` parameter into `PrefillActivationExtractor`, enabling seamless sharing of preloaded transformer model backends in memory.
- Unit test `test_prefill_activation_extractor_with_encoder` in `tests/test_transformer_activation_encoder.py` and factory state isolation test `test_gateway_app_factory_isolation` in `tests/test_gateway.py`.
- Added concurrent rate limiter testing via `asyncio.gather` in `tests/test_gateway_limits.py` verifying capacity limits and cross-key concurrency isolation.
- Added comprehensive boundary and edge-case unit tests in `tests/test_sae_engine.py` covering all-zero activations, single-feature activations, 1D tensor inputs, and `threshold=0.0`.
- Dedicated API reference documentation: `docs/reference/config.md` (RouterConfig and env vars), `docs/reference/routers.md` (strategy contracts), and `docs/reference/gateway.md` (FastAPI REST endpoints and headers).
- Developer workflows documentation: `docs/development/testing.md` (quality gates and label leakage invariants) and `docs/development/releasing.md` (SemVer 2.0.0 and distribution checklists).
- Secure serialization (`save`) and loading (`load`) in `LinearActivationProbe` using compressed NumPy archives (`.npz`), eliminating arbitrary code execution vulnerabilities from pickle.
- Offline training (`fit`), probe checkpoint persistence (`save_probe`, `load_probe`), and factory method (`from_saved_probe`) in `CausalProbeRouter`.
- Pure-NumPy vector math fallback in `LinearActivationProbe.predict_proba` and `decision_function`, enabling fast, secure inference in production without a `scikit-learn` runtime dependency.
- Architectural alias `SimulatedRouter = MechanisticRouter` in `mechanistic_router.routers.mechanistic` and package exports in `mechanistic_router.__init__.py`.
- Standalone runnable demonstration script `examples/causal_probe_routing.py` showcasing probe training, serialization to disk, checkpoint reloading, and async multi-tier routing.
- Comprehensive unit test suite in `tests/test_causal_probe_router.py` covering probe persistence fidelity, unfitted save rejection, missing file errors, pure-NumPy inference, and factory initialization.
- `CONTRIBUTING.md` developer guide covering quality gates, coding conventions, and PR workflows.
- `SECURITY.md` security policy with responsible vulnerability disclosure instructions.
- `CODE_OF_CONDUCT.md` adhering to the Contributor Covenant v2.1 standard.
- GitHub issue templates (`bug_report.md`, `feature_request.md`) and pull request template (`PULL_REQUEST_TEMPLATE.md`).
- Added `--all-policies` CLI argument to `scripts/run_benchmark.py` and unit/CLI integration test suite in `tests/test_run_benchmark.py`.
- Added `/health` endpoint alias in FastAPI gateway `server.py` alongside `/healthz` for universal healthchecker compatibility.
- Added Docker `HEALTHCHECK` instruction to `Dockerfile` and updated base images to `python:3.12-slim`.
- Exported `CausalProbeRouter` directly in top-level `mechanistic_router` package for symmetrical strategy imports.
- Added comprehensive edge-case test coverage for mathematical signals (`test_signals.py`), including zeros, 1D vectors, single-row tensors, NaN graceful fallback, identity matrix scaling, and Pareto convex hull edge cases (empty, single point, duplicates, collinear points).
- Added boundary and linguistic indicator unit tests for `estimate_complexity` (`test_routers.py`) across empty prompts, whitespace, domain indicators, and LRU cache hits.
- Added deterministic unit testing covering all 3 `BenchmarkHarness` Go/No-Go decision branches ("GO", "NO-GO", "CONDITIONAL GO (Pareto Efficient)").
- Added unit tests for `normalized_inverse_cost` uniform pricing and invalid argument validation, plus `normalized_accuracy` bounds clamping (`test_cost.py`).
- Added unit tests for `load_routerbench_eval_dataset` (`test_dataset.py`) covering missing file error handling, valid JSON and JSONL parsing, and corrupted line rejection.
- Added unit test verifying `_prompt_to_tensor` token truncation to 50 tokens on 100-word inputs (`test_router.py`).
- Added `CostPerformancePolicy`, `SemanticPolicy`, and `CausalProbePolicy` to `evaluation.baselines` for evaluating all router strategies on Pareto benchmarks.
- Supported `causal-probe` strategy and `causal-probe-auto` model in FastAPI gateway `/v1/chat/completions` and `/v1/models`.
- Upgraded GitHub Actions CI workflow to test matrix (`["3.11", "3.12"]`) using `actions/checkout@v4` and `actions/setup-python@v5`.
- Concrete `TransformerActivationEncoder` implementing `AbstractEncoder` over real HuggingFace transformer models (`AutoModel`, `AutoTokenizer`) for layer-wise prefill activation extraction.
- Enhanced `SAEEngine` with Top-K activation sparsity (Gao et al. 2024), standard MSE reconstruction + L1 sparsity loss (`compute_loss`), and custom semantic `circuit_feature_map` matching.
- OpenTelemetry GenAI semantic convention alignment in `RouterMetrics` with dual recording of `gen_ai.routing.duration`, `gen_ai.client.operation.duration`, `gen_ai.cost.saved`, and `gen_ai.routing.requests`.
- LRU caching (`@functools.lru_cache(maxsize=4096)`) on `estimate_complexity` for $O(1)$ evaluation of repeated prompts.
- Pluggable custom embedding function (`embedding_fn`) and centroids support in `SemanticRouter`.
- Streaming response support (`stream=True`) in `LiteLLMDispatcher.dispatch_stream` and FastAPI `/v1/chat/completions` with SSE (`text/event-stream`).
- Multi-turn conversation context extraction helper (`extract_routing_prompt`) in `gateway/server.py` to route based on complete dialogue flow.
- Abstract base encoder interface `AbstractEncoder` for plugging external HuggingFace and local transformer backends.
- Vectorized 2D NumPy implementation of `compute_paired_bootstrap_ci`, accelerating paired bootstrap intervals by 10–50x.
- Support for `X-Router-Strategy` header and `request.model` in the FastAPI gateway for per-request router policy selection.
- Dedicated RouterBench download script (`scripts/download_routerbench.py`) and benchmarking support (`--dataset routerbench`).
- Comprehensive documentation: `docs/getting-started/installation.md`, `docs/getting-started/quickstart.md`, `docs/guides/causal-probe.md`, `docs/guides/routerbench.md`, enriched `docs/index.md`, and updated `mkdocs.yml` navigation.

### Changed
- Translated all synthetic financial evaluation prompts in `mechanistic_router.evaluation.dataset` and lexical complexity markers in `mechanistic_router.routers.heuristics` to 100% international technical English.
- Configured `LITELLM_LOCAL_MODEL_COST_MAP=True` in `mechanistic_router.gateway.dispatcher` for deterministic offline startup without remote pricing map latency.
- Refactored `server.py` gateway to use FastAPI dependency injection (`Depends`) and `lifespan` application state.
- Enhanced `TokenBucketRateLimiter` with fine-grained per-key concurrency locking instead of a single global lock.
- Optimized `MechanisticPolicy.select_model()` in `evaluation/baselines.py` to reuse a shared persistent `ThreadPoolExecutor`.
- Moved `create_financial_dataset` directly into `mechanistic_router.evaluation.dataset` to decouple the runtime package from `tests/`.
- Updated MyPy target to `python_version = "3.12"` to ensure full compatibility with modern NumPy type stubs.

### Security
- Enforced explicit API key configuration on gateway endpoints, returning HTTP 503 if `ROUTER_API_KEY` is unconfigured.
- Validated `ROUTER_MAX_PROMPT_CHARS` environment variable as a strictly positive integer with safe fallback.
- Added security audit warning logging with truncated SHA-256 client key hash upon rate limit exhaustion.

### Fixed
- Resolved Ruff lint errors across the repository and normalized formatting across all modules.
- Decoupled runtime evaluation dataset generation from test fixtures.
- Chained upstream exceptions in `LiteLLMDispatcher.dispatch` retry loop.
- Added warning logging when Prometheus HTTP metrics exporter encounters a busy port.

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
