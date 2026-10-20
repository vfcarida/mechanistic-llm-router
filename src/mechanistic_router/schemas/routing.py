"""Pydantic v2 Routing Data Schemas."""

from typing import Any

from pydantic import BaseModel, Field

from ..models.types import TaskComplexity


class ModelCandidate(BaseModel):
    """Schema representing a candidate LLM endpoint target."""

    name: str = Field(..., description="Unique model identifier or API name.")
    cost_per_query: float = Field(..., gt=0.0, description="Nominal USD cost per request.")
    base_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Baseline capability benchmark accuracy."
    )
    complexity_ceiling: TaskComplexity = Field(
        ..., description="Maximum task complexity ceiling for competent execution."
    )


class ProbingSignals(BaseModel):
    """Diagnostic signals extracted from mechanistic prefill probing or vector matching."""

    d_eff_mean: float = Field(..., description="Mean Effective Dimensionality (entropy spectrum).")
    fisher_j: float = Field(..., description="Raw Fisher Discriminant separability measure.")
    fisher_j_norm: float = Field(
        ..., description="Normalized Fisher J competence score in range [0, 1]."
    )
    is_competent: bool = Field(..., description="Whether the model crossed the competence gate.")
    sae_features: list[int] = Field(default_factory=list, description="Active SAE feature indices.")
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional debug signals."
    )


class RoutingRequest(BaseModel):
    """Incoming prompt payload for router evaluation."""

    prompt: str = Field(..., min_length=1, description="Raw prompt query text.")
    user_id: str | None = Field(default=None, description="Optional caller user identifier.")
    max_budget_usd: float | None = Field(
        default=None, description="Optional hard constraint on cost."
    )


class RoutingDecision(BaseModel):
    """Final decision output from the router strategy."""

    selected_model: str = Field(..., description="Target model name selected for execution.")
    strategy_used: str = Field(..., description="Name of the router strategy applied.")
    estimated_cost_usd: float = Field(..., description="Estimated cost for selected route.")
    latency_ms: float = Field(default=0.0, description="Router execution decision latency in ms.")
    signals: dict[str, ProbingSignals] = Field(
        default_factory=dict,
        description="Topological probing signals per evaluated model candidate.",
    )


class RoutingInspectionRequest(BaseModel):
    """Payload for white-box mechanistic routing inspection without upstream LLM execution."""

    prompt: str = Field(..., min_length=1, description="Prompt text to inspect.")
    strategy: str = Field(
        default="causal-probe-auto",
        description="Router strategy identifier (e.g. causal-probe-auto, mechanistic-auto).",
    )


class RoutingInspectionResponse(BaseModel):
    """Detailed white-box diagnostic response from /v1/routing/inspect."""

    selected_model: str = Field(..., description="Selected target model tier.")
    strategy_used: str = Field(..., description="Router strategy executed.")
    estimated_cost_usd: float = Field(..., description="Estimated USD cost for selected tier.")
    cost_saved_usd: float = Field(..., description="USD saved relative to frontier oracle.")
    latency_ms: float = Field(..., description="Router inspection latency in milliseconds.")
    signals: dict[str, ProbingSignals] = Field(
        default_factory=dict,
        description="Per-model mechanistic probing and conformal telemetry signals.",
    )
    conformal_state: dict[str, Any] | None = Field(
        default=None,
        description="Active Conformal Risk Control state (thresholds, alpha, online steps).",
    )


class ConformalFeedbackRequest(BaseModel):
    """Online feedback payload for Adaptive Conformal Inference (Gibbs & Candes ACI)."""

    incurred_loss_low: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Observed quality loss in [0, 1] when routing to the cheap tier.",
    )
    incurred_loss_high: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional observed quality loss in [0, 1] when routing to the mid tier.",
    )
    step_size: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Optional online conformal step size eta in (0, 1].",
    )


class ConformalFeedbackResponse(BaseModel):
    """Updated Conformal Risk Control state after online feedback step."""

    low_threshold: float = Field(..., description="Updated low-tier probability threshold.")
    high_threshold: float = Field(..., description="Updated high-tier probability threshold.")
    effective_alpha: float = Field(..., description="Current online adaptive risk level alpha_t.")
    online_steps: int = Field(..., description="Total online feedback steps processed.")
    online_empirical_risk_low: float = Field(
        ..., description="Running online empirical risk for low tier."
    )
    online_empirical_risk_high: float = Field(
        ..., description="Running online empirical risk for mid/high tier."
    )


class ConformalSyncRequest(BaseModel):
    """Payload for multi-replica distributed Conformal Risk Control synchronization."""

    delta_steps: int = Field(
        default=0,
        ge=0,
        description="Incremental online feedback steps accumulated by remote replica.",
    )
    delta_loss_low: float = Field(
        default=0.0,
        ge=0.0,
        description="Incremental cumulative low-tier loss accumulated by remote replica.",
    )
    delta_loss_high: float = Field(
        default=0.0,
        ge=0.0,
        description="Incremental cumulative high-tier loss accumulated by remote replica.",
    )
    replica_id: str = Field(
        default="remote-peer",
        description="Identifier of the peer worker replica.",
    )


class ConformalStateResponse(BaseModel):
    """Synchronized distributed Conformal Risk Control snapshot response."""

    alpha: float = Field(..., description="Target risk budget alpha.")
    effective_alpha: float = Field(..., description="Current adaptive risk level alpha_t.")
    base_low_threshold: float = Field(..., description="Base calibrated low-tier threshold.")
    base_high_threshold: float = Field(..., description="Base calibrated high-tier threshold.")
    low_threshold: float = Field(..., description="Current synchronized low-tier threshold.")
    high_threshold: float = Field(..., description="Current synchronized high-tier threshold.")
    online_steps: int = Field(..., description="Synchronized total online steps across replicas.")
    cumulative_loss_low: float = Field(..., description="Synchronized cumulative low-tier loss.")
    cumulative_loss_high: float = Field(..., description="Synchronized cumulative high-tier loss.")
    replica_id: str = Field(..., description="Replica provenance identifier.")


class AuditLedgerResponse(BaseModel):
    """Cryptographically verifiable SHA-256 routing audit ledger response."""

    chain_valid: bool = Field(..., description="Whether the entire SHA-256 hash chain is valid.")
    verified_entries: int = Field(..., description="Total number of verified entries in ledger.")
    head_hash: str = Field(..., description="Latest SHA-256 entry hash at the head of the chain.")
    entries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent tamper-evident routing audit entries.",
    )


class CounterfactualExplainRequest(BaseModel):
    """Payload for counterfactual activation steering and trajectory drift analysis."""

    prompt: str | None = Field(
        default=None,
        description="Single prompt text to evaluate counterfactual steering on.",
    )
    messages: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional multi-turn conversation messages for trajectory drift analysis.",
    )
    target_probability: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Target complexity boundary probability for closed-form gamma*.",
    )


class CounterfactualExplainResponse(BaseModel):
    """White-box counterfactual steering and multi-turn trajectory explanation response."""

    original_probability: float = Field(..., description="Original complexity probability p(h).")
    steered_probability: float = Field(..., description="Probability after minimum boundary flip.")
    minimum_flip_gamma: float = Field(
        ..., description="Exact closed-form steering scalar gamma* to reach target_probability."
    )
    original_tier: str = Field(..., description="Original routed model tier.")
    steered_tier: str = Field(..., description="Routed model tier after counterfactual flip.")
    flipped_decision: bool = Field(..., description="Whether the routing decision flipped.")
    active_sae_features: list[int] = Field(
        default_factory=list, description="Top-K active Sparse Autoencoder feature indices."
    )
    sae_zero_clamp_probability: float = Field(
        ...,
        description="Complexity probability after clamping active Top-K SAE features to zero.",
    )
    trajectory_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Multi-turn conversation trajectory drift metrics when messages are supplied.",
    )


class ActivationCacheStatsResponse(BaseModel):
    """Telemetry response for the Semantic-Mechanistic Activation Cache."""

    size: int = Field(..., description="Current number of cached activation entries.")
    max_entries: int = Field(..., description="Maximum LRU cache capacity.")
    hits: int = Field(..., description="Total triple-verified mechanistic cache hits.")
    misses: int = Field(..., description="Total cache misses.")
    hit_rate: float = Field(..., description="Empirical cache hit rate in [0, 1].")
    circuit_mismatch_rejections: int = Field(
        ..., description="Candidate hits rejected due to SAE circuit or probe complexity mismatch."
    )
    uncertainty_rejections: int = Field(
        ..., description="Queries/insertions bypassed due to high epistemic uncertainty or OOD."
    )
    total_cost_saved_usd: float = Field(
        ..., description="Cumulative USD saved via mechanistic cache hits."
    )
    similarity_threshold: float = Field(
        ..., description="Configured residual-stream cosine similarity threshold."
    )


class BatchPortfolioOptimizeRequest(BaseModel):
    """Payload for global budget-constrained batch portfolio routing optimization."""

    prompts: list[str] = Field(
        ..., min_length=1, description="Batch of prompts to allocate across model tiers."
    )
    total_budget_usd: float = Field(
        ..., ge=0.0, description="Global USD budget constraint for the entire batch."
    )
    max_tier_counts: dict[str, int] | None = Field(
        default=None, description="Optional per-tier maximum capacity limits."
    )


class BatchAllocationItemSchema(BaseModel):
    """Single prompt allocation decision within a portfolio-optimized batch."""

    index: int = Field(..., description="Zero-based prompt index in the batch.")
    prompt: str = Field(..., description="Prompt text.")
    prob_complex: float = Field(
        ..., description="Predicted residual-stream complexity probability."
    )
    selected_model: str = Field(..., description="Assigned target model tier.")
    estimated_cost_usd: float = Field(..., description="Nominal USD cost for the assigned tier.")
    expected_quality: float = Field(..., description="Expected quality score in [0, 1].")


class BatchPortfolioOptimizeResponse(BaseModel):
    """Response from /v1/routing/batch-optimize."""

    allocations: list[BatchAllocationItemSchema] = Field(
        ..., description="Per-prompt portfolio routing allocations."
    )
    total_cost_usd: float = Field(..., description="Total USD cost of the allocated batch.")
    total_budget_usd: float = Field(..., description="Requested global USD budget constraint.")
    oracle_cost_usd: float = Field(..., description="Baseline USD cost if all routed to oracle.")
    cost_saved_usd: float = Field(..., description="Total USD saved relative to oracle routing.")
    mean_expected_quality: float = Field(..., description="Mean expected quality across the batch.")
    tier_distribution: dict[str, int] = Field(
        ..., description="Number of prompts allocated to each model tier."
    )
    budget_feasible: bool = Field(
        ..., description="Whether total_budget_usd was sufficient to cover baseline tier cost."
    )


class DriftStatusResponse(BaseModel):
    """Response from /v1/routing/drift-status reporting Kernel MMD population drift."""

    drift_detected: bool = Field(
        ..., description="Whether statistically significant activation drift was detected."
    )
    mmd_squared: float = Field(
        ..., description="Unbiased multi-bandwidth RBF Kernel MMD^2 statistic."
    )
    p_value: float = Field(..., description="Permutation test p-value in [0, 1].")
    threshold_mmd_squared: float = Field(
        ..., description="Critical MMD^2 threshold at significance level alpha."
    )
    window_size: int = Field(
        ..., description="Number of recent production activations in the sliding window."
    )
    reference_size: int = Field(..., description="Number of reference calibration activations.")
    recommended_alpha_multiplier: float = Field(
        ..., description="Recommended conformal risk tightening multiplier in (0, 1]."
    )
    effective_alpha: float | None = Field(
        default=None, description="Current effective conformal alpha after optional auto-tighten."
    )


class DAGNodeSchema(BaseModel):
    """Single agent node specification for /v1/routing/dag-plan."""

    node_id: str = Field(..., min_length=1, description="Unique identifier for the agent node.")
    prompt: str = Field(..., min_length=1, description="Prompt text executed by this agent node.")
    dependencies: list[str] = Field(
        default_factory=list, description="Parent node_ids that must complete before this node."
    )
    weight: float = Field(default=1.0, gt=0.0, description="Task importance weight > 0.")


class DAGPlanRequest(BaseModel):
    """Payload for multi-agent DAG topological routing and Shapley attribution."""

    nodes: list[DAGNodeSchema] = Field(
        ..., min_length=1, description="Nodes of the multi-agent Directed Acyclic Graph."
    )
    global_conformal_alpha: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="Global workflow conformal risk budget alpha_dag in (0, 1).",
    )


class DAGNodeAllocationSchema(BaseModel):
    """Per-node routing allocation and Shapley credit within a multi-agent DAG plan."""

    node_id: str = Field(..., description="Agent node identifier.")
    topological_order: int = Field(..., description="Topological execution index.")
    prob_complex: float = Field(
        ..., description="Predicted residual-stream complexity probability."
    )
    downstream_criticality: float = Field(
        ..., description="Normalized structural downstream criticality weight."
    )
    allocated_conformal_alpha: float = Field(
        ..., description="Node-specific conformal risk budget alpha_u."
    )
    selected_model: str = Field(..., description="Selected target model tier.")
    estimated_cost_usd: float = Field(..., description="Estimated USD cost for this node.")
    expected_quality: float = Field(..., description="Expected quality score in [0, 1].")
    shapley_credit: float = Field(
        ..., description="Exact cooperative Shapley value credit phi_u for workflow quality."
    )


class DAGPlanResponse(BaseModel):
    """Response from /v1/routing/dag-plan."""

    node_allocations: list[DAGNodeAllocationSchema] = Field(
        ..., description="Topologically ordered node routing allocations."
    )
    topological_order: list[str] = Field(..., description="Execution order of node_ids.")
    global_conformal_alpha: float = Field(..., description="Requested workflow risk budget.")
    total_cost_usd: float = Field(..., description="Total USD cost across all DAG nodes.")
    oracle_cost_usd: float = Field(..., description="Baseline USD cost if all routed to oracle.")
    cost_saved_usd: float = Field(..., description="Total USD saved relative to oracle routing.")
    workflow_expected_quality: float = Field(
        ..., description="Total expected workflow quality (equal to sum of Shapley credits)."
    )


class SafetyEvaluateRequest(BaseModel):
    """Payload for SecOps residual-stream safety evaluation and orthogonal subspace ablation."""

    prompt: str = Field(..., min_length=1, description="Prompt text to inspect and sanitize.")
    benign_calibration_prompts: list[str] | None = Field(
        default=None, description="Optional benign prompts to calibrate guardrail on the fly."
    )
    harmful_calibration_prompts: list[str] | None = Field(
        default=None, description="Optional adversarial prompts to calibrate guardrail on the fly."
    )
    false_positive_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Conformal false-positive budget on benign traffic.",
    )


class SafetyEvaluateResponse(BaseModel):
    """Response from /v1/routing/safety-evaluate."""

    is_unsafe: bool = Field(..., description="Whether prompt exceeded conformal safety threshold.")
    safety_score: float = Field(..., description="Composite safety non-conformity score in [0, 1].")
    threshold: float = Field(..., description="Calibrated conformal safety threshold.")
    direction_projection: float = Field(
        ..., description="Signed projection onto the primary refusal direction v_harm."
    )
    subspace_energy_ratio: float = Field(
        ..., description="Fraction of residual energy lying in the SVD jailbreak subspace."
    )
    original_complexity_prob: float = Field(
        ..., description="Probe complexity probability prior to subspace ablation."
    )
    ablated_complexity_prob: float = Field(
        ..., description="Probe complexity probability after orthogonal harm subspace ablation."
    )
    ablated_safety_score: float = Field(
        ..., description="Safety non-conformity score after orthogonal harm subspace ablation."
    )
    original_tier: str = Field(..., description="Routed tier before subspace ablation.")
    ablated_tier: str = Field(..., description="Routed tier after subspace ablation.")


class EarlyExitEvaluateRequest(BaseModel):
    """Payload for layerwise depth-adaptive early-exit probing."""

    prompt: str = Field(..., min_length=1, description="Prompt text to evaluate across layers.")
    entropy_threshold: float = Field(
        default=0.38,
        gt=0.0,
        le=1.0,
        description="Maximum predictive Shannon entropy allowed to trigger early exit.",
    )
    patience: int = Field(
        default=1, ge=1, description="Consecutive stable layers required to halt early."
    )


class EarlyExitEvaluateResponse(BaseModel):
    """Response from /v1/routing/early-exit."""

    exit_layer_index: int = Field(..., description="Zero-based layer index where probing halted.")
    total_layers: int = Field(..., description="Total number of probe layers evaluated.")
    halted_early: bool = Field(..., description="Whether probing halted prior to the final layer.")
    prob_complex: float = Field(..., description="Complexity probability at the exit layer.")
    epistemic_uncertainty: float = Field(
        ..., description="Normalized binary Shannon entropy at the exit layer."
    )
    compute_saved_ratio: float = Field(
        ..., description="Fraction of deeper layers skipped in [0, 1)."
    )
    layer_probabilities: list[float] = Field(
        ..., description="Per-layer complexity probabilities up to exit_layer_index."
    )
    inter_layer_cosine_similarity: float = Field(
        ..., description="Cosine similarity between exit layer and preceding layer."
    )
    selected_model: str = Field(..., description="Target model tier selected at exit layer.")


class KVCachePlanRequest(BaseModel):
    """Payload for mechanistic KV-cache budget and Heavy-Hitter token eviction planning."""

    prompt: str = Field(
        ..., min_length=1, description="Prompt text to compute KV-cache retention budget."
    )
    num_sink_tokens: int = Field(
        default=2, ge=0, description="Number of initial attention sink tokens to preserve."
    )
    recent_window_size: int = Field(
        default=2, ge=0, description="Number of most recent window tokens to preserve."
    )
    explicit_retention_ratio: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Optional explicit KV retention fraction override in (0, 1].",
    )


class KVCachePlanResponse(BaseModel):
    """Response from /v1/routing/kv-cache-plan."""

    seq_len: int = Field(..., description="Total sequence length evaluated.")
    prob_complex: float = Field(
        ..., description="Predicted residual-stream complexity probability."
    )
    target_retention_ratio: float = Field(
        ..., description="Target KV-cache retention fraction in (0, 1]."
    )
    actual_retention_ratio: float = Field(..., description="Actual fraction of KV tokens retained.")
    num_retained_tokens: int = Field(..., description="Number of retained KV tokens.")
    num_evicted_tokens: int = Field(..., description="Number of evicted KV tokens.")
    sink_token_indices: list[int] = Field(
        ..., description="Indices of preserved initial attention sink tokens."
    )
    recent_window_indices: list[int] = Field(
        ..., description="Indices of preserved local recent window tokens."
    )
    heavy_hitter_indices: list[int] = Field(
        ..., description="Indices of preserved intermediate Heavy-Hitter (H2O) tokens."
    )
    retained_token_indices: list[int] = Field(
        ..., description="Sorted union of all retained token indices."
    )
    evicted_token_indices: list[int] = Field(
        ..., description="Sorted indices of evicted KV tokens."
    )
    retained_attention_mass: float = Field(
        ..., description="Cumulative normalized attention mass retained in [0, 1]."
    )
    perturbation_error_bound: float = Field(
        ..., description="Analytical L1 softmax value perturbation upper bound."
    )
    memory_saved_ratio: float = Field(
        ..., description="Fraction of KV-cache memory saved in [0, 1)."
    )


class RobustnessCertificateRequest(BaseModel):
    """Payload for local Jacobian and Gaussian-smoothing L2 robustness certification."""

    prompt: str = Field(
        ..., min_length=1, description="Prompt text to certify local L2 routing robustness."
    )
    smoothing_sigma: float = Field(
        default=0.15, gt=0.0, description="Gaussian smoothing standard deviation sigma."
    )
    num_smoothing_samples: int = Field(
        default=128,
        ge=8,
        le=2048,
        description="Number of Monte Carlo smoothing samples.",
    )
    use_bilinear_probe: bool = Field(
        default=False,
        description=(
            "Whether to certify using LowRankBilinearProbe (2nd-order Taylor) "
            "instead of LinearActivationProbe."
        ),
    )


class RobustnessCertificateResponse(BaseModel):
    """Response from /v1/routing/robustness-certificate."""

    prob_complex: float = Field(..., description="Complexity probability at unperturbed input.")
    logit_value: float = Field(..., description="Raw probe logit at unperturbed input.")
    selected_tier: str = Field(..., description="Routed model tier at unperturbed input.")
    nearest_boundary_threshold: float = Field(
        ..., description="Closest conformal probability boundary."
    )
    nearest_boundary_logit: float = Field(
        ..., description="Closest conformal boundary in logit space."
    )
    logit_margin: float = Field(..., description="Absolute distance |z(h) - tau_*| in logit space.")
    jacobian_l2_norm: float = Field(
        ..., description="L2 norm of the local input gradient ||nabla_h z(h)||_2."
    )
    hessian_spectral_bound: float = Field(
        ..., description="Spectral norm upper bound L_H of the probe Hessian."
    )
    certified_l2_radius: float = Field(
        ..., description="Analytical 1st/2nd-order Taylor certified L2 perturbation radius."
    )
    relative_l2_radius: float = Field(
        ..., description="Certified L2 radius normalized by activation norm ||h||_2."
    )
    smoothing_sigma: float = Field(..., description="Gaussian noise standard deviation used.")
    empirical_stability_rate: float = Field(
        ..., description="Fraction of smoothed samples preserving selected_tier."
    )
    smoothed_certified_radius: float = Field(
        ..., description="Cohen et al. randomized smoothing certified L2 radius."
    )
    is_certified_stable: bool = Field(
        ..., description="Whether both relative analytical radius and smoothing stability pass."
    )
    worst_case_perturbation_norm: float = Field(
        ..., description="Euclidean norm of minimal boundary-reaching perturbation."
    )
    worst_case_direction_top_dims: list[int] = Field(
        ..., description="Top residual dimensions with highest adversarial sensitivity."
    )


class SLAEProcessStatusResponse(BaseModel):
    """Response from /v1/routing/sla-eprocess reporting anytime-valid betting martingale status."""

    alpha: float = Field(..., description="Target SLA risk budget alpha in (0, 1).")
    significance_delta: float = Field(
        ..., description="Type-I error bound delta for Ville's maximal inequality."
    )
    rejection_threshold: float = Field(..., description="Martingale rejection barrier 1 / delta.")
    num_observations: int = Field(
        ..., description="Total feedback observations processed by the E-process."
    )
    e_value_low: float = Field(..., description="Current betting capital E_t for low threshold.")
    e_value_high: float = Field(..., description="Current betting capital E_t for high threshold.")
    joint_e_value: float = Field(
        ..., description="Convex intersection E-value 0.5 * (E_low + E_high)."
    )
    running_max_e_value: float = Field(..., description="Running supremum max_{0 <= s <= t} E_s.")
    anytime_p_value: float = Field(
        ..., description="Anytime-valid sequential p-value min(1, 1 / max_s E_s)."
    )
    empirical_risk_low: float = Field(..., description="Cumulative empirical risk on low tier.")
    empirical_risk_high: float = Field(..., description="Cumulative empirical risk on high tier.")
    predictable_bet_low: float = Field(
        ..., description="Next predictable betting fraction lambda_{t+1} for low tier."
    )
    predictable_bet_high: float = Field(
        ..., description="Next predictable betting fraction lambda_{t+1} for high tier."
    )
    sla_violated: bool = Field(
        ..., description="Whether current E-value exceeds rejection threshold 1 / delta."
    )
    circuit_breaker_tripped: bool = Field(
        ..., description="Whether anytime supremum has tripped the SLA circuit breaker."
    )


class LoggedRoutingSampleSchema(BaseModel):
    """Single historical routing interaction with bandit feedback for off-policy evaluation."""

    prob_complex: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted complexity probability."
    )
    logged_tier_index: int = Field(
        ..., ge=0, le=2, description="Selected tier index (0=SLM, 1=Mid, 2=Oracle)."
    )
    logging_propensity: float = Field(
        ..., gt=0.0, le=1.0, description="Logging policy propensity pi_0(a_i | x_i)."
    )
    observed_loss: float = Field(
        ..., ge=0.0, le=1.0, description="Observed routing loss in [0, 1]."
    )
    observed_cost_usd: float = Field(
        default=0.02, ge=0.0, description="Observed USD cost for the logged interaction."
    )


class OffPolicyEvaluateRequest(BaseModel):
    """Payload for counterfactual IPS, SNIPS, and Doubly Robust threshold evaluation."""

    candidate_low_threshold: float = Field(
        ..., gt=0.0, lt=1.0, description="Candidate low conformal threshold lambda_low'."
    )
    candidate_high_threshold: float = Field(
        ..., gt=0.0, lt=1.0, description="Candidate high conformal threshold lambda_high'."
    )
    target_sla_alpha: float = Field(
        default=0.10, gt=0.0, lt=1.0, description="Target SLA risk budget alpha in (0, 1)."
    )
    temperature: float = Field(
        default=0.08, ge=0.0, le=1.0, description="Smoothing temperature for candidate policy."
    )
    logged_samples: list[LoggedRoutingSampleSchema] | None = Field(
        default=None,
        description="Optional logged bandit samples; uses calibration corpus if omitted.",
    )


class OffPolicyEvaluateResponse(BaseModel):
    """Response from /v1/routing/off-policy-evaluate."""

    num_samples: int = Field(..., description="Number of logged bandit samples evaluated.")
    candidate_low_threshold: float = Field(..., description="Evaluated low threshold.")
    candidate_high_threshold: float = Field(..., description="Evaluated high threshold.")
    direct_method_risk: float = Field(..., description="Direct Method (DM) estimated risk.")
    ips_risk: float = Field(..., description="Clipped Inverse Propensity Scoring (IPS) risk.")
    snips_risk: float = Field(..., description="Self-Normalized IPS (SNIPS) risk.")
    doubly_robust_risk: float = Field(..., description="Doubly Robust (DR) estimated risk.")
    dr_risk_ci_lower: float = Field(..., description="95% confidence interval lower bound for DR.")
    dr_risk_ci_upper: float = Field(..., description="95% confidence interval upper bound for DR.")
    expected_cost_usd: float = Field(
        ..., description="Expected USD cost per request under candidate thresholds."
    )
    logging_policy_mean_loss: float = Field(
        ..., description="Empirical mean loss under historical logging policy."
    )
    logging_policy_mean_cost_usd: float = Field(
        ..., description="Empirical mean USD cost under historical logging policy."
    )
    effective_sample_size: float = Field(
        ..., description="Kish's Effective Sample Size (ESS) under importance weights."
    )
    ess_ratio: float = Field(..., description="Normalized ESS / N ratio in [0, 1].")
    target_sla_alpha: float = Field(..., description="Target SLA risk budget alpha.")
    safe_to_promote: bool = Field(
        ..., description="Whether candidate passes both DR UCB <= alpha and ESS gate."
    )


class HallucinationCheckRequest(BaseModel):
    """Payload for mechanistic DoLa layer divergence and confabulation subspace check."""

    prompt: str = Field(..., min_length=1, description="Prompt text to inspect for confabulation.")
    false_alarm_alpha: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="Conformal false-alarm budget on grounded factual traffic.",
    )


class HallucinationCheckResponse(BaseModel):
    """Response from /v1/routing/hallucination-check."""

    hallucination_risk_score: float = Field(
        ..., description="Composite mechanistic hallucination risk score in [0, 1]."
    )
    conformal_threshold: float = Field(
        ..., description="Calibrated conformal escalation threshold."
    )
    should_escalate_to_oracle: bool = Field(
        ..., description="Whether prompt should be escalated to LLM-Frontier-Oracle."
    )
    dola_layer_divergence: float = Field(
        ..., description="DoLa premature-vs-mature layer divergence in [0, 1]."
    )
    truthfulness_projection: float = Field(
        ..., description="Normalized confabulation subspace risk projection in [0, 1]."
    )
    epistemic_entropy: float = Field(..., description="Binary Shannon epistemic entropy in [0, 1].")
    num_layers_analyzed: int = Field(..., description="Number of residual layers contrasted.")
    dominant_divergent_layer: int = Field(
        ..., description="Zero-based index of premature layer with peak divergence."
    )
    recommended_model: str = Field(
        ..., description="Recommended target model after hallucination risk gate."
    )


class MultiModalInspectRequest(BaseModel):
    """Payload for multimodal Vision-Language residual probing and FastV/PruMerge pruning."""

    prompt: str = Field(..., min_length=1, description="Text prompt accompanying visual input.")
    num_vision_patches: int = Field(
        default=16, ge=2, le=256, description="Number of visual patch tokens to inspect."
    )
    patch_retention_ratio: float = Field(
        default=0.50,
        gt=0.0,
        le=1.0,
        description="Target fraction of visual patch tokens to retain via FastV/PruMerge.",
    )
    image_complexity_hint: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Visual complexity dispersion hint when synthesizing patch activations.",
    )


class MultiModalInspectResponse(BaseModel):
    """Response from /v1/routing/multimodal-inspect."""

    num_vision_patches: int = Field(..., description="Total initial visual patch tokens.")
    num_text_tokens: int = Field(..., description="Number of text query token blocks.")
    vision_effective_dim: float = Field(
        ..., description="SVD Effective Dimensionality d_eff of visual patches."
    )
    cross_modal_alignment: float = Field(
        ..., description="Mean peak cosine alignment between text and vision patches."
    )
    cross_modal_entropy: float = Field(
        ..., description="Normalized Shannon entropy of text-to-vision cross-attention."
    )
    prob_complex: float = Field(
        ..., description="Combined multimodal task complexity probability in [0, 1]."
    )
    retained_patch_indices: list[int] = Field(
        ..., description="Indices of retained high-attention visual patches."
    )
    pruned_patch_indices: list[int] = Field(
        ..., description="Indices of pruned peripheral visual patches merged via PruMerge."
    )
    patch_retention_ratio: float = Field(
        ..., description="Actual fraction of visual patches retained."
    )
    prumerge_reconstruction_error: float = Field(
        ..., description="Normalized PruMerge context reconstruction error in [0, 1]."
    )
    selected_model: str = Field(..., description="Selected VLM/LLM tier for multimodal request.")


class WassersteinAlignRequest(BaseModel):
    """Payload for Bures-Wasserstein cross-lingual / cross-family subspace alignment."""

    source_prompts: list[str] = Field(
        ...,
        min_length=2,
        description="Unpaired source prompts (e.g., target language or secondary encoder).",
    )
    target_prompts: list[str] | None = Field(
        default=None,
        description=(
            "Optional reference target prompts; defaults to calibration corpus if omitted."
        ),
    )
    eval_prompt: str | None = Field(
        default=None,
        description="Optional prompt to evaluate before and after Bures-Wasserstein alignment.",
    )
    reg_lambda: float = Field(
        default=1e-3,
        ge=0.0,
        description="Covariance diagonal shrinkage regularization lambda_reg.",
    )


class WassersteinAlignResponse(BaseModel):
    """Response from /v1/routing/wasserstein-align."""

    wasserstein_distance: float = Field(
        ..., description="Closed-form L2 Bures-Wasserstein distance W_2."
    )
    mean_shift_l2: float = Field(
        ..., description="Euclidean distance ||mu_s - mu_t||_2 between empirical means."
    )
    bures_covariance_distance: float = Field(
        ..., description="Square root of the Bures trace divergence between covariances."
    )
    covariance_frobenius_residual: float = Field(
        ..., description="Post-alignment Frobenius error ||A Sigma_s A^T - Sigma_t||_F."
    )
    source_condition_number: float = Field(
        ..., description="Spectral condition number of regularized Sigma_s."
    )
    target_condition_number: float = Field(
        ..., description="Spectral condition number of regularized Sigma_t."
    )
    feature_dim: int = Field(..., description="Residual-stream activation dimension D.")
    n_source_samples: int = Field(..., description="Number of source calibration prompts.")
    n_target_samples: int = Field(..., description="Number of target reference prompts.")
    eval_prob_unaligned: float | None = Field(
        default=None, description="Probe complexity probability prior to alignment."
    )
    eval_prob_aligned: float | None = Field(
        default=None, description="Probe complexity probability after Monge transport."
    )
    eval_prob_pullback: float | None = Field(
        default=None, description="Zero-shot pulled-back probe complexity probability."
    )
    selected_model_aligned: str | None = Field(
        default=None, description="Selected model tier after Bures-Wasserstein alignment."
    )
