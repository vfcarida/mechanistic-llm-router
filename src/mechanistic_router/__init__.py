"""Mechanistic LLM Router Package."""

from .config import DEFAULT_CONFIG, RouterConfig
from .core.encoder import SharedTrunkEncoder
from .evaluation.calibration_pack import (
    DeterministicMultiLayerExtractor,
    build_and_save_calibration_pack,
    load_calibrated_causal_router,
    load_multidomain_calibration_dataset,
)
from .evaluation.off_policy_evaluator import (
    DoublyRobustPolicyEvaluator,
    LoggedRoutingSample,
    OffPolicyEvaluationResult,
)
from .gateway.activation_cache import (
    ActivationCacheEntry,
    CacheLookupResult,
    MechanisticActivationCache,
)
from .gateway.dispatcher import LiteLLMDispatcher
from .observability.audit_log import RoutingAuditEntry, RoutingAuditLedger
from .observability.metrics import RouterMetrics
from .probing.activation_extractor import PrefillActivationExtractor
from .probing.activation_steering import ActivationSteeringEngine, SteeringCounterfactualResult
from .probing.alignment import OrthogonalProcrustesAligner
from .probing.attention_probe import AttentionCircuitProbe, AttentionProbeSignals
from .probing.base import AbstractEncoder, TransformerActivationEncoder
from .probing.bilinear_probe import BilinearInteractionTelemetry, LowRankBilinearProbe
from .probing.causal_validator import CausalAblationHarness, CausalValidationResult
from .probing.conformal import (
    ConformalCalibrationSummary,
    ConformalRiskController,
    compute_epistemic_uncertainty,
)
from .probing.distributed_conformal import (
    AtomicFileConformalBackend,
    ConformalStateBackend,
    ConformalStateSnapshot,
    DistributedConformalCoordinator,
    InMemorySharedConformalBackend,
    merge_conformal_snapshots,
)
from .probing.drift_monitor import DriftTestResult, PopulationDriftMonitor
from .probing.early_exit_probe import EarlyExitDecision, LayerwiseEarlyExitProbe
from .probing.eprocess_monitor import BettingMartingaleMonitor, EProcessStatus
from .probing.federated_conformal import (
    DPClientCalibrationSummary,
    FederatedCalibrationResult,
    FederatedConformalAggregator,
)
from .probing.hallucination_detector import (
    HallucinationRiskReport,
    MechanisticHallucinationDetector,
)
from .probing.kv_cache_policy import KVCacheAllocationPlan, MechanisticKVCachePolicy
from .probing.linear_probe import LinearActivationProbe
from .probing.mondrian_conformal import (
    MondrianCalibrationSummary,
    MondrianConformalController,
    MondrianGroupCalibration,
)
from .probing.multimodal_probe import MultiModalActivationProbe, MultiModalProbeTelemetry
from .probing.ood_detector import MahalanobisOODDetector
from .probing.robustness_certifier import LocalRobustnessCertifier, RobustnessCertificate
from .probing.sae_engine import SAEEngine
from .probing.safety_guardrail import MechanisticSafetyGuardrail, SafetyEvaluationResult
from .probing.trajectory_probe import ConversationTrajectoryAnalyzer, TrajectoryMetrics
from .probing.wasserstein_alignment import BuresWassersteinAligner, WassersteinAlignmentTelemetry
from .routers.causal_probe import CausalProbeRouter
from .routers.cost_performance import CostPerformanceRouter
from .routers.dag_router import (
    AgentDAGNode,
    AgentDAGRoutingPlan,
    AgentNodeRoutingAllocation,
    TopologicalDAGRouter,
)
from .routers.linucb_probe import LinUCBActivationRouter
from .routers.mechanistic import MechanisticRouter, SimulatedRouter
from .routers.portfolio_optimizer import (
    BatchAllocationItem,
    BatchPortfolioResult,
    BatchPortfolioRouter,
    PrimalDualBudgetController,
    compute_expected_tier_quality,
)
from .routers.semantic import SemanticRouter
from .routers.speculative_cascade import SpeculativeCascadeMonitor, SpeculativeCascadeResult

__version__ = "0.3.0"
__author__ = "Vinicius Caridá"
__email__ = "vfcarida@gmail.com"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "RouterConfig",
    "DEFAULT_CONFIG",
    "AbstractEncoder",
    "SharedTrunkEncoder",
    "TransformerActivationEncoder",
    "MechanisticRouter",
    "SimulatedRouter",
    "CostPerformanceRouter",
    "SemanticRouter",
    "CausalProbeRouter",
    "LinUCBActivationRouter",
    "AgentDAGNode",
    "AgentDAGRoutingPlan",
    "AgentNodeRoutingAllocation",
    "TopologicalDAGRouter",
    "BatchAllocationItem",
    "BatchPortfolioResult",
    "BatchPortfolioRouter",
    "PrimalDualBudgetController",
    "compute_expected_tier_quality",
    "SpeculativeCascadeMonitor",
    "SpeculativeCascadeResult",
    "BettingMartingaleMonitor",
    "EProcessStatus",
    "BilinearInteractionTelemetry",
    "LowRankBilinearProbe",
    "BuresWassersteinAligner",
    "WassersteinAlignmentTelemetry",
    "DoublyRobustPolicyEvaluator",
    "LoggedRoutingSample",
    "OffPolicyEvaluationResult",
    "EarlyExitDecision",
    "LayerwiseEarlyExitProbe",
    "HallucinationRiskReport",
    "MechanisticHallucinationDetector",
    "KVCacheAllocationPlan",
    "MechanisticKVCachePolicy",
    "LinearActivationProbe",
    "LocalRobustnessCertifier",
    "RobustnessCertificate",
    "MondrianCalibrationSummary",
    "MondrianConformalController",
    "MondrianGroupCalibration",
    "MultiModalActivationProbe",
    "MultiModalProbeTelemetry",
    "MahalanobisOODDetector",
    "MechanisticSafetyGuardrail",
    "SafetyEvaluationResult",
    "ConversationTrajectoryAnalyzer",
    "TrajectoryMetrics",
    "ActivationSteeringEngine",
    "SteeringCounterfactualResult",
    "AttentionCircuitProbe",
    "AttentionProbeSignals",
    "PopulationDriftMonitor",
    "DriftTestResult",
    "FederatedConformalAggregator",
    "DPClientCalibrationSummary",
    "FederatedCalibrationResult",
    "MechanisticActivationCache",
    "ActivationCacheEntry",
    "CacheLookupResult",
    "PrefillActivationExtractor",
    "OrthogonalProcrustesAligner",
    "DeterministicMultiLayerExtractor",
    "CausalAblationHarness",
    "CausalValidationResult",
    "ConformalRiskController",
    "ConformalCalibrationSummary",
    "compute_epistemic_uncertainty",
    "AtomicFileConformalBackend",
    "ConformalStateBackend",
    "ConformalStateSnapshot",
    "DistributedConformalCoordinator",
    "InMemorySharedConformalBackend",
    "merge_conformal_snapshots",
    "load_multidomain_calibration_dataset",
    "build_and_save_calibration_pack",
    "load_calibrated_causal_router",
    "LiteLLMDispatcher",
    "RouterMetrics",
    "RoutingAuditEntry",
    "RoutingAuditLedger",
    "SAEEngine",
]
