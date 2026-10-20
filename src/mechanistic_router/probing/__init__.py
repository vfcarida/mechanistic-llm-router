"""Mechanistic Probing Engine Package."""

from .activation_extractor import PrefillActivationExtractor
from .activation_steering import ActivationSteeringEngine, SteeringCounterfactualResult
from .alignment import OrthogonalProcrustesAligner
from .attention_probe import AttentionCircuitProbe, AttentionProbeSignals
from .base import AbstractEncoder, TransformerActivationEncoder
from .bilinear_probe import BilinearInteractionTelemetry, LowRankBilinearProbe
from .causal_validator import CausalAblationHarness, CausalValidationResult
from .conformal import (
    ConformalCalibrationSummary,
    ConformalRiskController,
    compute_epistemic_uncertainty,
)
from .distributed_conformal import (
    AtomicFileConformalBackend,
    ConformalStateBackend,
    ConformalStateSnapshot,
    DistributedConformalCoordinator,
    InMemorySharedConformalBackend,
    merge_conformal_snapshots,
)
from .drift_monitor import DriftTestResult, PopulationDriftMonitor
from .early_exit_probe import EarlyExitDecision, LayerwiseEarlyExitProbe
from .eprocess_monitor import BettingMartingaleMonitor, EProcessStatus
from .federated_conformal import (
    DPClientCalibrationSummary,
    FederatedCalibrationResult,
    FederatedConformalAggregator,
)
from .hallucination_detector import (
    HallucinationRiskReport,
    MechanisticHallucinationDetector,
)
from .kv_cache_policy import KVCacheAllocationPlan, MechanisticKVCachePolicy
from .linear_probe import LinearActivationProbe
from .mondrian_conformal import (
    MondrianCalibrationSummary,
    MondrianConformalController,
    MondrianGroupCalibration,
)
from .multimodal_probe import MultiModalActivationProbe, MultiModalProbeTelemetry
from .nnsight_probe import NNsightProbe
from .ood_detector import MahalanobisOODDetector
from .robustness_certifier import LocalRobustnessCertifier, RobustnessCertificate
from .sae_engine import SAEEngine
from .safety_guardrail import MechanisticSafetyGuardrail, SafetyEvaluationResult
from .trajectory_probe import ConversationTrajectoryAnalyzer, TrajectoryMetrics
from .transformer_lens_hook import TransformerLensHook
from .wasserstein_alignment import BuresWassersteinAligner, WassersteinAlignmentTelemetry

__all__ = [
    "AbstractEncoder",
    "TransformerActivationEncoder",
    "PrefillActivationExtractor",
    "ActivationSteeringEngine",
    "SteeringCounterfactualResult",
    "OrthogonalProcrustesAligner",
    "AttentionCircuitProbe",
    "AttentionProbeSignals",
    "BettingMartingaleMonitor",
    "EProcessStatus",
    "BilinearInteractionTelemetry",
    "LowRankBilinearProbe",
    "BuresWassersteinAligner",
    "WassersteinAlignmentTelemetry",
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
    "PopulationDriftMonitor",
    "DriftTestResult",
    "FederatedConformalAggregator",
    "DPClientCalibrationSummary",
    "FederatedCalibrationResult",
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
    "TransformerLensHook",
    "NNsightProbe",
    "SAEEngine",
]
