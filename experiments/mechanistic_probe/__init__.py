"""Mechanistic Probe Research Spike (MLR-T06).

Extracts real activations from local open-weight models, trains lightweight probes,
validates directionality causally via directional projection/ablation vs random controls,
and benchmarks the probe on the cost-quality Pareto front.
"""

from .activation_extractor import PrefillActivationExtractor
from .causal_validator import CausalAblationHarness, CausalValidationResult
from .policy import CausalProbePolicy
from .probe import LinearActivationProbe

__all__ = [
    "PrefillActivationExtractor",
    "LinearActivationProbe",
    "CausalAblationHarness",
    "CausalValidationResult",
    "CausalProbePolicy",
]
