"""Internal Prompt Complexity Estimation Heuristics."""

import functools

from ..models.types import TaskComplexity

ComplexityTier = TaskComplexity


@functools.lru_cache(maxsize=4096)
def estimate_complexity(prompt: str) -> ComplexityTier:
    """Estimates the task complexity tier purely from the raw prompt text.

    This internal heuristic extracts surface and lexical features (word length,
    syntactic indicators, and domain keywords) to classify prompts into
    ComplexityTier (ROUTINE, MODERATE, or COMPLEX).

    Note: This is an internal heuristic used by router strategies to guide
    simulated competence allocation without caller-injected ground-truth labels.

    Args:
        prompt: Raw user prompt string.

    Returns:
        ComplexityTier: Inferred complexity classification (ROUTINE, MODERATE, or COMPLEX).
    """
    if not prompt or not isinstance(prompt, str):
        return ComplexityTier.ROUTINE

    clean = prompt.lower().strip()
    words = clean.split()
    word_count = len(words)

    complex_markers = (
        "dti",
        "ltv",
        "risk",
        "credit score",
        "credit bureau",
        "diversification",
        "macro",
        "compound interest",
        "loss ratio",
        "deductible",
        "projection",
        "cash flow",
        "apr",
        "strategy",
        "correlat",
        "net worth",
        "payment priority",
        "conditional",
        "comprehensive analysis",
        "complex",
        "mathematical",
        "derivative",
        "portfolio",
        "stochastic",
        "covenant",
        "arbitrage",
    )

    moderate_markers = (
        "installment",
        "declined",
        "why",
        "difference",
        "eligibility",
        "unrecognized",
        "do not recognize",
        "coverage",
        "options",
        "verify",
        "home insurance",
        "breakdown",
        "electrical damage",
        "moderate",
        "explain",
    )

    # Multi-step or high-density reasoning heuristics
    if word_count > 30 or any(marker in clean for marker in complex_markers):
        return ComplexityTier.COMPLEX

    # Single-step inferential or explanatory queries
    if word_count > 10 or any(marker in clean for marker in moderate_markers):
        return ComplexityTier.MODERATE

    return ComplexityTier.ROUTINE
