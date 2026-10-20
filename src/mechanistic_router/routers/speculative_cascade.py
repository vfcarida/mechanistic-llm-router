"""Mid-Generation Speculative Cascade Hand-Off (Token-Level Entropy Escalation).

Monitors streaming generation from a cheaper tier (`SLM-BERTau-Local` or `LLM-Mid-Tier`)
during the initial prefix window. If token-level negative log-likelihood or linguistic
uncertainty spikes above a calibrated threshold, triggers a seamless speculative
hand-off to the frontier oracle (`LLM-Frontier-Oracle`).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeculativeCascadeResult:
    """Summary of a speculative cascade generation execution."""

    initial_model: str
    final_model: str
    handoff_triggered: bool
    handoff_at_chunk: int | None
    peak_prefix_uncertainty: float
    generated_text: str


class SpeculativeCascadeMonitor:
    """Monitors mid-generation token uncertainty and orchestrates speculative tier hand-off."""

    _EPISTEMIC_FAILURE_MARKERS = (
        "i am not sure",
        "i cannot",
        "unable to solve",
        "beyond my capability",
        "insufficient knowledge",
        "cannot derive",
        "[uncertain]",
        "i don't know",
    )

    def __init__(
        self,
        handoff_threshold: float = 0.65,
        max_prefix_chunks: int = 4,
        nll_scale: float = 2.5,
    ) -> None:
        if not (0.0 < handoff_threshold <= 1.0):
            raise ValueError(f"handoff_threshold must be in (0, 1], got {handoff_threshold}.")
        if max_prefix_chunks < 1:
            raise ValueError(f"max_prefix_chunks must be >= 1, got {max_prefix_chunks}.")
        if nll_scale <= 0.0:
            raise ValueError(f"nll_scale must be > 0, got {nll_scale}.")

        self.handoff_threshold = float(handoff_threshold)
        self.max_prefix_chunks = int(max_prefix_chunks)
        self.nll_scale = float(nll_scale)

    def evaluate_chunk_uncertainty(
        self,
        chunk_text: str,
        token_logprobs: list[float] | None = None,
    ) -> float:
        """Quantifies mid-generation epistemic uncertainty in [0.0, 1.0] for a streamed chunk.

        Combines token-level negative log-likelihood (when logprobs are available) with
        self-reported epistemic failure / hedging markers and token degeneration checks.
        """
        nll_score = 0.0
        if token_logprobs:
            arr = np.asarray(token_logprobs, dtype=np.float64)
            mean_nll = float(np.mean(np.maximum(0.0, -arr)))
            nll_score = float(np.clip(mean_nll / self.nll_scale, 0.0, 1.0))

        lower = chunk_text.lower()
        marker_hits = sum(1 for m in self._EPISTEMIC_FAILURE_MARKERS if m in lower)
        lexical_score = float(np.clip(marker_hits * 0.70, 0.0, 1.0))

        # Detect severe token repetition / degeneration
        words = lower.split()
        repetition_score = 0.0
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35:
                repetition_score = 0.80

        return float(np.clip(max(nll_score, lexical_score, repetition_score), 0.0, 1.0))

    async def execute_with_handoff(
        self,
        prompt: str,
        initial_model: str,
        fallback_model: str,
        stream_provider: Callable[[str, str], AsyncGenerator[tuple[str, list[float] | None], None]],
    ) -> SpeculativeCascadeResult:
        """Executes streaming generation on `initial_model` with hand-off to `fallback_model`.

        Args:
            prompt: User prompt text.
            initial_model: Initially selected cheaper model tier.
            fallback_model: Frontier oracle tier for speculative escalation.
            stream_provider: Async generator callable `(model_name, prompt_text)` yielding
                `(chunk_text, optional_token_logprobs)`.

        Returns:
            SpeculativeCascadeResult detailing whether hand-off occurred and final text.
        """
        collected_chunks: list[str] = []
        peak_uncertainty = 0.0
        handoff_triggered = False
        handoff_at_chunk: int | None = None

        chunk_idx = 0
        async for chunk_text, logprobs in stream_provider(initial_model, prompt):
            chunk_idx += 1
            if chunk_idx <= self.max_prefix_chunks and initial_model != fallback_model:
                u_val = self.evaluate_chunk_uncertainty(chunk_text, logprobs)
                peak_uncertainty = max(peak_uncertainty, u_val)
                if u_val >= self.handoff_threshold:
                    handoff_triggered = True
                    handoff_at_chunk = chunk_idx
                    break
            collected_chunks.append(chunk_text)

        if not handoff_triggered:
            return SpeculativeCascadeResult(
                initial_model=initial_model,
                final_model=initial_model,
                handoff_triggered=False,
                handoff_at_chunk=None,
                peak_prefix_uncertainty=peak_uncertainty,
                generated_text="".join(collected_chunks),
            )

        # Hand-off to fallback frontier oracle conditioning on validated prefix (if any)
        valid_prefix = "".join(collected_chunks)
        continuation_prompt = (
            f"{prompt}\n[Validated Prefix]: {valid_prefix}" if valid_prefix else prompt
        )
        fallback_chunks: list[str] = [valid_prefix] if valid_prefix else []
        async for fb_text, _ in stream_provider(fallback_model, continuation_prompt):
            fallback_chunks.append(fb_text)

        return SpeculativeCascadeResult(
            initial_model=initial_model,
            final_model=fallback_model,
            handoff_triggered=True,
            handoff_at_chunk=handoff_at_chunk,
            peak_prefix_uncertainty=peak_uncertainty,
            generated_text="".join(fallback_chunks),
        )


__all__ = ["SpeculativeCascadeMonitor", "SpeculativeCascadeResult"]
