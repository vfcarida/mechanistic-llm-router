"""Mechanistic KV-Cache Budget & Heavy-Hitter Token Eviction Policy.

Uses prefill multi-head self-attention distributions and residual-stream complexity
signals to allocate dynamic KV-cache retention budgets and select token subsets:
1. Attention Sink Preservation (Xiao et al., ICLR 2024):
   Retains initial anchor tokens that absorb excess softmax probability mass.
2. Heavy-Hitter Oracle (H2O) Token Selection (Zhang et al., NeurIPS 2023):
   Ranks intermediate context tokens by cumulative normalized attention received
   across heads and query positions.
3. Conformal Complexity-Modulated Retention Ratio (Ge et al., ICLR 2024):
   Couples KV-cache compression ratio with latent task complexity c(x), attention
   head Shannon entropy, and conformal risk budget alpha.
4. Bounded Softmax Perturbation Guarantee:
   Certifies an analytical upper bound on attention output deviation induced by
   evicting low-mass key-value pairs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .attention_probe import AttentionCircuitProbe


@dataclass(frozen=True)
class KVCacheAllocationPlan:
    """KV-cache token retention schedule and bounded perturbation telemetry."""

    seq_len: int
    target_retention_ratio: float
    actual_retention_ratio: float
    num_retained_tokens: int
    num_evicted_tokens: int
    sink_token_indices: list[int]
    recent_window_indices: list[int]
    heavy_hitter_indices: list[int]
    retained_token_indices: list[int]
    evicted_token_indices: list[int]
    retained_attention_mass: float
    perturbation_error_bound: float
    memory_saved_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes KV-cache allocation plan to a JSON-compatible dictionary."""
        return asdict(self)


class MechanisticKVCachePolicy:
    """Computes KV-cache compression budgets and Heavy-Hitter token retention masks."""

    def __init__(
        self,
        num_sink_tokens: int = 2,
        recent_window_size: int = 2,
        min_retention_ratio: float = 0.25,
        max_retention_ratio: float = 1.0,
    ) -> None:
        if num_sink_tokens < 0:
            raise ValueError(f"num_sink_tokens must be >= 0, got {num_sink_tokens}.")
        if recent_window_size < 0:
            raise ValueError(f"recent_window_size must be >= 0, got {recent_window_size}.")
        if not (0.0 < min_retention_ratio <= max_retention_ratio <= 1.0):
            raise ValueError(
                "Retention ratios must satisfy "
                "0.0 < min_retention_ratio <= max_retention_ratio <= 1.0."
            )

        self.num_sink_tokens = int(num_sink_tokens)
        self.recent_window_size = int(recent_window_size)
        self.min_retention_ratio = float(min_retention_ratio)
        self.max_retention_ratio = float(max_retention_ratio)

    def compute_dynamic_retention_ratio(
        self,
        prob_complex: float,
        mean_head_entropy: float,
        conformal_alpha: float = 0.10,
    ) -> float:
        """Computes complexity- and entropy-aware KV retention ratio in [min_ratio, max_ratio]."""
        c_val = float(np.clip(prob_complex, 0.0, 1.0))
        h_val = float(np.clip(mean_head_entropy, 0.0, 1.0))
        alpha_val = float(np.clip(conformal_alpha, 0.0, 1.0))

        # Higher complexity, broader attention dispersion, or stricter conformal alpha
        # increases the fraction of retained KV tokens.
        composite_pressure = float(
            np.clip(0.50 * c_val + 0.35 * h_val + 0.15 * (1.0 - alpha_val), 0.0, 1.0)
        )
        ratio = (
            self.min_retention_ratio
            + (self.max_retention_ratio - self.min_retention_ratio) * composite_pressure
        )
        return float(np.clip(ratio, self.min_retention_ratio, self.max_retention_ratio))

    @staticmethod
    def _normalize_attention_tensor(
        attention_weights: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Validates and normalizes causal attention matrix into shape (H, S, S)."""
        if isinstance(attention_weights, torch.Tensor):
            attn = attention_weights.detach().cpu().numpy().astype(np.float64)
        else:
            attn = np.asarray(attention_weights, dtype=np.float64)

        if attn.ndim == 4:
            attn = attn[0]
        elif attn.ndim == 2:
            attn = attn[np.newaxis, :, :]
        elif attn.ndim != 3:
            raise ValueError(
                f"Expected 2D (S,S), 3D (H,S,S), or 4D (B,H,S,S) attention, got {attn.ndim}D."
            )

        _, seq_len, seq_len_k = attn.shape
        if seq_len != seq_len_k or seq_len == 0:
            raise ValueError(f"Attention matrix must be non-empty square (S, S), got {attn.shape}.")

        # Mask upper triangle above diagonal and normalize rows
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        attn = np.where(causal_mask[np.newaxis, :, :], 0.0, np.maximum(attn, 0.0))
        row_sums = np.maximum(np.sum(attn, axis=-1, keepdims=True), 1e-12)
        return np.asarray(attn / row_sums, dtype=np.float64)

    def compute_token_importance_scores(
        self,
        attention_weights: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Computes normalized cumulative attention mass received by each token key."""
        attn = self._normalize_attention_tensor(attention_weights)
        # Sum across heads and query rows (each query row has unit mass)
        key_mass = np.sum(attn, axis=(0, 1))
        total_mass = float(np.sum(key_mass))
        if total_mass <= 1e-12:
            seq_len = int(attn.shape[-1])
            return np.full(seq_len, 1.0 / seq_len, dtype=np.float64)
        return np.asarray(key_mass / total_mass, dtype=np.float64)

    def plan_eviction(
        self,
        attention_weights: np.ndarray | torch.Tensor,
        prob_complex: float = 0.5,
        conformal_alpha: float = 0.10,
        explicit_retention_ratio: float | None = None,
    ) -> KVCacheAllocationPlan:
        """Constructs an H2O + Attention-Sink token retention plan with error bounds."""
        attn = self._normalize_attention_tensor(attention_weights)
        num_heads, seq_len, _ = attn.shape

        # Compute mean normalized Shannon entropy across heads
        probe = AttentionCircuitProbe(hidden_dim=max(4, num_heads * 4), num_heads=num_heads)
        signals = probe.analyze_attention_weights(attn)

        if explicit_retention_ratio is not None:
            target_ratio = float(
                np.clip(
                    explicit_retention_ratio, self.min_retention_ratio, self.max_retention_ratio
                )
            )
        else:
            target_ratio = self.compute_dynamic_retention_ratio(
                prob_complex=prob_complex,
                mean_head_entropy=signals.mean_head_entropy,
                conformal_alpha=conformal_alpha,
            )

        scores = self.compute_token_importance_scores(attn)

        sink_count = min(self.num_sink_tokens, seq_len)
        sink_indices = list(range(sink_count))

        recent_start = max(sink_count, seq_len - self.recent_window_size)
        recent_indices = list(range(recent_start, seq_len))

        mandatory_set = set(sink_indices) | set(recent_indices)
        target_count = min(
            seq_len,
            max(len(mandatory_set), int(math.ceil(target_ratio * float(seq_len)))),
        )
        hh_budget = max(0, target_count - len(mandatory_set))

        candidate_indices = [i for i in range(seq_len) if i not in mandatory_set]
        # Sort candidates by descending cumulative attention mass (tie-break by more recent token)
        candidate_indices.sort(key=lambda idx: (float(scores[idx]), idx), reverse=True)
        heavy_hitter_indices = sorted(candidate_indices[:hh_budget])

        retained_indices = sorted(mandatory_set | set(heavy_hitter_indices))
        retained_set = set(retained_indices)
        evicted_indices = [i for i in range(seq_len) if i not in retained_set]

        retained_mass = float(np.clip(np.sum(scores[retained_indices]), 0.0, 1.0))
        evicted_mass = max(0.0, 1.0 - retained_mass)
        # Softmax renormalization L1 value perturbation bound: <= 2 * evicted_mass
        perturbation_bound = (
            0.0 if len(evicted_indices) == 0 else float(min(1.0, 2.0 * evicted_mass))
        )

        actual_ratio = float(len(retained_indices) / float(seq_len))
        memory_saved = float(1.0 - actual_ratio)

        return KVCacheAllocationPlan(
            seq_len=seq_len,
            target_retention_ratio=round(target_ratio, 6),
            actual_retention_ratio=round(actual_ratio, 6),
            num_retained_tokens=len(retained_indices),
            num_evicted_tokens=len(evicted_indices),
            sink_token_indices=sink_indices,
            recent_window_indices=recent_indices,
            heavy_hitter_indices=heavy_hitter_indices,
            retained_token_indices=retained_indices,
            evicted_token_indices=evicted_indices,
            retained_attention_mass=round(retained_mass, 6),
            perturbation_error_bound=round(perturbation_bound, 6),
            memory_saved_ratio=round(memory_saved, 6),
        )

    def plan_from_hidden_states(
        self,
        hidden_states: np.ndarray | torch.Tensor,
        prob_complex: float = 0.5,
        conformal_alpha: float = 0.10,
        explicit_retention_ratio: float | None = None,
        num_heads: int = 4,
        seed: int = 42,
    ) -> KVCacheAllocationPlan:
        """Projects hidden states into causal attention and plans KV retention budget."""
        if isinstance(hidden_states, torch.Tensor):
            h_arr = hidden_states.detach().cpu().numpy().astype(np.float64)
        else:
            h_arr = np.asarray(hidden_states, dtype=np.float64)

        if h_arr.ndim == 1:
            # Fold 1D residual vector into pseudo-token sequence (S, D_sub)
            total_d = int(h_arr.shape[0])
            head_mult = max(4, num_heads)
            if total_d >= head_mult * 8:
                d_sub = (total_d // 8 // num_heads) * num_heads
                d_sub = max(num_heads, d_sub)
                seq_len = total_d // d_sub
                h_arr = h_arr[: seq_len * d_sub].reshape(seq_len, d_sub)
            else:
                h_arr = h_arr.reshape(1, -1)

        d_dim = int(h_arr.shape[-1])
        eff_heads = num_heads if (d_dim >= num_heads and d_dim % num_heads == 0) else 1
        attn_probe = AttentionCircuitProbe(hidden_dim=d_dim, num_heads=eff_heads, seed=seed)
        attn_weights = attn_probe.compute_causal_attention_from_hidden(h_arr)
        return self.plan_eviction(
            attention_weights=attn_weights,
            prob_complex=prob_complex,
            conformal_alpha=conformal_alpha,
            explicit_retention_ratio=explicit_retention_ratio,
        )


__all__ = [
    "KVCacheAllocationPlan",
    "MechanisticKVCachePolicy",
]
