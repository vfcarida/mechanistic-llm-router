"""Attention-Head Routing Entropy & Induction Circuit Probing.

Extracts white-box structural signals from Transformer multi-head self-attention
distributions A in R^{B x H x S x S}, complementing residual-stream hidden state
probes with attention-head specialization metrics:
1. Normalized Shannon Attention Entropy (per-head & population dispersion).
2. Attention Sink Concentration on initial token k = 0 (Xiao et al., ICLR 2024).
3. Previous-Token Head Score on sub-diagonal k = t - 1 (Olsson et al., 2022).
4. Induction-Head Copy / Non-Local Retrieval Score (Olsson et al., 2022).

References:
- Olsson et al. (2022). In-Context Learning and Induction Heads. Transformer Circuits.
- Voita et al. (ACL 2019). Analyzing Multi-Head Self-Attention.
- Xiao et al. (ICLR 2024). Efficient Streaming Language Models with Attention Sinks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AttentionProbeSignals:
    """Mechanistic telemetry extracted from multi-head self-attention patterns."""

    mean_head_entropy: float
    head_entropy_std: float
    attention_sink_ratio: float
    previous_token_score: float
    induction_head_score: float
    max_induction_head_idx: int
    structural_complexity_score: float
    num_heads: int
    seq_len: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes attention probe signals to a JSON-compatible dictionary."""
        return asdict(self)


class AttentionCircuitProbe:
    """Analyzes multi-head self-attention patterns and induction head circuits."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        seed: int = 42,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.seed = seed

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(float(hidden_dim))
        self._w_q = rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float64)
        self._w_k = rng.normal(0.0, scale, size=(hidden_dim, hidden_dim)).astype(np.float64)

    def compute_causal_attention_from_hidden(
        self,
        hidden_states: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Projects sequence hidden states (S, D) or (B, S, D) into causal attention (H, S, S)."""
        if isinstance(hidden_states, torch.Tensor):
            h_arr = hidden_states.detach().cpu().numpy().astype(np.float64)
        else:
            h_arr = np.asarray(hidden_states, dtype=np.float64)

        if h_arr.ndim == 3:
            h_arr = h_arr[0]
        elif h_arr.ndim == 1:
            h_arr = h_arr.reshape(1, -1)
        elif h_arr.ndim != 2:
            raise ValueError(f"Expected 1D, 2D, or 3D hidden_states, got {h_arr.ndim}D.")

        seq_len, d = h_arr.shape
        if d != self.hidden_dim:
            # Adapt projection deterministically if hidden_dim differs
            rng = np.random.default_rng(self.seed)
            scale = 1.0 / np.sqrt(float(d))
            w_q = rng.normal(0.0, scale, size=(d, self.hidden_dim))
            w_k = rng.normal(0.0, scale, size=(d, self.hidden_dim))
        else:
            w_q = self._w_q
            w_k = self._w_k

        q = (h_arr @ w_q).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = (h_arr @ w_k).reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        logits = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(float(self.head_dim))
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        logits[:, causal_mask] = -1e9

        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        exp_logits[:, causal_mask] = 0.0
        denom = np.maximum(np.sum(exp_logits, axis=-1, keepdims=True), 1e-12)
        return np.asarray(exp_logits / denom, dtype=np.float64)

    def analyze_attention_weights(
        self,
        attention_weights: np.ndarray | torch.Tensor,
        token_ids: list[int] | np.ndarray | torch.Tensor | None = None,
    ) -> AttentionProbeSignals:
        """Computes attention entropy, sink concentration, and induction-head scores.

        Args:
            attention_weights: Tensor/array of shape (B, H, S, S), (H, S, S), or (S, S).
            token_ids: Optional 1D sequence of token IDs of length S for exact prefix matching.

        Returns:
            AttentionProbeSignals summarizing multi-head attention dynamics.
        """
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

        num_heads, seq_len, seq_len_k = attn.shape
        if seq_len != seq_len_k or seq_len == 0:
            raise ValueError(f"Attention matrix must be non-empty square (S, S), got {attn.shape}.")

        if seq_len == 1:
            return AttentionProbeSignals(
                mean_head_entropy=0.0,
                head_entropy_std=0.0,
                attention_sink_ratio=1.0,
                previous_token_score=0.0,
                induction_head_score=0.0,
                max_induction_head_idx=0,
                structural_complexity_score=0.0,
                num_heads=num_heads,
                seq_len=1,
            )

        # Normalize rows over causal triangle to guard against unnormalized inputs
        row_sums = np.maximum(np.sum(attn, axis=-1, keepdims=True), 1e-12)
        attn = np.clip(attn / row_sums, 0.0, 1.0)

        head_entropies = np.zeros(num_heads, dtype=np.float64)
        head_sink_ratios = np.zeros(num_heads, dtype=np.float64)
        head_prev_scores = np.zeros(num_heads, dtype=np.float64)
        head_induction_scores = np.zeros(num_heads, dtype=np.float64)

        tokens_arr: np.ndarray | None = None
        if token_ids is not None:
            if isinstance(token_ids, torch.Tensor):
                tokens_arr = token_ids.detach().cpu().numpy().ravel()
            else:
                tokens_arr = np.asarray(token_ids).ravel()

        for h in range(num_heads):
            ent_list: list[float] = []
            prev_list: list[float] = []
            sink_list: list[float] = []
            ind_list: list[float] = []

            for t in range(1, seq_len):
                row = attn[h, t, : t + 1]
                row_safe = np.maximum(row, 1e-12)
                raw_ent = float(-np.sum(row * np.log(row_safe)))
                max_ent = float(np.log(float(t + 1)))
                ent_list.append(raw_ent / max(max_ent, 1e-12))

                sink_list.append(float(row[0]))
                prev_list.append(float(row[t - 1]))

                if tokens_arr is not None and tokens_arr.shape[0] >= seq_len:
                    # Exact Olsson et al. prefix-matching induction score:
                    # Find prior occurrences p < t - 1 where token[p] == token[t]
                    prior_matches = [
                        p + 1 for p in range(t - 1) if int(tokens_arr[p]) == int(tokens_arr[t])
                    ]
                    if prior_matches:
                        ind_list.append(float(np.sum(row[prior_matches])))
                if not ind_list or (tokens_arr is None or tokens_arr.shape[0] < seq_len):
                    # Non-local structural retrieval score: peak attention on [1, t - 2]
                    if t >= 3:
                        ind_list.append(float(np.max(row[1 : t - 1])))

            head_entropies[h] = float(np.mean(ent_list)) if ent_list else 0.0
            head_sink_ratios[h] = float(np.mean(sink_list)) if sink_list else 1.0
            head_prev_scores[h] = float(np.mean(prev_list)) if prev_list else 0.0
            head_induction_scores[h] = float(np.mean(ind_list)) if ind_list else 0.0

        mean_entropy = float(np.clip(np.mean(head_entropies), 0.0, 1.0))
        entropy_std = float(np.std(head_entropies))
        sink_ratio = float(np.clip(np.mean(head_sink_ratios), 0.0, 1.0))
        prev_score = float(np.clip(np.mean(head_prev_scores), 0.0, 1.0))
        max_ind_idx = int(np.argmax(head_induction_scores))
        induction_score = float(np.clip(head_induction_scores[max_ind_idx], 0.0, 1.0))

        # Structural complexity rises with induction activity and head specialization
        structural_complexity = float(
            np.clip(
                0.45 * induction_score
                + 0.30 * (1.0 - sink_ratio)
                + 0.25 * min(entropy_std * 3.0, 1.0),
                0.0,
                1.0,
            )
        )

        return AttentionProbeSignals(
            mean_head_entropy=mean_entropy,
            head_entropy_std=entropy_std,
            attention_sink_ratio=sink_ratio,
            previous_token_score=prev_score,
            induction_head_score=induction_score,
            max_induction_head_idx=max_ind_idx,
            structural_complexity_score=structural_complexity,
            num_heads=num_heads,
            seq_len=seq_len,
        )

    def analyze_hidden_states(
        self,
        hidden_states: np.ndarray | torch.Tensor,
        token_ids: list[int] | np.ndarray | torch.Tensor | None = None,
    ) -> AttentionProbeSignals:
        """Computes causal multi-head attention from sequence hidden states and extracts signals."""
        attn = self.compute_causal_attention_from_hidden(hidden_states)
        return self.analyze_attention_weights(attn, token_ids=token_ids)


__all__ = [
    "AttentionCircuitProbe",
    "AttentionProbeSignals",
]
