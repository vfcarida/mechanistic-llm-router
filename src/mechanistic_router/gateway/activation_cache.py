"""Semantic-Mechanistic Activation Cache with SAE Circuit Safety.

Prevents false-positive semantic cache hits by enforcing a triple mechanistic
verification gate over prefill residual-stream representations:
1. Residual-Stream Cosine Ball: cos(h_query, h_cached) >= similarity_threshold.
2. Monosemantic Circuit Agreement: identical Top-K SAE dominant circuit_type
   and matching causal probe binary complexity regime.
3. Epistemic & OOD Safety Gate: high-uncertainty (H(p) > max_uncertainty) or
   out-of-distribution (is_ood=True) queries bypass caching altogether.

References:
- Bang (2023). GPTCache: An Open-Source Semantic Cache for LLM Applications.
- Zhu et al. (NeurIPS 2024). Optimal Caching for Large Language Model Inference.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ActivationCacheEntry:
    """Cached routing decision and optional completion anchored to a prefill vector."""

    key_id: str
    normalized_activation: np.ndarray
    selected_model: str
    estimated_cost_usd: float
    circuit_type: str
    sae_features: list[int]
    probe_predicted_hard: bool
    epistemic_uncertainty: float
    response_content: str | None
    created_at_epoch: float


@dataclass(frozen=True)
class CacheLookupResult:
    """Outcome of a triple-gated mechanistic activation cache lookup."""

    hit: bool
    entry: ActivationCacheEntry | None
    cosine_similarity: float
    rejection_reason: str | None = None


class MechanisticActivationCache:
    """Thread-safe LRU activation cache with SAE circuit and conformal uncertainty gates."""

    def __init__(
        self,
        max_entries: int = 512,
        similarity_threshold: float = 0.985,
        max_uncertainty_threshold: float = 0.99,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}.")
        if not (0.5 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0.5, 1.0], got {similarity_threshold}."
            )
        if not (0.0 < max_uncertainty_threshold <= 1.0):
            raise ValueError(
                f"max_uncertainty_threshold must be in (0, 1], got {max_uncertainty_threshold}."
            )
        if ttl_seconds <= 0.0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}.")

        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.max_uncertainty_threshold = max_uncertainty_threshold
        self.ttl_seconds = ttl_seconds

        self._lock = threading.Lock()
        self._store: OrderedDict[str, ActivationCacheEntry] = OrderedDict()
        self._seq_counter = 0

        self.hits = 0
        self.misses = 0
        self.circuit_mismatch_rejections = 0
        self.uncertainty_rejections = 0
        self.total_cost_saved_usd = 0.0

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            return np.zeros_like(arr)
        return arr / norm

    def put(
        self,
        activation: np.ndarray,
        selected_model: str,
        estimated_cost_usd: float,
        circuit_type: str = "general_syntax_or_factual",
        sae_features: list[int] | None = None,
        complexity_score: float = 0.0,
        epistemic_uncertainty: float = 0.0,
        is_ood: bool = False,
        response_content: str | None = None,
    ) -> bool:
        """Inserts a routing/completion entry if it satisfies epistemic and OOD safety gates."""
        if is_ood or epistemic_uncertainty > self.max_uncertainty_threshold:
            with self._lock:
                self.uncertainty_rejections += 1
            return False

        norm_act = self._normalize(activation)
        if norm_act.size == 0 or float(np.linalg.norm(norm_act)) < 1e-9:
            return False

        with self._lock:
            self._seq_counter += 1
            key_id = f"mcache-{self._seq_counter}"
            entry = ActivationCacheEntry(
                key_id=key_id,
                normalized_activation=norm_act,
                selected_model=selected_model,
                estimated_cost_usd=float(estimated_cost_usd),
                circuit_type=circuit_type,
                sae_features=list(sae_features) if sae_features is not None else [],
                probe_predicted_hard=bool(complexity_score >= 0.5),
                epistemic_uncertainty=float(epistemic_uncertainty),
                response_content=response_content,
                created_at_epoch=time.time(),
            )
            self._store[key_id] = entry
            self._store.move_to_end(key_id)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
            return True

    def lookup(
        self,
        activation: np.ndarray,
        circuit_type: str = "general_syntax_or_factual",
        complexity_score: float = 0.0,
        epistemic_uncertainty: float = 0.0,
        is_ood: bool = False,
    ) -> CacheLookupResult:
        """Searches the cache enforcing cosine similarity, SAE circuit, and probe safety gates."""
        if is_ood or epistemic_uncertainty > self.max_uncertainty_threshold:
            with self._lock:
                self.misses += 1
                self.uncertainty_rejections += 1
            return CacheLookupResult(
                hit=False,
                entry=None,
                cosine_similarity=0.0,
                rejection_reason="epistemic_or_ood_gate",
            )

        norm_query = self._normalize(activation)
        if norm_query.size == 0 or float(np.linalg.norm(norm_query)) < 1e-9:
            with self._lock:
                self.misses += 1
            return CacheLookupResult(
                hit=False,
                entry=None,
                cosine_similarity=0.0,
                rejection_reason="zero_norm_activation",
            )

        now = time.time()
        query_hard = bool(complexity_score >= 0.5)

        with self._lock:
            expired_keys = [
                k for k, v in self._store.items() if (now - v.created_at_epoch) > self.ttl_seconds
            ]
            for k in expired_keys:
                del self._store[k]

            best_sim = -1.0
            best_entry: ActivationCacheEntry | None = None
            saw_circuit_mismatch = False

            for entry in reversed(self._store.values()):
                if entry.normalized_activation.shape != norm_query.shape:
                    continue
                sim = float(np.dot(norm_query, entry.normalized_activation))
                if sim >= self.similarity_threshold:
                    if (
                        entry.circuit_type != circuit_type
                        or entry.probe_predicted_hard != query_hard
                    ):
                        saw_circuit_mismatch = True
                        continue
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_entry is not None:
                self._store.move_to_end(best_entry.key_id)
                self.hits += 1
                self.total_cost_saved_usd += best_entry.estimated_cost_usd
                return CacheLookupResult(
                    hit=True,
                    entry=best_entry,
                    cosine_similarity=best_sim,
                    rejection_reason=None,
                )

            self.misses += 1
            if saw_circuit_mismatch:
                self.circuit_mismatch_rejections += 1
                return CacheLookupResult(
                    hit=False,
                    entry=None,
                    cosine_similarity=max(best_sim, 0.0),
                    rejection_reason="circuit_or_complexity_mismatch",
                )
            return CacheLookupResult(
                hit=False,
                entry=None,
                cosine_similarity=max(best_sim, 0.0),
                rejection_reason="below_similarity_threshold",
            )

    def clear(self) -> None:
        """Clears all cached entries while resetting counters."""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.circuit_mismatch_rejections = 0
            self.uncertainty_rejections = 0
            self.total_cost_saved_usd = 0.0

    def stats(self) -> dict[str, Any]:
        """Returns live cache telemetry and hit-rate statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = float(self.hits) / float(total) if total > 0 else 0.0
            return {
                "size": len(self._store),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "circuit_mismatch_rejections": self.circuit_mismatch_rejections,
                "uncertainty_rejections": self.uncertainty_rejections,
                "total_cost_saved_usd": self.total_cost_saved_usd,
                "similarity_threshold": self.similarity_threshold,
            }


__all__ = [
    "ActivationCacheEntry",
    "CacheLookupResult",
    "MechanisticActivationCache",
]
