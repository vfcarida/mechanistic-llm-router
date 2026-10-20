"""Tamper-Evident SHA-256 Hash-Chained Routing Audit Ledger for AI TRiSM Governance.

Provides a privacy-preserving, append-only cryptographic hash chain recording every
routing decision (`prompt_sha256`, `selected_model`, `strategy_used`, `conformal_alpha`,
`sae_features`, `prev_hash`, `entry_hash`) in compliance with EU AI Act Article 12
and Gartner AI TRiSM auditability requirements.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class RoutingAuditEntry:
    """Single immutable record in the SHA-256 hash-chained routing audit ledger."""

    seq_id: int
    timestamp_utc: str
    prompt_sha256: str
    selected_model: str
    strategy_used: str
    estimated_cost_usd: float
    cost_saved_usd: float
    conformal_alpha: float | None
    sae_features: list[int]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serializes the audit entry to a JSON-compatible dictionary."""
        return {
            "seq_id": self.seq_id,
            "timestamp_utc": self.timestamp_utc,
            "prompt_sha256": self.prompt_sha256,
            "selected_model": self.selected_model,
            "strategy_used": self.strategy_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_saved_usd": self.cost_saved_usd,
            "conformal_alpha": self.conformal_alpha,
            "sae_features": list(self.sae_features),
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


def compute_audit_entry_hash(
    seq_id: int,
    timestamp_utc: str,
    prompt_sha256: str,
    selected_model: str,
    strategy_used: str,
    estimated_cost_usd: float,
    cost_saved_usd: float,
    conformal_alpha: float | None,
    sae_features: list[int],
    prev_hash: str,
) -> str:
    """Computes canonical SHA-256 digest over an audit entry's fields and predecessor hash."""
    canonical_payload = json.dumps(
        {
            "seq_id": seq_id,
            "timestamp_utc": timestamp_utc,
            "prompt_sha256": prompt_sha256,
            "selected_model": selected_model,
            "strategy_used": strategy_used,
            "estimated_cost_usd": round(float(estimated_cost_usd), 8),
            "cost_saved_usd": round(float(cost_saved_usd), 8),
            "conformal_alpha": (
                round(float(conformal_alpha), 8) if conformal_alpha is not None else None
            ),
            "sae_features": [int(f) for f in sae_features],
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


class RoutingAuditLedger:
    """Thread-safe append-only SHA-256 hash-chained audit ledger for routing telemetry."""

    def __init__(self, genesis_hash: str = GENESIS_HASH) -> None:
        self.genesis_hash = genesis_hash
        self._lock = threading.Lock()
        self._entries: list[RoutingAuditEntry] = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def head_hash(self) -> str:
        """Returns the latest entry_hash in the chain, or genesis_hash if empty."""
        with self._lock:
            if not self._entries:
                return self.genesis_hash
            return self._entries[-1].entry_hash

    def append_decision(
        self,
        prompt: str,
        selected_model: str,
        strategy_used: str,
        estimated_cost_usd: float,
        cost_saved_usd: float,
        conformal_alpha: float | None = None,
        sae_features: list[int] | None = None,
    ) -> RoutingAuditEntry:
        """Records a routing decision as a cryptographically chained audit entry."""
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        features = [int(f) for f in (sae_features or [])]
        ts = datetime.now(timezone.utc).isoformat()

        with self._lock:
            seq_id = len(self._entries) + 1
            prev_hash = self._entries[-1].entry_hash if self._entries else self.genesis_hash
            entry_hash = compute_audit_entry_hash(
                seq_id=seq_id,
                timestamp_utc=ts,
                prompt_sha256=prompt_digest,
                selected_model=selected_model,
                strategy_used=strategy_used,
                estimated_cost_usd=estimated_cost_usd,
                cost_saved_usd=cost_saved_usd,
                conformal_alpha=conformal_alpha,
                sae_features=features,
                prev_hash=prev_hash,
            )
            entry = RoutingAuditEntry(
                seq_id=seq_id,
                timestamp_utc=ts,
                prompt_sha256=prompt_digest,
                selected_model=selected_model,
                strategy_used=strategy_used,
                estimated_cost_usd=round(float(estimated_cost_usd), 8),
                cost_saved_usd=round(float(cost_saved_usd), 8),
                conformal_alpha=conformal_alpha,
                sae_features=features,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )
            self._entries.append(entry)
            return entry

    def verify_integrity(self) -> tuple[bool, int]:
        """Verifies cryptographic hash chain continuity and payload integrity across all entries."""
        with self._lock:
            expected_prev = self.genesis_hash
            for idx, entry in enumerate(self._entries):
                if entry.seq_id != idx + 1:
                    return False, idx
                if entry.prev_hash != expected_prev:
                    return False, idx
                recomputed = compute_audit_entry_hash(
                    seq_id=entry.seq_id,
                    timestamp_utc=entry.timestamp_utc,
                    prompt_sha256=entry.prompt_sha256,
                    selected_model=entry.selected_model,
                    strategy_used=entry.strategy_used,
                    estimated_cost_usd=entry.estimated_cost_usd,
                    cost_saved_usd=entry.cost_saved_usd,
                    conformal_alpha=entry.conformal_alpha,
                    sae_features=entry.sae_features,
                    prev_hash=entry.prev_hash,
                )
                if entry.entry_hash != recomputed:
                    return False, idx
                expected_prev = entry.entry_hash
            return True, len(self._entries)

    def export_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        """Exports the most recent `limit` audit entries as dictionaries."""
        with self._lock:
            if limit <= 0:
                return []
            return [e.to_dict() for e in self._entries[-limit:]]


__all__ = [
    "GENESIS_HASH",
    "RoutingAuditEntry",
    "RoutingAuditLedger",
    "compute_audit_entry_hash",
]
