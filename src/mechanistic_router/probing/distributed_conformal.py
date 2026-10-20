"""Distributed Multi-Replica Conformal State Synchronization (CRDT / Shared State).

Enables horizontal scaling of `CausalProbeRouter` across multiple worker replicas
by synchronizing Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021) online
risk counters (`online_steps`, `cumulative_loss_low`, `cumulative_loss_high`) via a
commutative and associative state merger.
"""

from __future__ import annotations

import abc
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .conformal import ConformalRiskController


@dataclass(frozen=True)
class ConformalStateSnapshot:
    """Immutable serializable snapshot of a ConformalRiskController's online state."""

    alpha: float
    max_loss_bound: float
    learning_rate: float
    base_low_threshold: float
    base_high_threshold: float
    low_threshold: float
    high_threshold: float
    effective_alpha: float
    online_steps: int
    cumulative_loss_low: float
    cumulative_loss_high: float
    replica_id: str = "default"

    @classmethod
    def from_controller(
        cls,
        controller: ConformalRiskController,
        replica_id: str = "default",
    ) -> ConformalStateSnapshot:
        """Extracts a ConformalStateSnapshot from a calibrated ConformalRiskController."""
        if not controller.is_calibrated:
            raise ValueError("Cannot snapshot an uncalibrated ConformalRiskController.")

        summary = controller.summary
        return cls(
            alpha=controller.alpha,
            max_loss_bound=controller.max_loss_bound,
            learning_rate=controller.learning_rate,
            base_low_threshold=summary.low_threshold,
            base_high_threshold=summary.high_threshold,
            low_threshold=controller.low_threshold,
            high_threshold=controller.high_threshold,
            effective_alpha=controller.effective_alpha,
            online_steps=controller.online_steps,
            cumulative_loss_low=controller._cumulative_loss_low,
            cumulative_loss_high=controller._cumulative_loss_high,
            replica_id=replica_id,
        )

    def apply_to_controller(self, controller: ConformalRiskController) -> None:
        """Applies this synchronized snapshot onto a calibrated ConformalRiskController."""
        if not controller.is_calibrated:
            raise ValueError("Target ConformalRiskController must be calibrated before sync.")

        controller._low_threshold = float(self.low_threshold)
        controller._high_threshold = float(self.high_threshold)
        controller._effective_alpha = float(self.effective_alpha)
        controller._online_steps = int(self.online_steps)
        controller._cumulative_loss_low = float(self.cumulative_loss_low)
        controller._cumulative_loss_high = float(self.cumulative_loss_high)

    def to_dict(self) -> dict[str, Any]:
        """Serializes snapshot to a JSON-compatible dictionary."""
        return {
            "alpha": self.alpha,
            "max_loss_bound": self.max_loss_bound,
            "learning_rate": self.learning_rate,
            "base_low_threshold": self.base_low_threshold,
            "base_high_threshold": self.base_high_threshold,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "effective_alpha": self.effective_alpha,
            "online_steps": self.online_steps,
            "cumulative_loss_low": self.cumulative_loss_low,
            "cumulative_loss_high": self.cumulative_loss_high,
            "replica_id": self.replica_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalStateSnapshot:
        """Deserializes a ConformalStateSnapshot from a dictionary."""
        return cls(
            alpha=float(data["alpha"]),
            max_loss_bound=float(data.get("max_loss_bound", 1.0)),
            learning_rate=float(data.get("learning_rate", 0.05)),
            base_low_threshold=float(data["base_low_threshold"]),
            base_high_threshold=float(data["base_high_threshold"]),
            low_threshold=float(data["low_threshold"]),
            high_threshold=float(data["high_threshold"]),
            effective_alpha=float(data["effective_alpha"]),
            online_steps=int(data["online_steps"]),
            cumulative_loss_low=float(data["cumulative_loss_low"]),
            cumulative_loss_high=float(data["cumulative_loss_high"]),
            replica_id=str(data.get("replica_id", "default")),
        )


def compute_thresholds_from_cumulative(
    alpha: float,
    max_loss_bound: float,
    learning_rate: float,
    base_low_threshold: float,
    base_high_threshold: float,
    online_steps: int,
    cumulative_loss_low: float,
    cumulative_loss_high: float,
) -> tuple[float, float, float]:
    """Derives (low_threshold, high_threshold, effective_alpha) from cumulative ACI counters.

    Because cumulative loss and step counts form a commutative monoid under addition,
    deriving thresholds directly from accumulated drift guarantees strict commutativity
    and associativity across multi-replica state merges.
    """
    if online_steps <= 0:
        return (
            float(np.clip(base_low_threshold, 0.0, 1.0)),
            float(np.clip(max(base_low_threshold, base_high_threshold), 0.0, 1.0)),
            float(np.clip(alpha, 1e-4, 1.0 - 1e-4)),
        )

    net_grad_low = (cumulative_loss_low - online_steps * alpha) / max_loss_bound
    net_grad_high = (cumulative_loss_high - online_steps * alpha) / max_loss_bound

    low_t = float(np.clip(base_low_threshold - learning_rate * net_grad_low, 0.0, 1.0))
    high_t = float(np.clip(base_high_threshold - learning_rate * net_grad_high, low_t, 1.0))
    eff_alpha = float(np.clip(alpha - learning_rate * net_grad_low, 1e-4, 1.0 - 1e-4))
    return low_t, high_t, eff_alpha


def merge_conformal_snapshots(
    first: ConformalStateSnapshot,
    second: ConformalStateSnapshot,
) -> ConformalStateSnapshot:
    """Commutative and associative CRDT merger for two ConformalStateSnapshots."""
    alpha = round(0.5 * (first.alpha + second.alpha), 10)
    max_loss_bound = round(0.5 * (first.max_loss_bound + second.max_loss_bound), 10)
    learning_rate = round(0.5 * (first.learning_rate + second.learning_rate), 10)
    base_low = round(0.5 * (first.base_low_threshold + second.base_low_threshold), 10)
    base_high = round(0.5 * (first.base_high_threshold + second.base_high_threshold), 10)

    total_steps = int(first.online_steps + second.online_steps)
    total_loss_low = round(float(first.cumulative_loss_low + second.cumulative_loss_low), 10)
    total_loss_high = round(float(first.cumulative_loss_high + second.cumulative_loss_high), 10)

    low_t, high_t, eff_alpha = compute_thresholds_from_cumulative(
        alpha=alpha,
        max_loss_bound=max_loss_bound,
        learning_rate=learning_rate,
        base_low_threshold=base_low,
        base_high_threshold=base_high,
        online_steps=total_steps,
        cumulative_loss_low=total_loss_low,
        cumulative_loss_high=total_loss_high,
    )

    merged_ids = sorted({first.replica_id, second.replica_id})
    replica_id = "+".join(merged_ids) if len(merged_ids) > 1 else merged_ids[0]

    return ConformalStateSnapshot(
        alpha=alpha,
        max_loss_bound=max_loss_bound,
        learning_rate=learning_rate,
        base_low_threshold=base_low,
        base_high_threshold=base_high,
        low_threshold=low_t,
        high_threshold=high_t,
        effective_alpha=eff_alpha,
        online_steps=total_steps,
        cumulative_loss_low=total_loss_low,
        cumulative_loss_high=total_loss_high,
        replica_id=replica_id,
    )


class ConformalStateBackend(abc.ABC):
    """Abstract storage backend for distributed conformal risk state."""

    @abc.abstractmethod
    def read(self) -> ConformalStateSnapshot | None:
        """Reads the current shared ConformalStateSnapshot, or None if uninitialized."""

    @abc.abstractmethod
    def write(self, snapshot: ConformalStateSnapshot) -> None:
        """Persists the updated shared ConformalStateSnapshot."""


class InMemorySharedConformalBackend(ConformalStateBackend):
    """Thread-safe in-memory shared state backend for multi-worker testing and local pools."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: ConformalStateSnapshot | None = None

    def read(self) -> ConformalStateSnapshot | None:
        with self._lock:
            return self._snapshot

    def write(self, snapshot: ConformalStateSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot


class AtomicFileConformalBackend(ConformalStateBackend):
    """Atomic file-backed shared conformal state store for multi-process workers."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self._lock = threading.Lock()

    def read(self) -> ConformalStateSnapshot | None:
        with self._lock:
            if not self.filepath.is_file():
                return None
            raw = json.loads(self.filepath.read_text(encoding="utf-8"))
            return ConformalStateSnapshot.from_dict(raw)

    def write(self, snapshot: ConformalStateSnapshot) -> None:
        with self._lock:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.filepath.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
            os.replace(tmp_path, self.filepath)


class DistributedConformalCoordinator:
    """Coordinates incremental CRDT state synchronization across CausalProbeRouter replicas."""

    def __init__(
        self,
        backend: ConformalStateBackend | None = None,
        replica_id: str = "replica-0",
    ) -> None:
        self.backend: ConformalStateBackend = backend or InMemorySharedConformalBackend()
        self.replica_id = replica_id
        self._lock = threading.Lock()
        self._last_synced_steps: int = 0
        self._last_synced_loss_low: float = 0.0
        self._last_synced_loss_high: float = 0.0

    def sync_controller(self, controller: ConformalRiskController) -> ConformalStateSnapshot:
        """Pushes local incremental online updates to the backend and pulls merged global state."""
        local_snap = ConformalStateSnapshot.from_controller(controller, replica_id=self.replica_id)

        with self._lock:
            delta_steps = max(0, local_snap.online_steps - self._last_synced_steps)
            delta_low = max(0.0, local_snap.cumulative_loss_low - self._last_synced_loss_low)
            delta_high = max(0.0, local_snap.cumulative_loss_high - self._last_synced_loss_high)

            remote_snap = self.backend.read()
            if remote_snap is None:
                merged = ConformalStateSnapshot(
                    alpha=local_snap.alpha,
                    max_loss_bound=local_snap.max_loss_bound,
                    learning_rate=local_snap.learning_rate,
                    base_low_threshold=local_snap.base_low_threshold,
                    base_high_threshold=local_snap.base_high_threshold,
                    low_threshold=local_snap.low_threshold,
                    high_threshold=local_snap.high_threshold,
                    effective_alpha=local_snap.effective_alpha,
                    online_steps=local_snap.online_steps,
                    cumulative_loss_low=local_snap.cumulative_loss_low,
                    cumulative_loss_high=local_snap.cumulative_loss_high,
                    replica_id=self.replica_id,
                )
            else:
                delta_snap = ConformalStateSnapshot(
                    alpha=local_snap.alpha,
                    max_loss_bound=local_snap.max_loss_bound,
                    learning_rate=local_snap.learning_rate,
                    base_low_threshold=local_snap.base_low_threshold,
                    base_high_threshold=local_snap.base_high_threshold,
                    low_threshold=local_snap.low_threshold,
                    high_threshold=local_snap.high_threshold,
                    effective_alpha=local_snap.effective_alpha,
                    online_steps=delta_steps,
                    cumulative_loss_low=delta_low,
                    cumulative_loss_high=delta_high,
                    replica_id=self.replica_id,
                )
                merged = merge_conformal_snapshots(remote_snap, delta_snap)

            self.backend.write(merged)
            merged.apply_to_controller(controller)
            self._last_synced_steps = merged.online_steps
            self._last_synced_loss_low = merged.cumulative_loss_low
            self._last_synced_loss_high = merged.cumulative_loss_high
            return merged

    def merge_remote_delta(
        self,
        controller: ConformalRiskController,
        delta_steps: int,
        delta_loss_low: float,
        delta_loss_high: float,
        remote_replica_id: str = "remote-peer",
    ) -> ConformalStateSnapshot:
        """Merges an explicit remote delta into the shared backend and updates local controller."""
        if delta_steps < 0 or delta_loss_low < 0.0 or delta_loss_high < 0.0:
            raise ValueError("Remote conformal delta counters must be non-negative.")

        current = self.sync_controller(controller)
        delta_snap = ConformalStateSnapshot(
            alpha=current.alpha,
            max_loss_bound=current.max_loss_bound,
            learning_rate=current.learning_rate,
            base_low_threshold=current.base_low_threshold,
            base_high_threshold=current.base_high_threshold,
            low_threshold=current.low_threshold,
            high_threshold=current.high_threshold,
            effective_alpha=current.effective_alpha,
            online_steps=int(delta_steps),
            cumulative_loss_low=float(delta_loss_low),
            cumulative_loss_high=float(delta_loss_high),
            replica_id=remote_replica_id,
        )
        with self._lock:
            merged = merge_conformal_snapshots(current, delta_snap)
            self.backend.write(merged)
            merged.apply_to_controller(controller)
            self._last_synced_steps = merged.online_steps
            self._last_synced_loss_low = merged.cumulative_loss_low
            self._last_synced_loss_high = merged.cumulative_loss_high
            return merged


__all__ = [
    "AtomicFileConformalBackend",
    "ConformalStateBackend",
    "ConformalStateSnapshot",
    "DistributedConformalCoordinator",
    "InMemorySharedConformalBackend",
    "compute_thresholds_from_cumulative",
    "merge_conformal_snapshots",
]
