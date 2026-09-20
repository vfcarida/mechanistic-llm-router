"""Deprecated legacy core router module.

This module re-exports the canonical MechanisticRouter from
`mechanistic_router.routers.mechanistic` to maintain backwards compatibility.
"""

from ..routers.mechanistic import MechanisticRouter

__all__ = ["MechanisticRouter"]
