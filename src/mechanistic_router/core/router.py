"""Deprecated legacy core router module.

This module re-exports the canonical MechanisticRouter from
`mechanistic_router.routers.mechanistic` to maintain backwards compatibility.
"""

import warnings

from ..routers.mechanistic import MechanisticRouter

warnings.warn(
    "`mechanistic_router.core.router` is deprecated and will be removed in a future release. "
    "Import `MechanisticRouter` from `mechanistic_router.routers.mechanistic` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MechanisticRouter"]
