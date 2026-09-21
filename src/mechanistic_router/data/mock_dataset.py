"""Deprecated synthetic mock dataset module.

This module is deprecated and maintained for backward compatibility.
Test fixtures should be imported from `tests.fixtures.mock_dataset`.
"""

import warnings

from ..schemas.eval import EvalCase

warnings.warn(
    "`mechanistic_router.data.mock_dataset` is deprecated and will be removed in a future release. "
    "Use test fixtures in `tests.fixtures.mock_dataset` or evaluation dataset loaders.",
    DeprecationWarning,
    stacklevel=2,
)


def create_financial_dataset(n_samples: int = 200, seed: int = 42) -> list[EvalCase]:
    """Generate a mock financial evaluation dataset emitting EvalCase instances.

    Deprecated: import from `tests.fixtures.mock_dataset` instead.
    """
    from tests.fixtures.mock_dataset import (
        create_financial_dataset as _fixture_create_dataset,
    )

    return _fixture_create_dataset(n_samples=n_samples, seed=seed)
