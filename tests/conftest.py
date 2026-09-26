"""Pytest Global Fixtures and Mocks Setup."""

import os

import pytest
import torch

os.environ.setdefault("ROUTER_API_KEY", "test-router-key")

from mechanistic_router.config import DEFAULT_CONFIG, RouterConfig
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TargetModel


@pytest.fixture
def default_config() -> RouterConfig:
    """Fixture providing global router configuration."""
    return DEFAULT_CONFIG


@pytest.fixture
def mock_encoder(default_config: RouterConfig) -> SharedTrunkEncoder:
    """Fixture providing simulated encoder instance."""
    return SharedTrunkEncoder(default_config)


@pytest.fixture
def model_pool() -> dict[str, TargetModel]:
    """Fixture providing model candidate pool dictionary."""
    return MODEL_POOL


@pytest.fixture
def sample_input_ids() -> torch.Tensor:
    """Fixture providing sample input token tensor."""
    return torch.tensor([[1, 10, 20, 30, 40]], dtype=torch.long)
