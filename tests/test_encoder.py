"""Unit Tests for SharedTrunkEncoder Module."""

import pytest
import torch
from mechanistic_router.core.encoder import SharedTrunkEncoder


def test_encoder_forward_pass(mock_encoder: SharedTrunkEncoder, sample_input_ids: torch.Tensor) -> None:
    """Test forward pass output shapes and activation layer collection."""
    output, activations = mock_encoder(sample_input_ids)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 5, mock_encoder.config.hidden_dim)
    assert len(activations) == mock_encoder.config.num_prefill_layers
    assert all(act.shape == (1, 5, mock_encoder.config.hidden_dim) for act in activations)


def test_encoder_empty_tensor(mock_encoder: SharedTrunkEncoder) -> None:
    """Test validation error on empty tensor."""
    empty_tensor = torch.tensor([], dtype=torch.long)
    with pytest.raises(ValueError, match="Input tensor .* cannot be empty"):
        mock_encoder(empty_tensor)
