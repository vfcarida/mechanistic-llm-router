"""Unit tests for TransformerActivationEncoder implementing AbstractEncoder."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from mechanistic_router.probing.base import (
    AbstractEncoder,
    TransformerActivationEncoder,
)


class MockTransformerModel(nn.Module):
    """Mock PyTorch transformer model simulating output_hidden_states=True."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True) -> MagicMock:
        batch_size, seq_len = input_ids.shape
        last_hidden = torch.randn(batch_size, seq_len, self.hidden_dim)
        hidden_states = [
            torch.randn(batch_size, seq_len, self.hidden_dim) for _ in range(self.num_layers)
        ]

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = last_hidden
        mock_outputs.hidden_states = hidden_states
        return mock_outputs


class MockTokenizer:
    """Mock tokenizer returning token tensors."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[101, 2054, 2003, 102]])}


def test_transformer_activation_encoder_interface() -> None:
    """Verify TransformerActivationEncoder inherits from AbstractEncoder."""
    model = MockTransformerModel()
    tokenizer = MockTokenizer(vocab_size=50000)

    encoder = TransformerActivationEncoder(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    assert isinstance(encoder, AbstractEncoder)
    assert encoder.vocab_size == 50000


def test_transformer_activation_encoder_forward() -> None:
    """Verify forward() returns last_hidden_state and per-layer intermediate hidden states."""
    model = MockTransformerModel(hidden_dim=32, num_layers=3)
    tokenizer = MockTokenizer()
    encoder = TransformerActivationEncoder(model=model, tokenizer=tokenizer)

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    last_hidden, hidden_states = encoder.forward(input_ids)

    assert last_hidden.shape == (2, 4, 32)
    assert len(hidden_states) == 3
    for h in hidden_states:
        assert h.shape == (2, 4, 32)


def test_transformer_activation_encoder_encode_text() -> None:
    """Verify encode_text tokenizes text input and extracts prefill activations."""
    model = MockTransformerModel(hidden_dim=48, num_layers=2)
    tokenizer = MockTokenizer()
    encoder = TransformerActivationEncoder(model=model, tokenizer=tokenizer)

    last_hidden, hidden_states = encoder.encode_text("Evaluate this prompt.")
    assert last_hidden.shape == (1, 4, 48)
    assert len(hidden_states) == 2
