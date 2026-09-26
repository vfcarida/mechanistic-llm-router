"""Abstract base encoder interface for prefill activation extraction."""

import abc

import torch
import torch.nn as nn


class AbstractEncoder(nn.Module, abc.ABC):
    """Abstract base class for all encoders providing prefill activation extraction.

    Subclasses must implement `forward(input_ids)` returning a tuple of:
    1. The final layer output tensor [batch_size, seq_len, hidden_dim].
    2. A list of unpooled activation tensors for intermediate layers, each
       with shape [batch_size, seq_len, hidden_dim].
    """

    vocab_size: int = 10000

    @abc.abstractmethod
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Executes forward pass and returns (final_output, intermediate_activations).

        Args:
            input_ids: Input token tensor of shape [batch_size, seq_len].

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Final output and per-layer activations.
        """
        raise NotImplementedError


__all__ = ["AbstractEncoder"]
