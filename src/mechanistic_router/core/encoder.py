"""SharedTrunkEncoder Module for Prefill Hidden Activation Extraction."""

import torch
import torch.nn as nn
from ..config import RouterConfig


class SharedTrunkEncoder(nn.Module):
    """Simulator of a Lightweight Shared-Trunk Encoder under Encoder-Target Decoupling.

    In production deployment, this component represents the prefill stage of a
    lightweight language model (e.g., BERT, DistilRoBERTa, or the initial N layers
    of the base LLM).

    The mechanistic approach extracts the hidden states tensor generated while
    processing the prompt sequence, bypassing the full autoregressive generation loop.

    Attributes:
        config: Instance specifying dimensionality parameters.
        vocab_size: Simulated static vocabulary size.
        embedding: Token embedding layer.
        layers: Stack of dense Feed-Forward layers simulating prefill.
    """

    def __init__(self, config: RouterConfig):
        """Initializes simulated neural network architecture using configuration settings."""
        super().__init__()
        self.config = config

        self.vocab_size = 10000
        self.embedding = nn.Embedding(self.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
            )
            for _ in range(config.num_prefill_layers)
        ])

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Executes the prefill forward pass and collects activation tensors.

        Args:
            input_ids: Token identifier tensor [batch_size, seq_len].

        Returns:
            Tuple containing:
                - Final output tensor after prefill layers.
                - List containing raw unpooled activations of each intermediate layer,
                  formatted as [batch_size, seq_len, hidden_dim].
        """
        if input_ids.numel() == 0:
            raise ValueError("Input tensor (input_ids) cannot be empty.")

        x = self.embedding(input_ids)

        layer_activations = []
        for layer in self.layers:
            x = layer(x)
            layer_activations.append(x)

        return x, layer_activations
