"""Abstract base encoder interface for prefill activation extraction."""

import abc
from typing import Any

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


class TransformerActivationEncoder(AbstractEncoder):
    """Concrete encoder using HuggingFace AutoModel for prefill activation extraction.

    Implements `AbstractEncoder` over real HuggingFace transformer backends (e.g.
    'HuggingFaceTB/SmolLM-135M', 'Qwen/Qwen2.5-0.5B', or local checkpoints).
    Extracts intermediate hidden states across all transformer blocks via
    `output_hidden_states=True`.
    """

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceTB/SmolLM-135M",
        model: nn.Module | None = None,
        tokenizer: Any = None,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        else:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as err:
                raise ImportError(
                    "The 'transformers' library is required to use TransformerActivationEncoder. "
                    "Install it via `pip install transformers`."
                ) from err

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype or torch.float32,
            ).to(self.device)

        self.model.eval()
        if hasattr(self.tokenizer, "vocab_size") and isinstance(self.tokenizer.vocab_size, int):
            self.vocab_size = self.tokenizer.vocab_size

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Executes forward pass with output_hidden_states=True.

        Args:
            input_ids: Input token tensor [batch_size, seq_len].

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                - Final last_hidden_state [batch_size, seq_len, hidden_dim].
                - List of intermediate hidden state tensors [batch_size, seq_len, hidden_dim].
        """
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids, output_hidden_states=True)

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        hidden_states = getattr(outputs, "hidden_states", None)

        if hidden_states is not None:
            intermediate_activations = [h.detach() for h in hidden_states]
        elif last_hidden_state is not None:
            intermediate_activations = [last_hidden_state.detach()]
        else:
            intermediate_activations = []

        if last_hidden_state is None and intermediate_activations:
            last_hidden_state = intermediate_activations[-1]

        if last_hidden_state is None:
            raise RuntimeError("Model outputs did not provide last_hidden_state or hidden_states.")

        return last_hidden_state, intermediate_activations

    def encode_text(self, text: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encodes raw text string into token representations and extracts activations."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not configured on TransformerActivationEncoder.")
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        return self.forward(input_ids)


__all__ = ["AbstractEncoder", "TransformerActivationEncoder"]
