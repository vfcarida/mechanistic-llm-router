"""TransformerLens Hook Module for Local SLM Prefill Probing."""

from typing import Any
import torch
import torch.nn as nn


class TransformerLensHook:
    """Manager for registering PyTorch forward hooks to capture hidden state activations.

    Designed for lightweight local Small Language Models (SLMs) acting as front-door
    topological classifiers. Captures unpooled hidden activations [batch, seq_len, hidden_dim]
    without modifying model parameters or interrupting standard execution.
    """

    def __init__(self, model: nn.Module):
        """Initializes hook manager around a target PyTorch model instance."""
        self.model = model
        self.captured_activations: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []

    def register_hooks(self, layer_names: list[str]) -> None:
        """Registers forward hooks on specified named submodules of the model."""
        self.clear_hooks()

        for name, module in self.model.named_modules():
            if name in layer_names or not layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook_fn(module: nn.Module, input_tensor: Any, output_tensor: Any) -> None:
            if isinstance(output_tensor, tuple):
                act = output_tensor[0]
            else:
                act = output_tensor
            self.captured_activations[layer_name] = act.detach().clone()

        return hook_fn

    def clear_hooks(self) -> None:
        """Removes all registered hooks and clears activation buffers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.captured_activations.clear()

    def get_activations(self) -> list[torch.Tensor]:
        """Returns ordered list of captured activation tensors."""
        return list(self.captured_activations.values())
