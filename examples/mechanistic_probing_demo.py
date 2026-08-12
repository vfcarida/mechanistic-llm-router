"""Mechanistic Probing & SAE Circuit Extraction Demonstration Script."""

import torch
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.probing.sae_engine import SAEEngine
from mechanistic_router.probing.transformer_lens_hook import TransformerLensHook
from mechanistic_router.signals.math_utils import (
    compute_effective_dimensionality,
    compute_fisher_separability,
)


def main() -> None:
    print("=== Mechanistic Interpretability Probing & SAE Circuit Extraction ===")

    # Initialize encoder and hook manager
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    hook_manager = TransformerLensHook(encoder)
    sae_engine = SAEEngine(d_in=DEFAULT_CONFIG.hidden_dim, d_sae=512)

    # Register forward hooks on encoder layers
    hook_manager.register_hooks([])

    # Process prompt query
    prompt_ids = torch.tensor([[101, 2054, 2003, 1037, 3000, 102]], dtype=torch.long)
    _, layer_activations = encoder(prompt_ids)

    print(f"Captured {len(layer_activations)} prefill layer activation tensors.")

    # Calculate Effective Dimensionality (d_eff)
    d_eff_values = [
        compute_effective_dimensionality(act.squeeze(0)) for act in layer_activations
    ]
    print(f"Layer-wise Effective Dimensionality (d_eff): {[round(v, 2) for v in d_eff_values]}")

    # Extract SAE Cognitive Circuits
    last_layer_act = layer_activations[-1].squeeze(0)
    circuit_data = sae_engine.extract_active_circuits(last_layer_act)

    print("\n--- SAE Sparse Autoencoder Circuit Signals ---")
    print(f"Circuit Classification: {circuit_data['circuit_type']}")
    print(f"Active Feature Indices: {circuit_data['active_feature_indices']}")
    print(f"Sparsity Ratio: {circuit_data['sparsity_ratio']:.4f}")
    print(f"Requires Frontier Oracle LLM: {circuit_data['requires_oracle']}")

    hook_manager.clear_hooks()


if __name__ == "__main__":
    main()
