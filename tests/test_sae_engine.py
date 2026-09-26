"""Unit tests for SAEEngine including TopK sparsity and loss computation."""

import torch

from mechanistic_router.probing.sae_engine import SAEEngine


def test_sae_engine_forward_and_loss() -> None:
    """Verify SAEEngine forward pass and compute_loss calculation."""
    d_in = 64
    d_sae = 256
    sae = SAEEngine(d_in=d_in, d_sae=d_sae, l1_coefficient=0.01)

    batch_x = torch.randn(8, d_in)
    recon_x, f_acts = sae.forward(batch_x)

    assert recon_x.shape == (8, d_in)
    assert f_acts.shape == (8, d_sae)
    assert (f_acts >= 0).all()  # ReLU non-negative

    loss_dict = sae.compute_loss(batch_x)
    assert "loss" in loss_dict
    assert "reconstruction_loss" in loss_dict
    assert "l1_loss" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_sae_engine_topk_sparsity() -> None:
    """Verify TopK sparsity preserves only the k highest features per sample."""
    d_in = 32
    d_sae = 128
    top_k = 8
    sae = SAEEngine(d_in=d_in, d_sae=d_sae, top_k=top_k)

    batch_x = torch.randn(4, d_in)
    f_acts = sae.encode(batch_x)

    # Count non-zero activations per row
    non_zeros = (f_acts > 0).sum(dim=-1)
    assert (non_zeros <= top_k).all()


def test_sae_engine_custom_circuit_mapping() -> None:
    """Verify custom circuit feature map classifies active circuits correctly."""
    d_in = 32
    d_sae = 64
    custom_map = {
        "mathematical_reasoning": [0, 1, 2, 3, 4],
        "factual_retrieval": [10, 11, 12, 13, 14],
    }
    sae = SAEEngine(d_in=d_in, d_sae=d_sae, circuit_feature_map=custom_map)

    # Craft an input that triggers circuit extraction
    x = torch.randn(1, 10, d_in)
    circuits = sae.extract_active_circuits(x, threshold=0.0)

    assert "circuit_type" in circuits
    assert circuits["circuit_type"] in ("mathematical_reasoning", "factual_retrieval")
    assert "sparsity_ratio" in circuits
