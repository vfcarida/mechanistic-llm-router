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


def test_sae_engine_extract_circuits_edge_cases() -> None:
    """Verify SAEEngine.extract_active_circuits on boundary and edge conditions."""
    d_in = 16
    d_sae = 32
    sae = SAEEngine(d_in=d_in, d_sae=d_sae)

    # 1. Edge Case: No features active (threshold higher than any activation)
    x_zero = torch.zeros(1, d_in)
    circuits_none = sae.extract_active_circuits(x_zero, threshold=1e9)
    assert circuits_none["active_feature_indices"] == []
    assert circuits_none["total_active_features"] == 0
    assert circuits_none["sparsity_ratio"] == 0.0
    assert circuits_none["circuit_type"] == "factual_retrieval"
    assert circuits_none["requires_oracle"] is False

    # 2. Edge Case: Exactly single feature active (synthetic override)
    with torch.no_grad():
        # Zero out encoder weights and bias
        sae.W_enc.zero_()
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        # Set feature index 5 (math reasoning: 5 % 5 == 0) to active
        sae.b_enc[5] = 2.0

    circuits_single_math = sae.extract_active_circuits(x_zero, threshold=0.5)
    assert circuits_single_math["active_feature_indices"] == [5]
    assert circuits_single_math["total_active_features"] == 1
    assert circuits_single_math["sparsity_ratio"] == 1.0 / d_sae
    assert circuits_single_math["circuit_type"] == "mathematical_computation"
    assert circuits_single_math["requires_oracle"] is True

    # 3. Edge Case: Single non-math, non-logic feature (index 1)
    with torch.no_grad():
        sae.b_enc.zero_()
        sae.b_enc[1] = 1.5

    circuits_single_routine = sae.extract_active_circuits(x_zero, threshold=0.5)
    assert circuits_single_routine["active_feature_indices"] == [1]
    assert circuits_single_routine["circuit_type"] == "factual_retrieval"
    assert circuits_single_routine["requires_oracle"] is False

    # 4. Edge Case: 1D tensor input (auto-unsqueeze)
    x_1d = torch.zeros(d_in)
    circuits_1d = sae.extract_active_circuits(x_1d, threshold=0.5)
    assert circuits_1d["active_feature_indices"] == [1]

    # 5. Edge Case: threshold = 0.0 boundary condition
    circuits_zero_thresh = sae.extract_active_circuits(x_zero, threshold=0.0)
    assert 1 in circuits_zero_thresh["active_feature_indices"]
