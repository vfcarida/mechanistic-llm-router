"""Unit Tests for Mechanistic Probing Engine Modules."""

import pytest
import torch

from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.probing.nnsight_probe import NNsightProbe
from mechanistic_router.probing.sae_engine import SAEEngine
from mechanistic_router.probing.transformer_lens_hook import TransformerLensHook


def test_transformer_lens_hook(
    mock_encoder: SharedTrunkEncoder, sample_input_ids: torch.Tensor
) -> None:
    """Test forward hook registration and activation capture."""
    hook_mgr = TransformerLensHook(mock_encoder)
    hook_mgr.register_hooks([])

    _, layer_acts = mock_encoder(sample_input_ids)
    captured = hook_mgr.get_activations()

    assert len(captured) > 0
    hook_mgr.clear_hooks()
    assert len(hook_mgr.get_activations()) == 0


@pytest.mark.asyncio
async def test_nnsight_probe() -> None:
    """Test nnsight remote NDIF probing trace simulation."""
    probe = NNsightProbe(model_key="meta-llama/Meta-Llama-3-8B")
    trace = await probe.probe_remote_activations(prompt="Analyze mathematical logic")

    assert trace["model_key"] == "meta-llama/Meta-Llama-3-8B"
    assert "simulated_ndif_trace" in trace
    assert len(trace["layers_inspected"]) > 0


def test_sae_engine_circuit_extraction() -> None:
    """Test SAEEngine encoding, decoding, and cognitive circuit extraction."""
    sae = SAEEngine(d_in=128, d_sae=256)
    x = torch.randn(10, 128)

    encoded = sae.encode(x)
    assert encoded.shape == (10, 256)

    decoded = sae.decode(encoded)
    assert decoded.shape == (10, 128)

    circuits = sae.extract_active_circuits(x)
    assert "circuit_type" in circuits
    assert "sparsity_ratio" in circuits
    assert "requires_oracle" in circuits
