import pytest
import torch
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder

@pytest.fixture
def encoder():
    return SharedTrunkEncoder(DEFAULT_CONFIG)

def test_encoder_output_shapes(encoder):
    """Testa se as dimensões tensor-matriz estão rigorosamente corretas."""
    # Batch = 2, SeqLen = 5
    input_ids = torch.tensor([[10, 20, 30, 40, 50], [1, 2, 3, 4, 5]], dtype=torch.long)
    final_out, layer_acts = encoder(input_ids)
    
    # 1. Output Final deve ser [batch, seq_len, hidden_dim]
    assert final_out.shape == (2, 5, DEFAULT_CONFIG.hidden_dim)
    
    # 2. Número de ativações capturadas deve igualar as camadas do config
    assert len(layer_acts) == DEFAULT_CONFIG.num_prefill_layers
    
    # 3. Cada ativação deve ser pareada (mean pooled -> [batch, hidden_dim])
    for act in layer_acts:
        assert act.shape == (2, DEFAULT_CONFIG.hidden_dim)

def test_encoder_empty_prompt(encoder):
    """Garante que falhamos graciosamente ao receber prompt sem tokens."""
    input_ids = torch.tensor([[]], dtype=torch.long)
    with pytest.raises(ValueError, match="não pode estar vazio"):
        encoder(input_ids)
