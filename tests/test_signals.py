import torch
import pytest
import numpy as np
from mechanistic_router.signals.math_utils import (
    compute_effective_dimensionality,
    compute_fisher_separability,
)

def test_effective_dimensionality_basic():
    """Testa se a dimensionalidade efetiva está retornando valores válidos (d_eff)."""
    # Matriz com apenas 1 valor singular dominante (entropia 0, d_eff = 1.0)
    activations_low_dim = torch.zeros((10, 128))
    activations_low_dim[:, 0] = 1.0
    
    d_eff_low = compute_effective_dimensionality(activations_low_dim)
    assert 1.0 <= d_eff_low < 1.1

    # Matriz com ruído gaussiano (deve ter dimensionalidade mais alta)
    torch.manual_seed(42)
    activations_high_dim = torch.randn((10, 128))
    d_eff_high = compute_effective_dimensionality(activations_high_dim)
    assert d_eff_high > 5.0

def test_effective_dimensionality_zero_energy():
    """Valida o fallback mecânico caso o tensor seja 100% nulo (0 energy)."""
    activations = torch.zeros((5, 128))
    d_eff = compute_effective_dimensionality(activations)
    assert d_eff == 1.0

def test_fisher_separability_basic():
    """Testa se o Fisher J captura a distância entre clusters topológicos."""
    # Clusters idênticos (J deve ser nulo)
    act1 = torch.ones((20, 128))
    act2 = torch.ones((20, 128))
    j_zero = compute_fisher_separability(act1, act2)
    assert j_zero < 1e-5

    # Clusters bem separados no hiperespaço
    act_success = torch.randn((20, 128)) + 5.0
    act_failure = torch.randn((20, 128)) - 5.0
    j_high = compute_fisher_separability(act_success, act_failure)
    assert j_high > 10.0

def test_fisher_separability_numerical_stability():
    """Garante que instabilidade divisória em clusters sem variância seja contida."""
    # Clusters sem variância (compostos por escalares idênticos)
    # the 1e-10 epsilon in the denominator should prevent division by zero NaN
    act_success = torch.full((10, 128), 2.0)
    act_failure = torch.full((10, 128), -2.0)
    j_value = compute_fisher_separability(act_success, act_failure)
    
    # Será um número gigantesco devido à variância zero, mas não NaN
    assert not np.isnan(j_value)
    assert j_value > 1000.0
