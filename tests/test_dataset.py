import pytest
from mechanistic_router.data.mock_dataset import create_financial_dataset
from mechanistic_router.models.types import TaskComplexity

def test_dataset_distribution_convergence():
    """Valida se o gerador converge para as proporções estatísticas exigidas:
       55% ROTINA, 30% MODERADA, 15% COMPLEXA.
    """
    n_samples = 1000  # Tamanho elevado para Lei dos Grandes Números
    df = create_financial_dataset(n_samples=n_samples, seed=123)
    
    assert len(df) == n_samples
    
    counts = df["complexity"].value_counts(normalize=True)
    
    # Tolerância de 3% devido ao rng.choice estocástico
    assert counts[TaskComplexity.ROUTINE] == pytest.approx(0.55, abs=0.03)
    assert counts[TaskComplexity.MODERATE] == pytest.approx(0.30, abs=0.03)
    assert counts[TaskComplexity.COMPLEX] == pytest.approx(0.15, abs=0.03)

def test_dataset_determinism():
    """Valida se a injeção da semente produz DataFrames idênticos."""
    df1 = create_financial_dataset(n_samples=50, seed=42)
    df2 = create_financial_dataset(n_samples=50, seed=42)
    
    assert df1.equals(df2)
