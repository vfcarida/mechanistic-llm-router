import numpy as np
import torch

def compute_effective_dimensionality(activation: torch.Tensor) -> float:
    """Calcula a Dimensionalidade Efetiva (d_eff) de uma matriz de ativação.

    Usa a entropia de Shannon sobre a distribuição de energia espectral (SVD).
    """
    act_np = activation.detach().cpu().numpy()
    if act_np.ndim == 1:
        act_np = act_np.reshape(1, -1)

    try:
        _, s, _ = np.linalg.svd(act_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    energy = s ** 2
    total_energy = np.sum(energy)
    if total_energy < 1e-10:
        return 1.0

    p = energy / total_energy
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))

    d_eff = np.exp(entropy)
    return float(d_eff)


def compute_fisher_separability(
    success_activations: torch.Tensor,
    failure_activations: torch.Tensor,
) -> float:
    """Calcula a métrica de separabilidade J de Fisher entre sucesso e falha."""
    mu_success = success_activations.mean(dim=0)
    mu_failure = failure_activations.mean(dim=0)

    var_success = success_activations.var(dim=0, unbiased=False)
    var_failure = failure_activations.var(dim=0, unbiased=False)

    within_class_scatter = var_success + var_failure + 1e-10
    fisher_per_dim = (mu_success - mu_failure) ** 2 / within_class_scatter
    J = fisher_per_dim.mean().item()

    return J
