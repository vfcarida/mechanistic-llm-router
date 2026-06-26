import numpy as np
import torch

def compute_effective_dimensionality(activation: torch.Tensor) -> float:
    """Calcula a Dimensionalidade Efetiva (d_eff) de uma matriz de ativação latente.

    Este algoritmo estima quão "espalhado" é o processamento da informação no
    espaço latente. Prompts densos/complexos ativam múltiplas dimensões ortogonais,
    enquanto prompts triviais colapsam em poucas dimensões dominantes.

    Metodologia:
    1. Calcula a Decomposição em Valores Singulares (SVD).
    2. Eleva os valores singulares ao quadrado para obter o espectro de "Energia".
    3. Normaliza a Energia numa distribuição de probabilidade.
    4. Computa a Entropia de Shannon (H) dessa distribuição.
    5. Retorna exp(H), que é o número de dimensões efetivas.

    Args:
        activation (torch.Tensor): O tensor de ativação [N, D] extraído do encoder.

    Returns:
        float: Valor contínuo representando o d_eff (>= 1.0). Se ocorrer instabilidade
            numérica na matriz, retorna 1.0 como fallback seguro.
    """
    act_np = activation.detach().cpu().numpy()
    if act_np.ndim == 1:
        act_np = act_np.reshape(1, -1)

    try:
        # Extração do Espectro via SVD
        _, s, _ = np.linalg.svd(act_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    # Energia Espectral
    energy = s ** 2
    total_energy = np.sum(energy)
    if total_energy < 1e-10:
        return 1.0

    # Distribuição de Probabilidade da Energia
    p = energy / total_energy
    p = p[p > 1e-10]  # Filtragem para evitar log(0)
    
    # Entropia de Shannon
    entropy = -np.sum(p * np.log(p))

    # Re-exponenciação para o domínio de dimensionalidade
    d_eff = np.exp(entropy)
    return float(d_eff)


def compute_fisher_separability(
    success_activations: torch.Tensor,
    failure_activations: torch.Tensor,
) -> float:
    """Calcula o Critério de Separabilidade J de Fisher entre dois clusters.

    No contexto de Roteamento Mecanístico, avaliamos o quão bem o modelo de destino
    consegue separar os padrões de "sucesso" e "falha" para o prompt atual.
    Se a variância inter-classes (distância das médias) é superior à variância
    intra-classes (dispersão interna do cluster), o Fisher J será alto, indicando
    que o modelo é mecanisticamente competente para resolver o prompt.

    Args:
        success_activations (torch.Tensor): Matriz [N, D] das ativações de sucessos.
        failure_activations (torch.Tensor): Matriz [N, D] das ativações de falhas.

    Returns:
        float: A medida contínua de Separabilidade J de Fisher (> 0).
    """
    # Centróides dos Clusters
    mu_success = success_activations.mean(dim=0)
    mu_failure = failure_activations.mean(dim=0)

    # Dispersão Interna (Intra-class Variance)
    var_success = success_activations.var(dim=0, unbiased=False)
    var_failure = failure_activations.var(dim=0, unbiased=False)

    within_class_scatter = var_success + var_failure + 1e-10
    
    # Fisher Discriminant Ratio por dimensão (Distância Mahalanobis diagonalizada)
    fisher_per_dim = (mu_success - mu_failure) ** 2 / within_class_scatter
    
    # J Global (Média across todas as dimensões latentes)
    J = fisher_per_dim.mean().item()

    return J
