def normalized_inverse_cost(cost: float, cost_min: float, cost_max: float) -> float:
    """Calcula o Custo Inverso Normalizado para um modelo dentro de um Pool.

    Para garantir que modelos baratos tenham vantagem matemática competitiva
    durante o 'Cost-Optimal Selection', invertemos o custo (1/C) e o
    normalizamos linearmente para o intervalo [0, 1].

    Args:
        cost (float): O custo nominal do modelo alvo.
        cost_min (float): O custo mais barato disponível no pool inteiro.
        cost_max (float): O custo mais caro disponível no pool inteiro.

    Returns:
        float: Valor [0.0, 1.0]. Um score mais alto representa um custo menor (melhor).
               Retorna 0.5 fixo se todos os modelos tiverem o mesmo custo para
               evitar divisão por zero.
    """
    if cost <= 0.0 or cost_min <= 0.0:
        raise ValueError("Custos de modelos devem ser estritamente positivos (>0).")

    inv_c = 1.0 / cost
    inv_c_min = 1.0 / cost_min
    inv_c_max = 1.0 / cost_max

    denominator = inv_c_min - inv_c_max
    if abs(denominator) < 1e-10:
        return 0.5  # Neutralização caso o pool seja de preço único

    score = (inv_c - inv_c_max) / denominator
    # Clamp safety para bounds flutuantes
    return max(0.0, min(float(score), 1.0))


def normalized_accuracy(accuracy: float, accuracy_floor: float, accuracy_ceiling: float) -> float:
    """Calcula a Acurácia Normalizada de um modelo.

    Mapeia a expectativa de acurácia degradada do modelo para o intervalo [0, 1]
    relativo ao pool, auxiliando na comparação multi-modelo.

    Args:
        accuracy (float): A acurácia bruta projetada [0.0, 1.0].
        accuracy_floor (float): A pior acurácia base tolerável.
        accuracy_ceiling (float): A maior acurácia base do melhor modelo do pool.

    Returns:
        float: Score normalizado [0.0, 1.0].
    """
    denominator = accuracy_ceiling - accuracy_floor
    if abs(denominator) < 1e-10:
        return 1.0

    acc_norm = (accuracy - accuracy_floor) / denominator
    return max(0.0, min(float(acc_norm), 1.0))
