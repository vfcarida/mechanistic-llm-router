def normalized_inverse_cost(cost: float, cost_min: float, cost_max: float) -> float:
    """Calcula o Custo Inverso Normalizado para um modelo."""
    inv_c = 1.0 / cost
    inv_c_min = 1.0 / cost_min
    inv_c_max = 1.0 / cost_max

    denominator = inv_c_min - inv_c_max
    if abs(denominator) < 1e-10:
        return 0.5  # Todos os modelos têm o mesmo custo

    return (inv_c - inv_c_max) / denominator


def normalized_accuracy(accuracy: float, accuracy_floor: float, accuracy_ceiling: float) -> float:
    """Calcula a Acurácia Normalizada de um modelo."""
    denominator = accuracy_ceiling - accuracy_floor
    if abs(denominator) < 1e-10:
        return 1.0

    acc_norm = (accuracy - accuracy_floor) / denominator
    return max(0.0, min(acc_norm, 1.0))
