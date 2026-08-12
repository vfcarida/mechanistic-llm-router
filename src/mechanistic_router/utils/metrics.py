"""Metrics Normalization and Performance Evaluation Helpers."""


def normalized_inverse_cost(cost: float, cost_min: float, cost_max: float) -> float:
    """Calculates Normalized Inverse Cost (1/C) mapped to interval [0, 1].

    Ensures lower cost models receive higher competitive advantage during scoring.

    Args:
        cost: Nominal cost of target candidate model.
        cost_min: Minimal cost available in full candidate pool.
        cost_max: Maximal cost available in full candidate pool.

    Returns:
        Normalized score in [0.0, 1.0]. Returns 0.5 if all costs are uniform.
    """
    if cost <= 0.0 or cost_min <= 0.0:
        raise ValueError("Model costs must be strictly positive (> 0.0).")

    inv_c = 1.0 / cost
    inv_c_min = 1.0 / cost_min
    inv_c_max = 1.0 / cost_max

    denominator = inv_c_min - inv_c_max
    if abs(denominator) < 1e-10:
        return 0.5

    score = (inv_c - inv_c_max) / denominator
    return max(0.0, min(float(score), 1.0))


def normalized_accuracy(accuracy: float, accuracy_floor: float, accuracy_ceiling: float) -> float:
    """Calculates Normalized Accuracy mapped to interval [0, 1].

    Args:
        accuracy: Raw accuracy score.
        accuracy_floor: Minimum baseline accuracy floor.
        accuracy_ceiling: Maximum accuracy ceiling.

    Returns:
        Normalized accuracy score in [0.0, 1.0].
    """
    denominator = accuracy_ceiling - accuracy_floor
    if abs(denominator) < 1e-10:
        return 1.0

    acc_norm = (accuracy - accuracy_floor) / denominator
    return max(0.0, min(float(acc_norm), 1.0))
