import math
import numpy as np
import torch
import torch.nn as nn

from ..config import RouterConfig
from ..models.types import TargetModel, TaskComplexity
from ..models.pool import get_model_accuracy
from ..signals.math_utils import compute_effective_dimensionality, compute_fisher_separability
from ..utils.metrics import normalized_inverse_cost, normalized_accuracy
from .encoder import SharedTrunkEncoder

class MechanisticRouter:
    """Implementa o Roteador Mecanístico baseado em prefill e Fisher Gating."""

    def __init__(
        self,
        encoder: SharedTrunkEncoder,
        model_pool: dict[str, TargetModel],
        config: RouterConfig,
    ):
        self.encoder = encoder
        self.model_pool = model_pool
        self.config = config

        self.encoder.eval()

        # Determinar limites para normalização
        self.cost_min = min(m.cost for m in self.model_pool.values())
        self.cost_max = max(m.cost for m in self.model_pool.values())
        
        # O piso da acurácia é 0.40 (pior caso de degradação)
        self.acc_floor = 0.40
        self.acc_ceil = max(m.base_accuracy for m in self.model_pool.values())

    def _prompt_to_tensor(self, text: str) -> torch.Tensor:
        """Simula a tokenização usando hashing."""
        tokens = [hash(word) % self.encoder.vocab_size for word in text.split()[:50]]
        if not tokens:
            tokens = [0]
        return torch.tensor([tokens], dtype=torch.long)

    @torch.no_grad()
    def _simulate_fisher_activations(
        self,
        layer_activations: list[torch.Tensor],
        model: TargetModel,
        complexity: TaskComplexity,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gera clusters artificiais de sucesso/falha para simular a separabilidade."""
        # Usa a representação agregada média como base
        base_norm = torch.stack(layer_activations).mean(dim=0).squeeze(0)
        
        complexity_order = list(TaskComplexity)
        task_idx = complexity_order.index(complexity)
        ceiling_idx = complexity_order.index(model.complexity_ceiling)

        if task_idx <= ceiling_idx:
            # Modelo competente
            competence = 0.85 + 0.15 * (1.0 - model.cost / self.cost_max)
        else:
            # Modelo além do limite
            gap = task_idx - ceiling_idx
            competence = max(0.02, 0.15 - gap * 0.06)

        n_samples_per_class = 20
        delta = competence * 0.5 

        rng = torch.Generator()
        rng.manual_seed(hash(model.name) % (2**31))

        noise_scale = 0.3 * (1.0 - competence + 0.1)

        success_activations = (
            base_norm.unsqueeze(0).expand(n_samples_per_class, -1)
            + delta
            + torch.randn(n_samples_per_class, base_norm.shape[0], generator=rng) * noise_scale
        )
        failure_activations = (
            base_norm.unsqueeze(0).expand(n_samples_per_class, -1)
            - delta
            + torch.randn(n_samples_per_class, base_norm.shape[0], generator=rng) * noise_scale
        )

        return success_activations, failure_activations

    @torch.no_grad()
    def route(
        self,
        prompt_text: str,
        complexity: TaskComplexity,
    ) -> tuple[str, dict[str, dict[str, float]]]:
        """Determina o modelo ideal combinando sinais mecanísticos e eficiência."""
        input_tensor = self._prompt_to_tensor(prompt_text)
        _, layer_activations = self.encoder(input_tensor)

        d_eff_values = [compute_effective_dimensionality(act) for act in layer_activations]
        d_eff_mean = float(np.mean(d_eff_values))
        complexity_signal = min(d_eff_mean / (self.config.hidden_dim * 0.5), 1.0)

        model_signals: dict[str, dict[str, float]] = {}

        for name, model in self.model_pool.items():
            acc = get_model_accuracy(model, complexity)
            inv_cost_score = normalized_inverse_cost(model.cost, self.cost_min, self.cost_max)
            acc_score = normalized_accuracy(acc, self.acc_floor, self.acc_ceil)

            success_act, failure_act = self._simulate_fisher_activations(
                layer_activations, model, complexity
            )
            fisher_j = compute_fisher_separability(success_act, failure_act)
            fisher_j_norm = min(fisher_j / (fisher_j + 1.0), 1.0)
            is_competent = fisher_j_norm > self.config.fisher_j_threshold

            model_signals[name] = {
                "inv_cost_norm": inv_cost_score,
                "acc_norm": acc_score,
                "d_eff_mean": d_eff_mean,
                "fisher_j": fisher_j,
                "fisher_j_norm": fisher_j_norm,
                "is_competent": float(is_competent),
                "cost": model.cost,
                "accuracy": acc,
            }

        effective_lambda = self.config.lambda_budget * (1.0 - 0.4 * complexity_signal ** 2)

        competent_models = {
            name: signals
            for name, signals in model_signals.items()
            if signals["is_competent"] > 0.5
        }

        scores: dict[str, float] = {}
        details: dict[str, dict[str, float]] = {}

        if competent_models:
            log_ratio = math.log(self.cost_max / self.cost_min)

            for name, signals in model_signals.items():
                if signals["is_competent"] > 0.5:
                    if log_ratio > 1e-10:
                        log_cost_score = math.log(self.cost_max / signals["cost"]) / log_ratio
                    else:
                        log_cost_score = 0.5

                    score = effective_lambda * log_cost_score + (1.0 - effective_lambda) * signals["acc_norm"]
                else:
                    score = (1.0 - effective_lambda) * signals["acc_norm"] * 0.3

                scores[name] = score
                details[name] = {
                    **{k: v for k, v in signals.items() if k != "cost"},
                    "effective_lambda": effective_lambda,
                    "final_score": score,
                }
        else:
            for name, signals in model_signals.items():
                score = signals["acc_norm"]
                scores[name] = score
                details[name] = {
                    **{k: v for k, v in signals.items() if k != "cost"},
                    "effective_lambda": effective_lambda,
                    "final_score": score,
                }

        selected_model = max(scores, key=scores.get)  # type: ignore[arg-type]
        return selected_model, details
