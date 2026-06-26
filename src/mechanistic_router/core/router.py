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
    """Implementa o Orquestrador Mecanístico de Seleção de Rota.

    O `MechanisticRouter` atua na interseção matemática entre a extração
    de estado topológico do encoder (Prefill) e as heurísticas de otimização
    do pool de modelos. 

    A lógica emprega **Encoder-Target Decoupling** (Duas Fases):
    1. **Portão de Competência (Gating via Fisher):** Filtra imediatamente
       modelos que colapsam a matriz de ativação (J < Threshold).
    2. **Log-Scale Cost Selection:** Dentre os sobreviventes, escala os custos
       no espaço logarítmico para priorizar consistentemente o modelo mais barato
       que é comprovadamente competente (em vez de usar bônus lineares que enviesam
       sempre para LLMs).
    """

    def __init__(
        self,
        encoder: SharedTrunkEncoder,
        model_pool: dict[str, TargetModel],
        config: RouterConfig,
    ):
        """Inicializa o router e injeta o modelo estático de encoder.

        Args:
            encoder (SharedTrunkEncoder): O modelo leve para extrair o prefill.
            model_pool (dict[str, TargetModel]): Dicionário de LLMs disponíveis.
            config (RouterConfig): Objeto de injeção de parâmetros (limiares e orçamento).
        """
        self.encoder = encoder
        self.model_pool = model_pool
        self.config = config

        # O encoder atua apenas em inferência
        self.encoder.eval()

        # Determinação antecipada dos limites para normalização linear rápida
        self.cost_min = min(m.cost for m in self.model_pool.values())
        self.cost_max = max(m.cost for m in self.model_pool.values())
        
        # O piso da acurácia é 0.40 (pior caso de degradação admissível)
        self.acc_floor = 0.40
        self.acc_ceil = max(m.base_accuracy for m in self.model_pool.values())

    def _prompt_to_tensor(self, text: str) -> torch.Tensor:
        """Processa a query crua (String) num tensor de token IDs simulado.

        Usa um hashing estocástico (limitado ao vocabulário) para não depender
        de bibliotecas pesadas de tokenização na simulação atual.

        Args:
            text (str): A entrada crua do usuário.

        Returns:
            torch.Tensor: Tensor shape [1, seq_len]
        """
        if not text.strip():
            # Edge case preventivo
            return torch.tensor([[0]], dtype=torch.long)

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
        """Gera distribuições artificiais de variância para o Fisher J.

        Em produção real, este método interpolaria com um banco vetorial
        histórico offline das ativações das camadas para aquele prompt. Aqui,
        forjamos clusters de 'sucesso' e 'falha' injetando ruído Gaussiano em 
        torno da ativação base em proporção à degradação de competência do LLM.

        Args:
            layer_activations (list[torch.Tensor]): Matrizes agregadas pós-prefill.
            model (TargetModel): O alvo sob teste estatístico.
            complexity (TaskComplexity): Complexidade ground-truth inferida do prompt.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: As matrizes sintéticas de sucesso e falha.
        """
        # Usa a representação agregada (Média nas camadas)
        base_norm = torch.stack(layer_activations).mean(dim=0).squeeze(0)
        
        complexity_order = list(TaskComplexity)
        task_idx = complexity_order.index(complexity)
        ceiling_idx = complexity_order.index(model.complexity_ceiling)

        # O modelo é competente se o prompt está no seu limite ou abaixo
        if task_idx <= ceiling_idx:
            competence = 0.85 + 0.15 * (1.0 - model.cost / self.cost_max)
        else:
            # Degrada severamente o cluster e funde-o (reprovação automática no Fisher)
            gap = task_idx - ceiling_idx
            competence = max(0.02, 0.15 - gap * 0.06)

        n_samples_per_class = 20
        delta = competence * 0.5 

        # Semente estática para determinismo por alvo (garante estabilidade no pool)
        rng = torch.Generator()
        rng.manual_seed(hash(model.name) % (2**31))

        # Espalhamento aumenta conforme a competência cai
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
        """Aplica a heurística matemática Desacoplada e retorna o alvo ideal.

        Args:
            prompt_text (str): Requerimento do usuário.
            complexity (TaskComplexity): Informação ground-truth do dataset.

        Returns:
            tuple[str, dict]: O nome do modelo vencedor e o rastreio (debug) dos
                sinais latentes de cada modelo da pool.
        """
        input_tensor = self._prompt_to_tensor(prompt_text)
        _, layer_activations = self.encoder(input_tensor)

        # 1. Dimensionalidade Efetiva Média (d_eff) 
        d_eff_values = [compute_effective_dimensionality(act) for act in layer_activations]
        d_eff_mean = float(np.mean(d_eff_values))
        complexity_signal = min(d_eff_mean / (self.config.hidden_dim * 0.5), 1.0)

        model_signals: dict[str, dict[str, float]] = {}

        # Geração de assinaturas matemáticas de cada LLM alvo
        for name, model in self.model_pool.items():
            acc = get_model_accuracy(model, complexity)
            inv_cost_score = normalized_inverse_cost(model.cost, self.cost_min, self.cost_max)
            acc_score = normalized_accuracy(acc, self.acc_floor, self.acc_ceil)

            success_act, failure_act = self._simulate_fisher_activations(
                layer_activations, model, complexity
            )
            fisher_j = compute_fisher_separability(success_act, failure_act)
            
            # Normalização suave J / (J+1) mapeia o domínio para [0,1)
            fisher_j_norm = min(fisher_j / (fisher_j + 1.0), 1.0)
            
            # O Portão Central: O modelo é aprovado topologicamente?
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

        # Orçamento dinâmico que relaxa em complexidade extremas (lambda elástico)
        effective_lambda = self.config.lambda_budget * (1.0 - 0.4 * complexity_signal ** 2)

        competent_models = {
            name: signals
            for name, signals in model_signals.items()
            if signals["is_competent"] > 0.5
        }

        scores: dict[str, float] = {}
        details: dict[str, dict[str, float]] = {}

        # FASE DE SELEÇÃO OTIMIZADA
        if competent_models:
            # Se pelo menos 1 modelo é competente, aplicamos Decoupling Puro.
            # Comparamos os competentes na escala Log do Preço.
            log_ratio = math.log(self.cost_max / self.cost_min)

            for name, signals in model_signals.items():
                if signals["is_competent"] > 0.5:
                    if log_ratio > 1e-10:
                        log_cost_score = math.log(self.cost_max / signals["cost"]) / log_ratio
                    else:
                        log_cost_score = 0.5

                    # O score final pondera agressivamente o menor preço no espaço Log
                    score = effective_lambda * log_cost_score + (1.0 - effective_lambda) * signals["acc_norm"]
                else:
                    # Modelos incompetentes recebem um peso mortal de decaimento (alpha penalty)
                    score = (1.0 - effective_lambda) * signals["acc_norm"] * self.config.fisher_alpha

                scores[name] = score
                details[name] = {
                    **{k: v for k, v in signals.items() if k != "cost"},
                    "effective_lambda": effective_lambda,
                    "final_score": score,
                }
        else:
            # Failsafe Sub-Ótimo (Emergency Routing):
            # Se a query foi tão extrema que todos falharam na separabilidade,
            # roteia puramente por acurácia bruta, ignorando preço.
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
