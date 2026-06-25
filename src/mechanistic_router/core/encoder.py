import torch
import torch.nn as nn
from ..config import RouterConfig

class SharedTrunkEncoder(nn.Module):
    """Simulador de um Encoder Leve (SharedTrunk) no modelo de desacoplamento.

    Na prática, este seria o prefill stage de um modelo menor (ex: BERT ou as
    primeiras camadas de um LLM). Aqui, simulamos extraindo ativações de
    redes lineares projetadas a partir dos embeddings do prompt.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Simulação do vocabulário (hash-based embedding)
        self.vocab_size = 10000
        self.embedding = nn.Embedding(self.vocab_size, config.hidden_dim)

        # Camadas do prefill
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
            for _ in range(config.num_prefill_layers)
        ])

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Executa o prefill e retorna as ativações ocultas.

        Args:
            input_ids: Tensor de tokens [batch_size, seq_len]

        Returns:
            Tupla (ultima_ativacao, lista_todas_ativacoes_por_camada)
        """
        x = self.embedding(input_ids)

        layer_activations = []
        for layer in self.layers:
            x = layer(x)
            # Pooling simples (média) para simular o estado latente da sentença
            pooled_x = x.mean(dim=1)
            layer_activations.append(pooled_x)

        return x, layer_activations
