import torch
import torch.nn as nn
from ..config import RouterConfig

class SharedTrunkEncoder(nn.Module):
    """Simulador de um Encoder Leve (SharedTrunk) no modelo de Desacoplamento.

    Na prática, este componente representa o 'prefill stage' de um modelo
    de linguagem de pequeno porte (ex: BERT, DistilRoBERTa, ou as primeiras
    N camadas do LLM de base). 

    A abordagem mecanística requer extrair o tensor de ativações ocultas 
    (hidden states) gerado enquanto o prompt original é processado, sem a
    necessidade de rodar o loop autoregressivo completo.
    
    Attributes:
        config (RouterConfig): Instância contendo arquitetura de dimensionalidade.
        vocab_size (int): Tamanho estático simulado de vocabulário.
        embedding (nn.Embedding): Camada de mapeamento token -> tensor denso.
        layers (nn.ModuleList): Pilha de redes Feed-Forward densas representando
            as camadas de prefill.
    """

    def __init__(self, config: RouterConfig):
        """Inicializa a arquitetura da rede simulada baseada nas dimensões da configuração."""
        super().__init__()
        self.config = config
        
        # Simulação do vocabulário (hash-based embedding lock)
        self.vocab_size = 10000
        self.embedding = nn.Embedding(self.vocab_size, config.hidden_dim)

        # Camadas do prefill (Simulando uma arquitetura Transformer FFN reduzida)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU()
            )
            for _ in range(config.num_prefill_layers)
        ])

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Executa o forward pass do prefill e coleta ativações.

        Args:
            input_ids (torch.Tensor): Tensor de identificadores de tokens 
                [batch_size, seq_len] do prompt do usuário.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: 
                - O tensor de saída final pós-prefill.
                - Lista contendo a ativação condensada (mean pooling) de *cada* 
                  uma das camadas intermediárias (usada para o Fisher J).
        """
        if input_ids.numel() == 0:
            raise ValueError("O tensor de entrada (input_ids) não pode estar vazio.")

        x = self.embedding(input_ids)

        layer_activations = []
        for layer in self.layers:
            x = layer(x)
            
            # Pooling espacial (média ao longo do comprimento da sequência)
            # Produz um vetor representativo de tamanho [batch, hidden_dim]
            # que engloba o "estado latente do conceito" daquela camada.
            pooled_x = x.mean(dim=1)
            layer_activations.append(pooled_x)

        return x, layer_activations
