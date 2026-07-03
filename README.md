<div align="center">
  <h1>⚡ Cost-Optimal-Mechanistic-Router</h1>
  <p><em>Roteamento Inteligente de LLMs via Encoder-Target Decoupling e Análise de Prefill</em></p>

  ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
  ![Accelerated SVD](https://img.shields.io/badge/Backend-Native_GPU_SVD-8a2be2.svg)
  ![QA Edge Cases](https://img.shields.io/badge/QA_Tests-Passing-brightgreen.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
</div>

---

## 📖 Visão Executiva

A inteligência por trás de sistemas multi-modelo (Roteadores LLM) determina a viabilidade econômica de operações de IA em larga escala. Roteadores semânticos tradicionais roteiam com base em heurísticas rasas. O **Cost-Optimal-Mechanistic-Router** (antigo SharedTrunkNet) revoluciona este pipeline com o **Encoder-Target Decoupling**.

Em nossa versão mais recente, introduzimos suporte **Nativo a PyTorch para Decomposição de Valores Singulares (SVD)** e vetorização matemática. Isso elimina o gargalo catastrófico de quebra de grafo na CPU, permitindo inferências seguras de dimensionalidade topológica diretamente na GPU (ou aceleradas via CPU tensor math), além de simulação de tokenização determinística resistente a injeções de nulos e matrizes singulares corrompidas.

> [!TIP]
> **Impacto no Negócio:** Na PoC corporativa, a arquitetura alcançou **78,75% de redução de custo inferencial** com acurácia rigorosamente compatível ao "Oráculo Perfeito" (acerto >91%). A nova refatoração de Álgebra Linear nativa previne interrupções (Out-of-Memory / NaN Entropies) e impulsiona o throughput (TPS).

---

## 🧠 Arquitetura Mecanística

```mermaid
graph TD
    A[Prompt do Usuário] --> B(SharedTrunk Encoder<br><i>Simulador de Prefill O(1) determinístico</i>)
    
    subgraph Sinais Topológicos PyTorch
        B --> C[Dimensionalidade Efetiva d_eff<br><i>torch.linalg.svd & Shannon Entropy</i>]
        B --> D[Separabilidade de Fisher J<br><i>Gating Threshold</i>]
    end
    
    C --> E{Mechanistic Router}
    D --> E
    
    subgraph Orquestração e Seleção
        E -- Tarefa Rotineira --> F((SLM Local<br>$0.02))
        E -- Tarefa Moderada --> G((Mid-Tier LLM<br>$0.25))
        E -- Tarefa Complexa --> H((Frontier Oracle<br>$1.50))
    end
```

### O Motor Matemático (Sinais Extraídos)

1. **Dimensionalidade Efetiva ($d_{eff}$)**  
   Mede a entropia dos valores singulares do espaço latente, indicando a "complexidade cognitiva" imediata exigida para preencher o prompt. Se colapsa em uma dimensão, é uma tarefa rotineira.
2. **Separabilidade de Fisher ($J$)**  
   Filtra modelos incompetentes descartando aqueles cujos clusters simulados de falha e sucesso se sobrepõem topologicamente.
   
---

## ⚙️ Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior
- Ambiente virtual (`venv` ou `conda`)

```bash
# Clone o repositório
git clone https://github.com/empresa/cost-optimal-mechanistic-router.git
cd cost-optimal-mechanistic-router

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate no Windows

# Instale no modo de desenvolvimento com as dependências de testes
pip install -e .[dev]
```

### Rodando o Simulador (PoC)

O simulador embutido roda a suíte de avaliação completa contra um mock do *BERTaú* (Atendimento Financeiro).

```bash
# Com o ambiente virtual ativado
python scripts/run_poc.py
```

---

## 🧪 Qualidade e Testes (QA)

A arquitetura contém defesas estritas contra instabilidade numérica, com cobertura total das equações de entropia e SVD contra tensores rank-deficientes e infinitos.

```bash
# Rodar testes de unidade e condições de contorno
$env:PYTHONPATH="src"
pytest tests/ -v
```

---

<div align="center">
  <small>Desenvolvido com Foco em Performance Escalonável e IA Contemporânea.</small>
</div>
