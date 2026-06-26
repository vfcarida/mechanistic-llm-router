# Cost-Optimal-Mechanistic-Router – Mechanistic LLM Router

Este repositório contém a Prova de Conceito (PoC) para o roteador mecanístico **Cost-Optimal-Mechanistic-Router**. Ao contrário dos roteadores semânticos tradicionais, que baseiam suas decisões no espaço de *embeddings* (o "significado" do prompt), esta abordagem captura a **complexidade intrínseca do processamento** através da análise do prefill (as ativações das camadas ocultas) num *encoder* leve.

A abordagem baseia-se no princípio de **Encoder-Target Decoupling** para tomar decisões rigorosas de custo-eficiência.

## 🎯 Por que Roteamento Mecanístico?

Roteadores de embeddings tradicionais freqüentemente falham em distinguir consultas curtas porém difíceis ("Quais as implicações fiscais de...") de consultas curtas triviais ("Qual meu saldo?"). 

Ao usar um simulador de *prefill*, o Cost-Optimal-Mechanistic-Router detecta assinaturas matemáticas que predizem com alta precisão se um SLM (Pequeno Modelo de Linguagem) conseguirá processar a requisição ou se ela precisará escalar para um modelo de fronteira (Frontier LLM).

## 🧮 Sinais Matemáticos Extraídos

1. **Dimensionalidade Efetiva ($d_{eff}$):** Mede a complexidade do raciocínio extraindo a entropia de Shannon sobre a distribuição de energia espectral (SVD) da matriz de ativação. Prompts que esmagam o SVD em múltiplas dimensões indicam que modelos baratos falharão.
2. **Separabilidade de Fisher ($J$):** Quantifica o quão bem as ativações ocultas agrupam os clusters de sucesso e falha para um determinado modelo. É usado como um *Portão de Competência*.

## ⚙️ Instalação

O projeto utiliza `pyproject.toml` e estrutura modular padrão Python:

```bash
# Clone o repositório
# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Instale no modo de desenvolvimento
pip install -e .[dev]
```

## 🚀 Executando a PoC

Para rodar a prova de conceito e o simulador financeiro BERTaú:

```bash
python scripts/run_poc.py
```

Isto executará o pipeline completo:
1. Geração de um dataset financeiro representativo (55% rotina, 30% moderado, 15% complexo).
2. Simulação do *SharedTrunkEncoder*.
3. Avaliação contra as rotas de *Oráculo* vs *Cost-Optimal-Mechanistic-Router*.
4. Relatório executivo de economia de custos.

## 🧪 Testes

A suíte de testes unitários cobre as propriedades matemáticas de $d_{eff}$ e Separabilidade de Fisher, além das árvores de decisão do roteador:

```bash
pytest tests/ -v
```

## 📐 Estrutura do Repositório

- `src/mechanistic_router/config.py`: Parâmetros e limites de roteamento.
- `src/mechanistic_router/core/`: Encoder e lógica de Roteamento (*Encoder-Target Decoupling*).
- `src/mechanistic_router/models/`: Definições de modelos alvo e custos.
- `src/mechanistic_router/signals/`: Funções puramente matemáticas ($d_{eff}$, Fisher J).
- `tests/`: Suíte de testes.
- `scripts/`: Entrypoints de execução.

## 📄 Licença

MIT License
