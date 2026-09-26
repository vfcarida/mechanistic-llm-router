"""Dataset Partitioning and Adapters for Leakage-Free Benchmark Evaluation."""

import json
import os
import random

import numpy as np

from ..models.pool import MODEL_POOL, get_model_accuracy
from ..models.types import TaskComplexity
from ..schemas.eval import EvalCase


def split_dataset_prompt_disjoint(
    cases: list[EvalCase],
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[EvalCase], list[EvalCase], list[EvalCase]]:
    """Partitions an evaluation dataset into train, dev, and test splits with zero prompt overlap.

    Ensures that no prompt string present in the test evaluation split appears in either
    the train or dev sets, preventing data leakage into learned routers or threshold tuners.

    Args:
        cases: Complete list of EvalCase records.
        train_ratio: Fraction of unique prompts assigned to train.
        dev_ratio: Fraction of unique prompts assigned to dev.
        test_ratio: Fraction of unique prompts assigned to test.
        seed: Random seed for reproducible partitioning.

    Returns:
        tuple (train_cases, dev_cases, test_cases) with strictly disjoint prompt texts.
    """
    if not cases:
        return [], [], []

    # Identify unique prompt texts preserving order
    unique_prompts = list(dict.fromkeys(case.prompt for case in cases))
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)

    n_prompts = len(unique_prompts)
    n_train = max(1, int(n_prompts * train_ratio))
    n_dev = max(1, int(n_prompts * dev_ratio))

    train_prompt_set = set(unique_prompts[:n_train])
    dev_prompt_set = set(unique_prompts[n_train : n_train + n_dev])
    test_prompt_set = set(unique_prompts[n_train + n_dev :])

    # If rounding left test set empty, ensure at least 1 prompt in test
    if not test_prompt_set and len(train_prompt_set) > 1:
        moved = train_prompt_set.pop()
        test_prompt_set.add(moved)

    train_cases = [c for c in cases if c.prompt in train_prompt_set]
    dev_cases = [c for c in cases if c.prompt in dev_prompt_set]
    test_cases = [c for c in cases if c.prompt in test_prompt_set]

    return train_cases, dev_cases, test_cases


def create_financial_dataset(n_samples: int = 200, seed: int = 42) -> list[EvalCase]:
    """Generate a synthetic financial evaluation dataset emitting EvalCase instances.

    Prompts maintain an illustrative synthetic banking domain (24 prompts across 6 categories).
    Complexity labels are assigned exclusively as `reference_tier` for post-routing
    evaluation, and are never supplied to models during runtime routing.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducible sampling distributions.

    Returns:
        list[EvalCase]: Evaluation instances containing prompt text, per-model outcomes,
            price table, and ground-truth reference_tier.
    """
    categories_and_prompts: dict[str, list[tuple[str, TaskComplexity]]] = {
        "consulta_fatura": [
            (
                "Qual o valor da minha fatura do cartão de crédito este mês?",
                TaskComplexity.ROUTINE,
            ),
            (
                "Gostaria de ver o detalhamento das últimas transações da fatura.",
                TaskComplexity.ROUTINE,
            ),
            (
                "Minha fatura veio com um valor que não reconheço. Pode verificar?",
                TaskComplexity.MODERATE,
            ),
            (
                "Preciso entender por que os juros rotativos foram aplicados na minha "
                "fatura dos últimos 3 meses e como isso impacta meu CET.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "renegociacao": [
            ("Quero renegociar minha dívida do cartão.", TaskComplexity.ROUTINE),
            (
                "Quais as opções de parcelamento para minha dívida de R$5.000?",
                TaskComplexity.MODERATE,
            ),
            (
                "Considerando meu histórico de pagamentos, score e renda, qual seria a "
                "melhor estratégia de renegociação para minimizar juros compostos no longo prazo?",
                TaskComplexity.COMPLEX,
            ),
            (
                "Tenho três dívidas em atraso. Qual a prioridade de pagamento considerando "
                "taxas de juros compostos e impacto no meu score Serasa?",
                TaskComplexity.COMPLEX,
            ),
        ],
        "analise_credito": [
            ("Qual meu limite de crédito disponível?", TaskComplexity.ROUTINE),
            (
                "Quero solicitar aumento de limite. Qual minha elegibilidade?",
                TaskComplexity.MODERATE,
            ),
            (
                "Preciso de uma análise completa do meu perfil de crédito considerando DTI, "
                "LTV, histórico de utilização de crédito rotativo e projeção de capacidade "
                "de pagamento para os próximos 12 meses.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "investimentos": [
            ("Quais os CDBs disponíveis hoje?", TaskComplexity.ROUTINE),
            (
                "Qual a diferença entre CDB pré e pós-fixado para meu perfil?",
                TaskComplexity.MODERATE,
            ),
            (
                "Monte uma estratégia de diversificação considerando meu perfil de risco "
                "moderado, horizonte de 5 anos, exposição atual em renda fixa e correlação "
                "entre classes de ativos no cenário macroeconômico atual.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "pix_transferencias": [
            ("Quero fazer um Pix de R$100 para fulano.", TaskComplexity.ROUTINE),
            (
                "Houve uma transferência Pix que não reconheço. Pode verificar?",
                TaskComplexity.MODERATE,
            ),
            (
                "Preciso configurar uma política automatizada de Pix agendado com regras "
                "condicionais baseadas no saldo mínimo e projeções de fluxo de caixa.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "seguros": [
            ("Quais seguros eu tenho contratados?", TaskComplexity.ROUTINE),
            (
                "Qual a cobertura do meu seguro residencial para danos elétricos?",
                TaskComplexity.MODERATE,
            ),
            (
                "Compare a relação custo-benefício entre minha apólice atual e três "
                "alternativas de mercado, considerando sinistralidade esperada, carência, "
                "franquia e minha exposição a riscos com base no CEP e perfil de uso.",
                TaskComplexity.COMPLEX,
            ),
        ],
    }

    all_prompts: list[tuple[str, str, TaskComplexity]] = []
    for category, prompts in categories_and_prompts.items():
        for text, complexity in prompts:
            all_prompts.append((category, text, complexity))

    # Group prompts by complexity for controlled sampling
    prompts_by_complexity = {
        TaskComplexity.ROUTINE: [p for p in all_prompts if p[2] == TaskComplexity.ROUTINE],
        TaskComplexity.MODERATE: [p for p in all_prompts if p[2] == TaskComplexity.MODERATE],
        TaskComplexity.COMPLEX: [p for p in all_prompts if p[2] == TaskComplexity.COMPLEX],
    }

    target_dist = {
        TaskComplexity.ROUTINE: 0.55,
        TaskComplexity.MODERATE: 0.30,
        TaskComplexity.COMPLEX: 0.15,
    }

    price_table = {name: model.cost for name, model in MODEL_POOL.items()}
    rng = np.random.RandomState(seed)
    eval_cases: list[EvalCase] = []

    complexity_keys = list(target_dist.keys())
    complexity_probs = list(target_dist.values())

    for _ in range(n_samples):
        choice_idx = int(rng.choice(len(complexity_keys), p=complexity_probs))
        cplx = complexity_keys[choice_idx]
        pool = prompts_by_complexity[cplx]
        idx = rng.randint(0, len(pool))
        _, text, complexity = pool[idx]

        per_model_outcome = {
            name: get_model_accuracy(model, complexity) for name, model in MODEL_POOL.items()
        }

        eval_cases.append(
            EvalCase(
                prompt=text,
                per_model_outcome=per_model_outcome,
                price_table=price_table,
                reference_tier=complexity,
            )
        )

    return eval_cases


def load_synthetic_eval_dataset(n_samples: int = 300, seed: int = 42) -> list[EvalCase]:
    """Generates synthetic benchmark evaluation cases based on financial BERTaú-domain scenarios.

    Args:
        n_samples: Total number of sampled evaluation records.
        seed: Random seed.

    Returns:
        List of EvalCase instances with per-model ground-truth outcomes and pricing.
    """
    return create_financial_dataset(n_samples=n_samples, seed=seed)


def load_routerbench_eval_dataset(path: str) -> list[EvalCase]:
    """Loads precomputed evaluation cases from RouterBench dataset format (JSON or JSONL).

    Args:
        path: Filesystem path to precomputed RouterBench benchmark file.

    Returns:
        List of loaded EvalCase records.

    Raises:
        FileNotFoundError: If the RouterBench dataset file is unavailable offline.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RouterBench dataset file not found at '{path}'. "
            "Please use '--dataset synthetic' or provide a precomputed RouterBench benchmark file."
        )

    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cases.append(EvalCase(**item))
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    cases.append(EvalCase(**item))

    return cases


__all__ = [
    "split_dataset_prompt_disjoint",
    "load_synthetic_eval_dataset",
    "load_routerbench_eval_dataset",
]
