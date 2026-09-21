"""Synthetic evaluation dataset fixtures for testing."""

import numpy as np

from mechanistic_router.models.pool import MODEL_POOL, get_model_accuracy
from mechanistic_router.models.types import TaskComplexity
from mechanistic_router.schemas.eval import EvalCase


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
