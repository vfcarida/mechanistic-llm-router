import numpy as np
from ..models.pool import MODEL_POOL, get_model_accuracy
from ..models.types import TaskComplexity
from ..schemas.eval import EvalCase


def create_financial_dataset(n_samples: int = 200, seed: int = 42) -> list[EvalCase]:
    """Gera um dataset mock de avaliação financeira emitindo EvalCase instances.

    Os prompts mantêm o domínio financeiro sintético (24 prompts em 6 categorias).
    Os rótulos de complexidade são atribuídos exclusivamente como `reference_tier`
    para avaliação posterior, nunca fornecidos aos modelos durante o roteamento.

    Args:
        n_samples: Quantidade de amostras a gerar.
        seed: Semente aleatória para reprodutibilidade da distribuição.

    Returns:
        list[EvalCase]: Lista de instâncias de avaliação contendo prompt, desfechos
            por modelo, tabela de preços e reference_tier de ground-truth.
    """
    categories_and_prompts: dict[str, list[tuple[str, TaskComplexity]]] = {
        "consulta_fatura": [
            ("Qual o valor da minha fatura do cartão de crédito este mês?", TaskComplexity.ROUTINE),
            ("Gostaria de ver o detalhamento das últimas transações da fatura.", TaskComplexity.ROUTINE),
            ("Minha fatura veio com um valor que não reconheço. Pode verificar?", TaskComplexity.MODERATE),
            ("Preciso entender por que os juros rotativos foram aplicados na minha fatura dos últimos 3 meses e como isso impacta meu CET.", TaskComplexity.COMPLEX),
        ],
        "renegociacao": [
            ("Quero renegociar minha dívida do cartão.", TaskComplexity.ROUTINE),
            ("Quais as opções de parcelamento para minha dívida de R$5.000?", TaskComplexity.MODERATE),
            ("Considerando meu histórico de pagamentos, score e renda, qual seria a melhor estratégia de renegociação para minimizar juros compostos no longo prazo?", TaskComplexity.COMPLEX),
            ("Tenho três dívidas em atraso. Qual a prioridade de pagamento considerando taxas de juros compostos e impacto no meu score Serasa?", TaskComplexity.COMPLEX),
        ],
        "analise_credito": [
            ("Qual meu limite de crédito disponível?", TaskComplexity.ROUTINE),
            ("Quero solicitar aumento de limite. Qual minha elegibilidade?", TaskComplexity.MODERATE),
            ("Preciso de uma análise completa do meu perfil de crédito considerando DTI, LTV, histórico de utilização de crédito rotativo e projeção de capacidade de pagamento para os próximos 12 meses.", TaskComplexity.COMPLEX),
        ],
        "investimentos": [
            ("Quais os CDBs disponíveis hoje?", TaskComplexity.ROUTINE),
            ("Qual a diferença entre CDB pré e pós-fixado para meu perfil?", TaskComplexity.MODERATE),
            ("Monte uma estratégia de diversificação considerando meu perfil de risco moderado, horizonte de 5 anos, exposição atual em renda fixa e correlação entre classes de ativos no cenário macroeconômico atual.", TaskComplexity.COMPLEX),
        ],
        "pix_transferencias": [
            ("Quero fazer um Pix de R$100 para fulano.", TaskComplexity.ROUTINE),
            ("Houve uma transferência Pix que não reconheço. Pode verificar?", TaskComplexity.MODERATE),
            ("Preciso configurar uma política automatizada de Pix agendado com regras condicionais baseadas no saldo mínimo e projeções de fluxo de caixa.", TaskComplexity.COMPLEX),
        ],
        "seguros": [
            ("Quais seguros eu tenho contratados?", TaskComplexity.ROUTINE),
            ("Qual a cobertura do meu seguro residencial para danos elétricos?", TaskComplexity.MODERATE),
            ("Compare a relação custo-benefício entre minha apólice atual e três alternativas de mercado, considerando sinistralidade esperada, carência, franquia e minha exposição a riscos com base no CEP e perfil de uso.", TaskComplexity.COMPLEX),
        ],
    }

    all_prompts: list[tuple[str, str, TaskComplexity]] = []
    for category, prompts in categories_and_prompts.items():
        for text, complexity in prompts:
            all_prompts.append((category, text, complexity))

    # Agrupar prompts por complexidade para amostragem controlada
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

    for _ in range(n_samples):
        cplx = rng.choice(list(target_dist.keys()), p=list(target_dist.values()))
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
