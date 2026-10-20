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

    Prompts cover an enterprise banking domain across 6 categories and 3 complexity tiers.
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
        "statement_inquiry": [
            (
                "What is my credit card statement balance this month?",
                TaskComplexity.ROUTINE,
            ),
            (
                "Show my recent credit card statement transactions.",
                TaskComplexity.ROUTINE,
            ),
            (
                "My statement has an unrecognized charge. Can you verify?",
                TaskComplexity.MODERATE,
            ),
            (
                "Explain why revolving compound interest was applied to my statement "
                "over the last three months and how it impacts my APR.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "debt_restructuring": [
            ("I want to renegotiate my card balance.", TaskComplexity.ROUTINE),
            (
                "What are the installment options for my $5,000 debt?",
                TaskComplexity.MODERATE,
            ),
            (
                "Given my payment history, credit score, and income, what is the "
                "optimal debt restructuring strategy to minimize compound interest long term?",
                TaskComplexity.COMPLEX,
            ),
            (
                "I have three overdue balances. What is the payment priority considering "
                "compound interest rates and credit bureau score impact?",
                TaskComplexity.COMPLEX,
            ),
        ],
        "credit_analysis": [
            ("What is my available credit limit?", TaskComplexity.ROUTINE),
            (
                "I want to request a credit limit increase. What is my eligibility?",
                TaskComplexity.MODERATE,
            ),
            (
                "Provide a comprehensive analysis of my credit profile considering DTI, "
                "LTV, revolving utilization history, and 12-month cash flow projection.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "investments": [
            ("Which fixed-income certificates are available today?", TaskComplexity.ROUTINE),
            (
                "What is the difference between fixed and floating rate "
                "certificates for my profile?",
                TaskComplexity.MODERATE,
            ),
            (
                "Build a portfolio diversification strategy considering my moderate risk "
                "profile, 5-year horizon, fixed-income exposure, and asset correlation "
                "under current macroeconomic conditions.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "instant_transfers": [
            ("Send a $100 instant transfer to John.", TaskComplexity.ROUTINE),
            (
                "There is an instant transfer I do not recognize. Can you verify?",
                TaskComplexity.MODERATE,
            ),
            (
                "Configure an automated scheduled transfer policy with conditional rules "
                "based on minimum balance thresholds and cash flow projections.",
                TaskComplexity.COMPLEX,
            ),
        ],
        "insurance": [
            ("Which insurance policies do I currently hold?", TaskComplexity.ROUTINE),
            (
                "What is the home insurance coverage for electrical damage?",
                TaskComplexity.MODERATE,
            ),
            (
                "Compare the cost-benefit ratio between my current insurance policy and "
                "three market alternatives, considering expected loss ratio, deductible, "
                "and regional risk exposure.",
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
    """Generates synthetic benchmark evaluation cases based on enterprise financial scenarios.

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
