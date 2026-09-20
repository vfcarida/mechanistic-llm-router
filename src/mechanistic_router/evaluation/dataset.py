"""Dataset Partitioning and Adapters for Leakage-Free Benchmark Evaluation."""

import json
import os
import random

from ..data.mock_dataset import create_financial_dataset
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
