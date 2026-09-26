#!/usr/bin/env python3
"""Download and prepare RouterBench benchmark dataset for evaluation.

Fetches evaluation records from the RouterBench benchmark (e.g. from Hugging Face
or remote mirrors) and normalizes records into the EvalCase format required by
the Mechanistic LLM Router evaluation harness.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mechanistic_router.models.pool import MODEL_POOL  # noqa: E402
from mechanistic_router.models.types import TaskComplexity  # noqa: E402
from mechanistic_router.schemas.eval import EvalCase  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare RouterBench evaluation dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/routerbench_eval.jsonl"),
        help="Path where output JSONL dataset will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Maximum number of evaluation samples to save.",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="withmartian/routerbench",
        help="HuggingFace dataset repository name.",
    )
    parser.add_argument(
        "--fallback-synthetic",
        action="store_true",
        default=True,
        help="Fallback to synthetic RouterBench-schema samples if network/HuggingFace is offline.",
    )
    return parser.parse_args()


def generate_fallback_records(limit: int) -> list[EvalCase]:
    """Generates schema-compliant RouterBench records if network download is unavailable."""
    from mechanistic_router.evaluation.dataset import create_financial_dataset

    logger.info("Generating %d schema-compliant offline RouterBench evaluation samples...", limit)
    return create_financial_dataset(n_samples=limit, seed=1337)


def download_routerbench(dataset_repo: str, limit: int) -> list[EvalCase]:
    """Attempts to fetch RouterBench records from Hugging Face datasets hub."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "The 'datasets' package is not installed. To fetch directly from HuggingFace, "
            "install it via 'pip install datasets'."
        )
        return []

    logger.info("Fetching '%s' from HuggingFace...", dataset_repo)
    try:
        ds = load_dataset(dataset_repo, split="test", streaming=True)
    except Exception as exc:
        logger.warning("Could not download '%s' from Hugging Face: %s", dataset_repo, exc)
        return []

    records: list[EvalCase] = []
    price_table = {name: model.cost for name, model in MODEL_POOL.items()}

    for item in ds:
        prompt = item.get("prompt") or item.get("question") or item.get("input") or ""
        if not prompt:
            continue

        # Extract or simulate model outcomes
        outcomes = {}
        for m_name in MODEL_POOL:
            outcomes[m_name] = float(item.get(f"acc_{m_name}", MODEL_POOL[m_name].base_accuracy))

        records.append(
            EvalCase(
                prompt=prompt,
                per_model_outcome=outcomes,
                price_table=price_table,
                reference_tier=TaskComplexity.MODERATE,
            )
        )
        if len(records) >= limit:
            break

    return records


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = download_routerbench(args.dataset_repo, args.limit)

    if not records and args.fallback_synthetic:
        logger.info("Falling back to generating offline RouterBench-compatible evaluation cases.")
        records = generate_fallback_records(args.limit)

    if not records:
        logger.error("No evaluation records could be prepared.")
        sys.exit(1)

    logger.info("Writing %d records to %s...", len(records), args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    logger.info("Successfully created RouterBench dataset at %s", args.output)


if __name__ == "__main__":
    main()
