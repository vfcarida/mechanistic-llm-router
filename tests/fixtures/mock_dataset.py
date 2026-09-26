"""Synthetic evaluation dataset fixtures for testing."""

from mechanistic_router.evaluation.dataset import create_financial_dataset
from mechanistic_router.models.pool import MODEL_POOL, get_model_accuracy
from mechanistic_router.models.types import TaskComplexity
from mechanistic_router.schemas.eval import EvalCase

__all__ = [
    "MODEL_POOL",
    "EvalCase",
    "TaskComplexity",
    "create_financial_dataset",
    "get_model_accuracy",
]
