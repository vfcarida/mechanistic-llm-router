"""Leakage Isolation Verification Suite (MLR-T03).

Asserts that:
1. RoutingRequest schema has no task_complexity field.
2. Routers accept only prompt (+ optional budget/metadata) without ground-truth labels.
3. Identical prompts yield identical routing decisions regardless of eval context (purity).
4. No call site in src/ passes reference_tier or per_model_outcome into route().
"""

import inspect
import pathlib

import pytest

from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TaskComplexity
from mechanistic_router.routers.cost_performance import CostPerformanceRouter
from mechanistic_router.routers.mechanistic import MechanisticRouter
from mechanistic_router.routers.semantic import SemanticRouter
from mechanistic_router.schemas.eval import EvalCase
from mechanistic_router.schemas.routing import RoutingRequest


def test_routing_request_has_no_task_complexity() -> None:
    """Verify RoutingRequest does not accept or define task_complexity."""
    fields = RoutingRequest.model_fields
    assert "task_complexity" not in fields
    assert "complexity" not in fields
    assert "reference_tier" not in fields

    # Attempting to initialize with task_complexity should ignore or fail
    req = RoutingRequest(prompt="Valid prompt")
    assert not hasattr(req, "task_complexity")


def test_route_signature_has_no_complexity_input() -> None:
    """Verify route() signatures on router strategies do not take ground-truth parameters."""
    for router_cls in [CostPerformanceRouter, SemanticRouter, MechanisticRouter]:
        sig = inspect.signature(router_cls.route)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "request" in params
        assert "complexity" not in params
        assert "task_complexity" not in params
        assert "reference_tier" not in params


@pytest.mark.asyncio
async def test_routing_purity_and_metadata_isolation() -> None:
    """Demonstrate router purity: identical prompt produces identical decision."""
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)

    prompt = "Qual o valor da minha fatura de cartão?"

    # Case A: EvalCase tagged as ROUTINE
    case_a = EvalCase(
        prompt=prompt,
        per_model_outcome={"SLM-BERTau-Local": 0.95, "LLM-Frontier-Oracle": 0.99},
        price_table={"SLM-BERTau-Local": 0.02, "LLM-Frontier-Oracle": 1.50},
        reference_tier=TaskComplexity.ROUTINE,
    )

    # Case B: EvalCase tagged as COMPLEX with completely inverted outcomes
    case_b = EvalCase(
        prompt=prompt,
        per_model_outcome={"SLM-BERTau-Local": 0.10, "LLM-Frontier-Oracle": 0.99},
        price_table={"SLM-BERTau-Local": 0.02, "LLM-Frontier-Oracle": 1.50},
        reference_tier=TaskComplexity.COMPLEX,
    )

    decision_a = await router.route(RoutingRequest(prompt=case_a.prompt))
    decision_b = await router.route(RoutingRequest(prompt=case_b.prompt))

    # The router does not observe the eval metadata; decisions must be identical
    assert decision_a.selected_model == decision_b.selected_model
    assert decision_a.strategy_used == decision_b.strategy_used
    assert decision_a.estimated_cost_usd == decision_b.estimated_cost_usd


def test_source_tree_has_no_eval_leakage_to_route() -> None:
    """Scan source tree to verify no call site passes eval metadata to route()."""
    src_dir = pathlib.Path(__file__).parent.parent / "src"
    py_files = list(src_dir.rglob("*.py"))

    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        assert "route(reference_tier" not in content
        assert "route(per_model_outcome" not in content
        assert "route(..., reference_tier" not in content
        # Ensure task_complexity is not on request in routers
        if "routers" in str(py_file):
            assert "request.task_complexity" not in content
