"""FastAPI Drop-in OpenAI-Compatible Gateway API Server."""

import time
from typing import Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from ..config import DEFAULT_CONFIG, RouterConfig
from ..core.encoder import SharedTrunkEncoder
from ..models.pool import MODEL_POOL
from ..models.types import TaskComplexity
from ..observability.metrics import RouterMetrics
from ..routers.mechanistic import MechanisticRouter
from ..schemas.openai import ChatCompletionRequest, ChatCompletionResponse
from ..schemas.routing import RoutingRequest
from .dispatcher import LiteLLMDispatcher

app = FastAPI(
    title="Mechanistic LLM Router Gateway",
    description="Drop-in OpenAI-compatible API Gateway performing ultra-low latency mechanistic route selection.",
    version="0.1.0",
)

# Global singleton dependencies
encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)
dispatcher = LiteLLMDispatcher()
metrics = RouterMetrics()


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize OpenTelemetry prometheus exporter metrics."""
    metrics.start_exporter()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check status endpoint."""
    return {"status": "ok", "service": "mechanistic-llm-router-gateway"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible models list endpoint."""
    model_list = [
        {"id": "mechanistic-auto", "object": "model", "owned_by": "mechanistic-router"},
        {"id": "cost-performance-auto", "object": "model", "owned_by": "mechanistic-router"},
        {"id": "semantic-auto", "object": "model", "owned_by": "mechanistic-router"},
    ]
    for key in MODEL_POOL:
        model_list.append({"id": key, "object": "model", "owned_by": "target-pool"})
    return {"object": "list", "data": model_list}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Drop-in replacement for OpenAI /v1/chat/completions endpoint."""
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Messages list cannot be empty.",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "missing_messages",
                }
            },
        )

    # Extract user prompt from conversation history
    last_user_message = next(
        (msg.content for msg in reversed(request.messages) if msg.role == "user"),
        request.messages[-1].content,
    )

    # Infer complexity heuristic
    length = len(last_user_message.split())
    if length > 40 or "math" in last_user_message.lower() or "code" in last_user_message.lower():
        task_cplx = TaskComplexity.COMPLEX
    elif length > 15:
        task_cplx = TaskComplexity.MODERATE
    else:
        task_cplx = TaskComplexity.ROUTINE

    routing_req = RoutingRequest(prompt=last_user_message, task_complexity=task_cplx)

    # Execute router decision
    start_route = time.perf_counter()
    decision = await router.route(routing_req)
    route_latency = (time.perf_counter() - start_route) * 1000.0

    # Dispatch completion to selected model backend
    start_endpoint = time.perf_counter()
    response = await dispatcher.dispatch(decision.selected_model, request)
    endpoint_latency = (time.perf_counter() - start_endpoint) * 1000.0

    # Calculate financial savings delta compared to defaulted Oracle route ($1.50)
    oracle_cost = 1.50
    cost_saved = max(0.0, oracle_cost - decision.estimated_cost_usd)

    # Record OpenTelemetry metrics
    metrics.record_route(
        route_name=decision.selected_model,
        strategy=decision.strategy_used,
        router_latency_ms=route_latency,
        endpoint_latency_ms=endpoint_latency,
        cost_saved_usd=cost_saved,
    )

    response.router_strategy = f"{decision.strategy_used} -> {decision.selected_model}"
    return response
