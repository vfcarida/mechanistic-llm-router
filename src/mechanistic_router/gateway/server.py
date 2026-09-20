"""FastAPI Drop-in OpenAI-Compatible Gateway API Server."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from ..config import DEFAULT_CONFIG
from ..core.encoder import SharedTrunkEncoder
from ..models.pool import MODEL_POOL
from ..models.types import TargetModel
from ..observability.metrics import RouterMetrics
from ..routers.mechanistic import MechanisticRouter
from ..schemas.openai import ChatCompletionRequest, ChatCompletionResponse
from ..schemas.routing import RoutingRequest
from .dispatcher import LiteLLMDispatcher, ProviderDispatchError

logger = logging.getLogger(__name__)

# Security & limits configuration
DEFAULT_ROUTER_API_KEY = "test-router-key"
DEFAULT_MAX_PROMPT_CHARS = 10_000
DEFAULT_RATE_LIMIT_CAPACITY = 10.0
DEFAULT_RATE_LIMIT_REFILL_RATE = 5.0


class TokenBucketRateLimiter:
    """Thread-safe / async per-key token-bucket rate limiter."""

    def __init__(
        self,
        capacity: float = DEFAULT_RATE_LIMIT_CAPACITY,
        refill_rate: float = DEFAULT_RATE_LIMIT_REFILL_RATE,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, tokens_needed: float = 1.0) -> bool:
        """Attempts to consume tokens from the bucket associated with key."""
        async with self._lock:
            now = time.monotonic()
            if key not in self._buckets:
                self._buckets[key] = (self.capacity, now)

            tokens, last_time = self._buckets[key]
            elapsed = now - last_time
            tokens = min(self.capacity, tokens + elapsed * self.refill_rate)

            if tokens >= tokens_needed:
                self._buckets[key] = (tokens - tokens_needed, now)
                return True
            else:
                self._buckets[key] = (tokens, now)
                return False

    def reset(self) -> None:
        """Clears all rate limit buckets (useful for tests)."""
        self._buckets.clear()


def compute_cost_savings(
    selected_model_name: str, pool: dict[str, TargetModel] = MODEL_POOL
) -> float:
    """Calculates dynamic cost savings USD versus the strongest/oracle model in the pool."""
    if not pool:
        return 0.0
    oracle_cost = max(m.cost for m in pool.values())
    selected = pool.get(selected_model_name)
    selected_cost = selected.cost if selected else 0.0
    return max(0.0, round(oracle_cost - selected_cost, 6))


# Global singleton dependencies
encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)
dispatcher = LiteLLMDispatcher()
metrics = RouterMetrics()
rate_limiter = TokenBucketRateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize OpenTelemetry prometheus exporter metrics."""
    metrics.start_exporter()
    yield


app = FastAPI(
    title="Mechanistic LLM Router Gateway",
    description="Drop-in OpenAI-compatible Gateway performing ultra-low latency routing.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ProviderDispatchError)
async def provider_dispatch_error_handler(
    request: Request, exc: ProviderDispatchError
) -> JSONResponse:
    """Handles upstream provider failures without fabricating assistant responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": "provider_dispatch_error",
                "target_model": exc.target_model,
                "code": "upstream_provider_failure",
            }
        },
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Formats HTTPException into standard OpenAI-compatible top-level error dict."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "api_error",
                "code": "http_error",
            }
        },
    )


async def verify_auth_and_rate_limit(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """Verifies API key authentication and enforces per-key token-bucket rate limits."""
    expected_key = os.getenv("ROUTER_API_KEY", DEFAULT_ROUTER_API_KEY)

    provided_key: str | None = None
    if x_api_key:
        provided_key = x_api_key.strip()
    elif authorization:
        if authorization.startswith("Bearer "):
            provided_key = authorization[7:].strip()
        else:
            provided_key = authorization.strip()

    if not provided_key or provided_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid or missing API key.",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )

    # Check token-bucket rate limit for the authenticated key
    allowed = await rate_limiter.acquire(provided_key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "message": "Rate limit exceeded. Too many requests.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    return provided_key


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
async def chat_completions(
    request: ChatCompletionRequest,
    response_http: Response,
    _auth_key: str = Depends(verify_auth_and_rate_limit),
) -> ChatCompletionResponse:
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

    # Check max prompt chars cap
    max_chars = int(os.getenv("ROUTER_MAX_PROMPT_CHARS", str(DEFAULT_MAX_PROMPT_CHARS)))
    total_prompt_chars = sum(len(msg.content) for msg in request.messages)
    if total_prompt_chars > max_chars:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "error": {
                    "message": (
                        f"Prompt length ({total_prompt_chars} chars) "
                        f"exceeds maximum allowed limit of {max_chars} characters."
                    ),
                    "type": "invalid_request_error",
                    "code": "prompt_too_large",
                }
            },
        )

    # Extract user prompt from conversation history
    last_user_message = next(
        (msg.content for msg in reversed(request.messages) if msg.role == "user"),
        request.messages[-1].content,
    )

    routing_req = RoutingRequest(prompt=last_user_message)

    # Execute router decision
    start_route = time.perf_counter()
    decision = await router.route(routing_req)
    route_latency = (time.perf_counter() - start_route) * 1000.0

    # Dispatch completion to selected model backend (raises error on provider failure)
    start_endpoint = time.perf_counter()
    response = await dispatcher.dispatch(decision.selected_model, request)
    endpoint_latency = (time.perf_counter() - start_endpoint) * 1000.0

    # Calculate financial savings delta dynamically compared to strongest Oracle model
    cost_saved = compute_cost_savings(decision.selected_model, MODEL_POOL)

    # Record OpenTelemetry metrics
    metrics.record_route(
        route_name=decision.selected_model,
        strategy=decision.strategy_used,
        router_latency_ms=route_latency,
        endpoint_latency_ms=endpoint_latency,
        cost_saved_usd=cost_saved,
    )

    response.router_strategy = f"{decision.strategy_used} -> {decision.selected_model}"
    response.cost_saved_usd = cost_saved
    response_http.headers["x-cost-saved-usd"] = str(cost_saved)
    return response


__all__ = [
    "app",
    "router",
    "dispatcher",
    "metrics",
    "rate_limiter",
    "TokenBucketRateLimiter",
    "compute_cost_savings",
]
