"""FastAPI Drop-in OpenAI-Compatible Gateway API Server."""

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import DEFAULT_CONFIG
from ..core.encoder import SharedTrunkEncoder
from ..models.pool import MODEL_POOL
from ..models.types import TargetModel
from ..observability.metrics import RouterMetrics
from ..routers.base import AbstractRouter
from ..routers.cost_performance import CostPerformanceRouter
from ..routers.mechanistic import MechanisticRouter
from ..routers.semantic import SemanticRouter
from ..schemas.openai import ChatCompletionRequest, ChatCompletionResponse
from ..schemas.routing import RoutingRequest
from .dispatcher import LiteLLMDispatcher, ProviderDispatchError

logger = logging.getLogger(__name__)

# Limits configuration
DEFAULT_MAX_PROMPT_CHARS = 10_000
DEFAULT_RATE_LIMIT_CAPACITY = 10.0
DEFAULT_RATE_LIMIT_REFILL_RATE = 5.0


def get_max_prompt_chars() -> int:
    """Reads and validates ROUTER_MAX_PROMPT_CHARS as a positive integer."""
    raw = os.getenv("ROUTER_MAX_PROMPT_CHARS")
    if raw is not None:
        try:
            val = int(raw.strip())
            if val > 0:
                return val
            logger.warning(
                "Invalid ROUTER_MAX_PROMPT_CHARS='%s' (must be > 0). Using default %d.",
                raw,
                DEFAULT_MAX_PROMPT_CHARS,
            )
        except ValueError:
            logger.warning(
                "Non-integer ROUTER_MAX_PROMPT_CHARS='%s'. Using default %d.",
                raw,
                DEFAULT_MAX_PROMPT_CHARS,
            )
    return DEFAULT_MAX_PROMPT_CHARS


class TokenBucketRateLimiter:
    """Thread-safe / async per-key token-bucket rate limiter with per-key concurrency locking."""

    def __init__(
        self,
        capacity: float = DEFAULT_RATE_LIMIT_CAPACITY,
        refill_rate: float = DEFAULT_RATE_LIMIT_REFILL_RATE,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, tuple[float, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_key_lock(self, key: str) -> asyncio.Lock:
        """Retrieves or creates a dedicated asyncio.Lock for the specified client key."""
        if key not in self._locks:
            async with self._registry_lock:
                if key not in self._locks:
                    self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def acquire(self, key: str, tokens_needed: float = 1.0) -> bool:
        """Attempts to consume tokens from the bucket associated with key."""
        lock = await self._get_key_lock(key)
        async with lock:
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
        self._locks.clear()


def extract_routing_prompt(messages: list[Any]) -> str:
    """Extracts prompt text from conversation messages for routing evaluation.

    If single-turn, returns the message content directly.
    If multi-turn, formats the conversation history (system, user, assistant) so that
    the routing strategy has context over preceding conversational turns.
    """
    if not messages:
        return ""
    if len(messages) == 1:
        return str(messages[0].content)
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)


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


# Module-level instances for fallback / backwards compatibility
encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)
semantic_router = SemanticRouter(MODEL_POOL, DEFAULT_CONFIG)
cost_performance_router = CostPerformanceRouter(MODEL_POOL, DEFAULT_CONFIG)
dispatcher = LiteLLMDispatcher()
metrics = RouterMetrics()
rate_limiter = TokenBucketRateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle manager to initialize OpenTelemetry prometheus exporter metrics."""
    app.state.encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    app.state.router = MechanisticRouter(app.state.encoder, MODEL_POOL, DEFAULT_CONFIG)
    app.state.semantic_router = SemanticRouter(MODEL_POOL, DEFAULT_CONFIG)
    app.state.cost_performance_router = CostPerformanceRouter(MODEL_POOL, DEFAULT_CONFIG)
    app.state.dispatcher = LiteLLMDispatcher()
    app.state.metrics = RouterMetrics()
    app.state.rate_limiter = TokenBucketRateLimiter()
    app.state.metrics.start_exporter()
    yield


app = FastAPI(
    title="Mechanistic LLM Router Gateway",
    description="Drop-in OpenAI-compatible Gateway performing ultra-low latency routing.",
    version="0.1.0",
    lifespan=lifespan,
)

# Populate initial default app state
app.state.encoder = encoder
app.state.router = router
app.state.semantic_router = semantic_router
app.state.cost_performance_router = cost_performance_router
app.state.dispatcher = dispatcher
app.state.metrics = metrics
app.state.rate_limiter = rate_limiter


# FastAPI Dependency Providers
def get_rate_limiter(request: Request) -> TokenBucketRateLimiter:
    """Retrieves TokenBucketRateLimiter from application state."""
    return getattr(request.app.state, "rate_limiter", rate_limiter)


def get_router(
    request: Request,
    x_router_strategy: str | None = Header(default=None),
) -> AbstractRouter:
    """Retrieves router strategy based on X-Router-Strategy header or defaults."""
    strategy = (x_router_strategy or "").lower().strip()
    if strategy in ("semantic", "semantic-auto"):
        return getattr(request.app.state, "semantic_router", semantic_router)
    if strategy in ("cost-performance", "cost-performance-auto"):
        return getattr(request.app.state, "cost_performance_router", cost_performance_router)
    if strategy in ("causal-probe", "causal-probe-auto", "causal"):
        return getattr(request.app.state, "causal_probe_router", router)
    return getattr(request.app.state, "router", router)


def get_dispatcher(request: Request) -> LiteLLMDispatcher:
    """Retrieves LiteLLMDispatcher from application state."""
    return getattr(request.app.state, "dispatcher", dispatcher)


def get_metrics(request: Request) -> RouterMetrics:
    """Retrieves RouterMetrics from application state."""
    return getattr(request.app.state, "metrics", metrics)


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
    limiter: TokenBucketRateLimiter = Depends(get_rate_limiter),
) -> str:
    """Verifies API key authentication and enforces per-key token-bucket rate limits."""
    expected_key = os.getenv("ROUTER_API_KEY")
    if not expected_key:
        logger.error("ROUTER_API_KEY environment variable is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": (
                        "Gateway API key is not configured. Set ROUTER_API_KEY environment "
                        "variable."
                    ),
                    "type": "configuration_error",
                    "code": "missing_router_api_key",
                }
            },
        )

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
    allowed = await limiter.acquire(provided_key)
    if not allowed:
        key_hash = hashlib.sha256(provided_key.encode("utf-8")).hexdigest()[:8]
        logger.warning(
            "Rate limit exceeded for client (key_hash=%s). Request rejected with 429.", key_hash
        )
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
        {"id": "causal-probe-auto", "object": "model", "owned_by": "mechanistic-router"},
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
    request_http: Request,
    _auth_key: str = Depends(verify_auth_and_rate_limit),
    router_dep: AbstractRouter = Depends(get_router),
    dispatcher_dep: LiteLLMDispatcher = Depends(get_dispatcher),
    metrics_dep: RouterMetrics = Depends(get_metrics),
) -> ChatCompletionResponse | StreamingResponse:
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
    max_chars = get_max_prompt_chars()
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

    # Allow request.model to also select strategy if not explicitly forced by header
    active_router = router_dep
    req_model = request.model.lower().strip()
    if req_model in ("semantic-auto", "semantic"):
        active_router = getattr(request_http.app.state, "semantic_router", semantic_router)
    elif req_model in ("cost-performance-auto", "cost-performance"):
        active_router = getattr(
            request_http.app.state, "cost_performance_router", cost_performance_router
        )
    elif req_model in ("causal-probe-auto", "causal-probe", "causal"):
        active_router = getattr(request_http.app.state, "causal_probe_router", router)

    # Extract user prompt from conversation history (multi-turn context aware)
    prompt_text = extract_routing_prompt(request.messages)
    routing_req = RoutingRequest(prompt=prompt_text)

    # Execute router decision
    start_route = time.perf_counter()
    decision = await active_router.route(routing_req)
    route_latency = (time.perf_counter() - start_route) * 1000.0

    # Calculate financial savings delta dynamically compared to strongest Oracle model
    cost_saved = compute_cost_savings(decision.selected_model, MODEL_POOL)

    # Handle streaming SSE responses when stream=True
    if request.stream:
        # Record initial route metrics
        metrics_dep.record_route(
            route_name=decision.selected_model,
            strategy=decision.strategy_used,
            router_latency_ms=route_latency,
            endpoint_latency_ms=0.0,
            cost_saved_usd=cost_saved,
        )
        stream_generator = dispatcher_dep.dispatch_stream(decision.selected_model, request)
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-router-strategy": f"{decision.strategy_used} -> {decision.selected_model}",
                "x-cost-saved-usd": str(cost_saved),
            },
        )

    # Dispatch completion to selected model backend (raises error on provider failure)
    start_endpoint = time.perf_counter()
    response = await dispatcher_dep.dispatch(decision.selected_model, request)
    endpoint_latency = (time.perf_counter() - start_endpoint) * 1000.0

    # Record OpenTelemetry metrics
    metrics_dep.record_route(
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
    "semantic_router",
    "cost_performance_router",
    "dispatcher",
    "metrics",
    "rate_limiter",
    "TokenBucketRateLimiter",
    "compute_cost_savings",
    "extract_routing_prompt",
    "get_router",
    "get_dispatcher",
    "get_metrics",
    "get_rate_limiter",
]
