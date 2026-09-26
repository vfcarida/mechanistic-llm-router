"""LiteLLM Universal Provider Dispatcher Module."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import litellm

from ..schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)


class ProviderDispatchError(RuntimeError):
    """Raised when an upstream model provider fails or is unreachable."""

    def __init__(self, target_model: str, message: str, status_code: int = 502):
        super().__init__(f"Provider dispatch failed for target_model='{target_model}': {message}")
        self.target_model = target_model
        self.message = message
        self.status_code = status_code


class LiteLLMDispatcher:
    """Universal API Gateway dispatcher leveraging LiteLLM abstraction.

    Handles outbound and inbound calls to major proprietary providers (OpenAI, Anthropic,
    AWS Bedrock, Google Vertex AI) and local open-weights servers (Ollama, vLLM).
    Integrates exponential backoff retries, rate limit recovery, and typed error propagation.
    """

    def __init__(
        self,
        max_retries: int = 3,
        cooldown_seconds: float = 5.0,
        drop_params: bool = False,
    ):
        """Initializes dispatcher with retries, cooldown, and parameter-dropping settings."""
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.drop_params = drop_params
        litellm.drop_params = drop_params
        if drop_params:
            logger.info(
                "litellm.drop_params explicitly enabled: "
                "unsupported provider parameters will be dropped."
            )

    async def dispatch(
        self, target_model: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Executes async completion call via LiteLLM to specified target model.

        Args:
            target_model: Target provider model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
            request: Standardized ChatCompletionRequest object.

        Returns:
            Populated ChatCompletionResponse object.

        Raises:
            ProviderDispatchError: When upstream provider fails after max retries.
        """
        messages_dict = [msg.model_dump(exclude_none=True) for msg in request.messages]

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                # Call litellm async completion endpoint
                response: Any = await litellm.acompletion(
                    model=target_model,
                    messages=messages_dict,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stream=False,
                )

                # Format response into Pydantic schema
                content = ""
                if hasattr(response, "choices") and response.choices:
                    content = getattr(response.choices[0].message, "content", "") or ""

                usage_obj = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 10))
                completion_tokens = int(getattr(usage_obj, "completion_tokens", 20))
                total_tokens = int(
                    getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens)
                )

                return ChatCompletionResponse(
                    id=getattr(response, "id", "chatcmpl-mocked"),
                    model=target_model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=content,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    ),
                    router_strategy="LiteLLMDispatcher",
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Dispatch attempt %d/%d for target_model=%s failed: %s",
                    attempt + 1,
                    self.max_retries,
                    target_model,
                    str(exc),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.cooldown_seconds * (2**attempt) * 0.1)

        raise ProviderDispatchError(
            target_model=target_model,
            message=str(last_exc) if last_exc else "No dispatch attempts configured.",
            status_code=502,
        ) from last_exc

    async def dispatch_stream(
        self, target_model: str, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Executes streaming completion call via LiteLLM and yields SSE events.

        Yields:
            Formatted Server-Sent Event (SSE) strings like 'data: {...}\\n\\n' followed
            by 'data: [DONE]\\n\\n'.

        Raises:
            ProviderDispatchError: When upstream provider fails during initialization.
        """
        messages_dict = [msg.model_dump(exclude_none=True) for msg in request.messages]
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await litellm.acompletion(
                    model=target_model,
                    messages=messages_dict,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stream=True,
                )
                if hasattr(response, "__aiter__"):
                    async for chunk in response:
                        content = ""
                        if hasattr(chunk, "choices") and chunk.choices:
                            delta = getattr(chunk.choices[0], "delta", None)
                            content = getattr(delta, "content", "") or ""
                        chunk_id = getattr(chunk, "id", "chatcmpl-stream")
                        data_payload = json.dumps(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "model": target_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        yield f"data: {data_payload}\n\n"
                elif hasattr(response, "__iter__"):
                    for chunk in response:
                        content = ""
                        if hasattr(chunk, "choices") and chunk.choices:
                            delta = getattr(chunk.choices[0], "delta", None)
                            content = getattr(delta, "content", "") or ""
                        chunk_id = getattr(chunk, "id", "chatcmpl-stream")
                        data_payload = json.dumps(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "model": target_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        yield f"data: {data_payload}\n\n"

                yield "data: [DONE]\n\n"
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Streaming dispatch attempt %d/%d for target_model=%s failed: %s",
                    attempt + 1,
                    self.max_retries,
                    target_model,
                    str(exc),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.cooldown_seconds * (2**attempt) * 0.1)

        raise ProviderDispatchError(
            target_model=target_model,
            message=str(last_exc) if last_exc else "Streaming dispatch failed.",
            status_code=502,
        ) from last_exc


__all__ = ["LiteLLMDispatcher", "ProviderDispatchError"]
