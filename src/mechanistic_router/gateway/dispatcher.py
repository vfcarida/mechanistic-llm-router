"""LiteLLM Universal Provider Dispatcher Module."""

import asyncio
import logging
from typing import Any

import litellm

from ..schemas.openai import ChatCompletionRequest, ChatCompletionResponse

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
                return ChatCompletionResponse(
                    id=getattr(response, "id", "chatcmpl-mocked"),
                    model=target_model,
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response.choices[0].message.content or "",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    usage={
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 10),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 20),
                        "total_tokens": getattr(response.usage, "total_tokens", 30),
                    },
                    router_strategy="LiteLLMDispatcher",
                )
            except Exception as exc:
                logger.warning(
                    "Dispatch attempt %d/%d for target_model=%s failed: %s",
                    attempt + 1,
                    self.max_retries,
                    target_model,
                    str(exc),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.cooldown_seconds * (2**attempt) * 0.1)
                else:
                    raise ProviderDispatchError(
                        target_model=target_model,
                        message=str(exc),
                        status_code=502,
                    ) from exc

        raise ProviderDispatchError(
            target_model=target_model,
            message="Provider call failed after retries.",
            status_code=502,
        )


__all__ = ["LiteLLMDispatcher", "ProviderDispatchError"]
