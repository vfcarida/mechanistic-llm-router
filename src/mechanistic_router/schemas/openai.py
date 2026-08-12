"""Pydantic v2 OpenAI Chat Completions API Specification Schemas."""

import time
from typing import Literal
from pydantic import BaseModel, Field


class ChatCompletionMessage(BaseModel):
    """OpenAI message format payload."""

    role: Literal["system", "user", "assistant", "function"] = Field(..., description="Message author role.")
    content: str = Field(..., description="Text content of the message.")
    name: str | None = Field(default=None, description="Optional author identifier.")


class ChatCompletionRequest(BaseModel):
    """Standard OpenAI /v1/chat/completions HTTP POST payload."""

    model: str = Field(default="mechanistic-auto", description="Model or router profile requested.")
    messages: list[ChatCompletionMessage] = Field(..., min_length=1, description="Conversation history.")
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    stream: bool | None = Field(default=False, description="Whether to stream response tokens via SSE.")
    max_tokens: int | None = Field(default=None, gt=0)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = Field(default=None, description="End-user identifier.")


class UsageInfo(BaseModel):
    """Token consumption breakdown."""

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class ChatCompletionChoice(BaseModel):
    """Single completion choice in OpenAI response."""

    index: int = Field(default=0)
    message: ChatCompletionMessage = Field(...)
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] = Field(default="stop")


class ChatCompletionResponse(BaseModel):
    """Standard OpenAI /v1/chat/completions HTTP JSON response."""

    id: str = Field(..., description="Unique completion ID.")
    object: str = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(..., description="Actual model backend utilized for generation.")
    choices: list[ChatCompletionChoice] = Field(...)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    router_strategy: str | None = Field(default=None, description="Router strategy applied.")
