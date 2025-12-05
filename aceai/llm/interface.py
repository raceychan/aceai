"""
OpenAI adapter and LLM provider abstraction.

This module centralizes interactions with the OpenAI SDK and exposes a small
provider interface so upstream application code doesn't depend on the SDK
directly.

Notes
- We re-export OpenAI exception types so callers can catch them via this module
  without importing the SDK directly.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, Required, TypedDict, Unpack

import msgspec

from aceai.interface import JsonSchema

from .interface import ChatResponse, LLMRequest


class ToolCallFunction(msgspec.Struct):
    name: str
    arguments: str


class ToolCall(msgspec.Struct):
    id: str
    function: ToolCallFunction
    type: Literal["function"] = "function"


class ChatMessage(msgspec.Struct):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatResponse(msgspec.Struct):
    content: str
    tool_calls: list[ToolCall] = msgspec.field(default_factory=list)


class LLMRequest(TypedDict, total=False):
    model: Required[str]
    messages: Required[list[ChatMessage]]
    temperature: float
    max_tokens: int
    top_p: float
    json_schema: JsonSchema
    tools: list[dict[str, Any]]
    tool_choice: dict[str, Any]


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def get_response(
        self,
        **request: Unpack[LLMRequest],
    ) -> ChatResponse:
        """Send chat messages and return the assistant response plus tool calls."""
        raise NotImplementedError
