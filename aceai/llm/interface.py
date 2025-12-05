from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, BinaryIO, Literal, Required, Self, TypedDict

from msgspec import UNSET, field
from msgspec.structs import asdict

from aceai.interface import Record, Struct, Unset
from aceai.tools.interface import ToolSpec


class LLMToolCall(Record):
    """Normalized representation of an LLM-triggered tool/function call."""

    name: str
    arguments: str
    type: Literal["function", "mcp", "custom"] = "function"
    call_id: str | None = None


class LLMToolCallDelta(Record):
    """Incremental tool call update emitted during streaming."""

    id: str
    arguments_delta: str


class LLMMessage(Struct):
    """Typed chat message used across LLM providers."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[LLMToolCall] | None = None
    tool_call_id: str | None = None

    def __ior__(self, other: "LLMMessage | str") -> Self:
        if isinstance(other, str):
            other = LLMMessage(self.role, content=other)
        elif self.role != other.role:
            raise ValueError(f"Can't merge {other}, {self.role=}, {other.role=}")
        self.content += other.content
        return self

    def asdict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls is not None:
            result["tool_calls"] = [asdict(tc) for tc in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


class ResponseFormat(Record):
    """Desired response format, e.g., JSON with optional schema."""

    type: Literal["json_object", "text", "json_schema"] = "text"
    schema: Unset[dict[str, Any]] = UNSET
    json_schema: Unset[dict[str, Any]] = UNSET


class LLMRequestMeta(TypedDict, total=False):
    model: str


class LLMRequest(TypedDict, total=False):
    """Unified request bag accepted by all LLM adapters."""

    messages: Required[list[LLMMessage]]
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop: list[str]
    tools: list[ToolSpec]
    tool_choice: Literal["auto", "none"] | str
    response_format: ResponseFormat
    stream: bool
    metadata: LLMRequestMeta


class LLMUsage(Record):
    """Token accounting for a single completion."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(Record):
    """Provider-agnostic completion payload."""

    id: str | None = None
    model: str | None = None
    text: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    usage: Unset[LLMUsage] = UNSET


class LLMStreamChunk(Record):
    """Streaming delta emitted by adapters."""

    text_delta: Unset[str] = UNSET
    tool_call_delta: Unset[LLMToolCallDelta] = UNSET
    response: Unset[LLMResponse] = UNSET
    error: Unset[str] = UNSET


class LLMProviderBase(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat conversation with a unified request."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Preferred streaming API returning structured deltas."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the provider's default model identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_stream_model(self) -> str:
        """Return the provider's default streaming model identifier."""
        raise NotImplementedError

    @abstractmethod
    async def stt(self, filename: str, file: BinaryIO, *, model: str) -> str:
        """Speech-to-text for an audio file. Default impl not provided."""
        raise NotImplementedError
