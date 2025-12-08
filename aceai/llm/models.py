"""Shared LLM data models and provider contracts."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, BinaryIO, Literal, TypedDict

from msgspec import UNSET, field
from msgspec.structs import asdict
from typing_extensions import Required, Self

from aceai.errors import AceAIImplementationError, AceAIValidationError
from aceai.interface import Record, Struct, Unset
from aceai.tools.interface import ToolSpec


class LLMToolCall(Record, kw_only=True):
    """Normalized representation of an LLM-triggered tool/function call."""

    type: Literal["function_call", "mcp", "custom"] = "function_call"
    """Provider-declared call category; defaults to OpenAI-style function calls."""

    name: str
    """Registered tool name the model wants to invoke."""

    arguments: str
    """Raw JSON payload emitted by the model for the tool invocation."""

    call_id: str
    """Stable identifier used to correlate streaming deltas and tool outputs."""


class LLMToolCallDelta(Record):
    """Incremental tool call update emitted during streaming."""

    id: str
    """Identifier of the tool call being incrementally populated."""

    arguments_delta: str
    """Slice of JSON arguments appended to the buffered payload."""


class LLMMessage(Struct, kw_only=True):
    """Typed chat message that backs `LLMRequest.messages`."""

    role: Literal["system", "user", "assistant", "tool"]
    """Canonical chat role assigned to this turn."""

    content: str
    """Message body provided to or produced by the model."""

    name: str | None = None
    """Optional because only tool/function messages populate this label."""

    def __ior__(self, other: "LLMMessage | str") -> Self:
        if isinstance(other, str):
            other = LLMMessage(role=self.role, content=other)
        elif self.role != other.role:
            raise AceAIValidationError(
                f"Can't merge {other}, {self.role=}, {other.role=}"
            )
        self.content += other.content
        return self

    def asdict(self) -> dict[str, Any]:
        # TODO: exclude UNSET values
        return asdict(self)


class LLMToolCallMessage(LLMMessage, kw_only=True):
    """Assistant message variant that batches tool call declarations."""

    type: Literal["function_call"] = "function_call"
    """Assistant subtype emitted when tool calls are present."""

    role: Literal["system", "user", "assistant", "tool"] = "assistant"
    """Fixed assistant role for compatibility with chat providers."""

    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    """Always a list; empty list indicates a bug in the caller."""

    def asdict(self) -> dict[str, Any]:
        res = super().asdict()
        if self.tool_calls is not None:
            res["tool_calls"] = [tc.asdict() for tc in self.tool_calls]
        return res


class LLMToolUseMessage(LLMMessage, kw_only=True):
    """Synthetic tool response message sent back to the model."""

    role: Literal["system", "user", "assistant", "tool"] = "tool"
    """Marks the message as tool output when relayed to the model."""

    call_id: str
    """Identifier of the `LLMToolCall` whose output is being returned."""

    def asdict(self) -> dict[str, Any]:
        res = super().asdict()
        if self.call_id is not None:
            res["tool_call_id"] = self.call_id
        return res


class LLMResponseFormat(Record):
    """Response format hint shared between adapters and providers."""

    type: Literal["json_object", "text", "json_schema"] = "text"
    """Desired response style: plain text, generic JSON, or schema-constrained JSON."""

    schema: Unset[dict[str, Any]] = UNSET
    """JSON Schema payload for providers that expect a `schema` field."""

    json_schema: Unset[dict[str, Any]] = UNSET
    """Alternate schema slot for providers that expect `json_schema`."""


class ReasoningConfig(TypedDict, total=False):
    """Reasoning configuration supported by providers like OpenAI Responses."""

    effort: Literal["low", "medium", "high"]
    """Requested reasoning depth; providers map onto model-specific tiers."""

    summary: Literal["auto", "none"]
    """Whether the provider should emit a sanitized reasoning summary."""

    generate_summary: Literal["auto", "none"]
    """Alias supported by some providers; kept for forward compatibility."""

    encrypted_content: bool
    """Whether encrypted reasoning blobs should be included in the payload."""


class LLMRequestMeta(TypedDict, total=False):
    """Adapter-specific metadata attached to a request."""

    model: str
    """Provider-facing model identifier; adapter defaults must supply this."""

    stream_model: str
    """Optional streaming override used when the provider exposes separate models."""

    reasoning: ReasoningConfig
    """Provider-specific reasoning options (e.g., OpenAI Responses `reasoning`)."""


class LLMRequest(TypedDict, total=False):
    """Unified request bag accepted by all LLM adapters."""

    messages: Required[list[LLMMessage]]
    """Ordered chat history supplied to the provider."""

    temperature: float
    """Optional override for randomness; provider defaults apply when omitted."""

    top_p: float
    """Optional nucleus sampling override for providers that expose it."""

    top_k: int
    """Optional top-k setting; only a subset of providers honour it."""

    max_tokens: int
    """Optional completion cap; provider fallback limits apply when absent."""

    stop: list[str]
    """Optional stop-sequence list when the caller needs early termination."""

    tools: list[ToolSpec]
    """Optional tool registry; omit when the task is pure text completion."""

    tool_choice: Literal["auto", "none"] | str
    """Optional override when the caller must pin or disable tool usage."""

    response_format: LLMResponseFormat
    """Optional hint for providers that support JSON/structured output."""

    stream: bool
    """Optional streaming flag; defaults to provider-specific behaviour."""

    metadata: LLMRequestMeta
    """Optional adapter overrides (e.g., forcing a model name)."""


class LLMUsage(Record):
    """Token accounting for a single completion."""

    input_tokens: int | None = None
    """Optional because some providers never report prompt token usage."""

    output_tokens: int | None = None
    """Optional when providers omit completion-side accounting."""

    total_tokens: int | None = None
    """Optional; only present if the provider supplies the aggregate."""


class LLMCitationRef(Record):
    """Minimal citation/ref grounding metadata attached to a segment."""

    id: str | None = None
    """Optional because many providers don't emit stable citation IDs."""

    source: str | None = None
    """Optional label; absent when the provider only returns free text."""

    start: int | None = None
    """Optional offset because not all providers expose span data."""

    end: int | None = None
    """Optional offset because not all providers expose span data."""

    url: str | None = None
    """Optional public URL; only present when the provider shares resolved links."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Loose bag for provider-specific citation metadata."""


class LLMSafetyAnnotation(Record):
    """Minimal safety label metadata surfaced by providers."""

    category: str
    """Provider-reported safety category (e.g., hate, violence)."""

    level: Literal["none", "low", "medium", "high"] = "none"
    """Relative severity ranking emitted by the provider."""

    blocked: bool = False
    """Indicates whether the provider blocked or truncated the content."""

    reasons: list[str] = field(default_factory=list[str])
    """Optional human-readable reasons or evidence strings."""


class LLMProviderMeta(Record):
    """Per-provider attempt metadata captured during completion."""

    provider_name: str
    """Human-friendly provider identifier (e.g., openai, anthropic)."""

    model: str
    """Model identifier used for this attempt."""

    latency_ms: float | None = None
    """Optional because some adapters cannot observe provider latency."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata kept for debugging/telemetry."""


class LLMSegment(Record):
    """Structured slice of provider output with annotations."""

    type: Literal["text", "reasoning", "tool_call", "citation", "error", "other"]
    """Semantic segment type to help consumers interpret content."""

    content: str
    """Raw content payload for this segment."""

    citations: list[LLMCitationRef] = field(default_factory=list[LLMCitationRef])
    """Structured citation metadata attached to this segment."""

    safety: list[LLMSafetyAnnotation] = field(default_factory=list[LLMSafetyAnnotation])
    """Structured safety annotations attached to this segment."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form provider-specific metadata for forward compatibility."""


class LLMResponse(Record):
    """Provider-agnostic completion payload."""

    id: str | None = None
    """Optional because several providers do not return stable response IDs."""

    model: str | None = None
    """Optional; adapters fall back to request defaults when providers omit."""

    text: str = ""
    """Aggregated assistant text returned by the provider."""

    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    """Tool/function invocations extracted from the provider output."""

    usage: Unset[LLMUsage] = UNSET
    """Unset when the provider omits token accounting details."""

    segments: list[LLMSegment] = field(default_factory=list[LLMSegment])
    """Structured segments capturing provider output with annotations."""

    provider_meta: list[LLMProviderMeta] = field(default_factory=list[LLMProviderMeta])
    """All provider attempts (including retries/failovers) for this request."""

    raw_events: list[dict[str, Any]] = field(default_factory=list)
    """Provider-native event payloads preserved for debugging."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Adapter-defined extension fields for advanced consumers."""


class LLMStreamEvent(Record):
    """Provider-agnostic streaming event with structured metadata."""

    event_type: Literal[
        "response.output_text.delta",
        "response.function_call_arguments.delta",
        "response.completed",
        "response.error",
    ]
    """Explicit event type mirroring the provider's semantic intent."""

    text_delta: Unset[str] = UNSET
    """Partial text token(s) produced during streaming."""

    tool_call_delta: Unset[LLMToolCallDelta] = UNSET
    """Incremental tool call arguments emitted during streaming."""

    response: Unset[LLMResponse] = UNSET
    """Optional final response snapshot delivered when the stream completes."""

    error: Unset[str] = UNSET
    """Provider-reported error message surfaced mid-stream."""

    segments: list[LLMSegment] = field(default_factory=list[LLMSegment])
    """Structured segments associated with this event."""

    provider_meta: list[LLMProviderMeta] = field(default_factory=list[LLMProviderMeta])
    """Provider attempt metadata as of this event emission."""

    raw_event: dict[str, Any] | None = None
    """Optional raw payload; only present when adapters expose vendor JSON."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Adapter-defined extension fields for advanced consumers."""


class LLMProviderBase(ABC):
    """Interface for LLM providers that bridge vendor SDKs to this contract."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat conversation with a unified request."""
        raise AceAIImplementationError(
            f"{self.__class__.__name__} must implement complete()"
        )

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Preferred streaming API returning structured events."""
        raise AceAIImplementationError(
            f"{self.__class__.__name__} must implement stream()"
        )

    @abstractmethod
    async def stt(self, filename: str, file: BinaryIO, *, model: str) -> str:
        """Speech-to-text for an audio file. Default impl not provided."""
        raise AceAIImplementationError(
            f"{self.__class__.__name__} must implement stt()"
        )
