"""Shared LLM data models and provider contracts."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, BinaryIO, Literal, TypedDict

from msgspec import UNSET, field
from msgspec.structs import asdict
from opentelemetry.context import Context
from typing_extensions import Required, Self

from aceai.errors import AceAIImplementationError
from aceai.interface import MessageRole, Record, StrDict, Struct, Unset
from aceai.tools import IToolSpec


class LLMMessagePart(TypedDict, total=False):
    """Structured message chunk describing a single modality."""

    type: Required[Literal["text", "image", "audio", "file"]]
    """Semantic modality of this part."""

    data: Required[str | bytes]
    """Inline payload: string for text, bytes for binary modalities."""

    mime_type: str
    """Optional MIME type describing the payload."""

    url: str
    """External reference (HTTP URL, provider handle, etc.)."""

    metadata: StrDict
    """Provider- or caller-specific metadata."""


def value_to_msgpart(val: str | LLMMessagePart) -> LLMMessagePart:
    if isinstance(val, str):
        return LLMMessagePart(type="text", data=val)
    else:
        return val


type SupportedValueType = str | LLMMessagePart | list[LLMMessagePart]


def parts_factory(val: SupportedValueType):
    if isinstance(val, str):
        return [value_to_msgpart(val)]
    elif isinstance(val, dict):
        return [value_to_msgpart(val)]
    else:
        return [value_to_msgpart(v) for v in val]


class LLMUploadedAsset(Record, kw_only=True):
    """Descriptor for an uploaded asset that can be re-used by providers."""

    id: str
    """Provider-specific identifier for the uploaded asset."""

    mime_type: str
    """MIME type of the uploaded asset."""

    url: str | None = None
    """Optional URL referencing the uploaded asset."""


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

    role: MessageRole
    """Canonical chat role assigned to this turn."""

    content: list[LLMMessagePart] = field(default_factory=list[LLMMessagePart])
    """Structured message content; plain strings are not accepted."""

    def __ior__(self, other: Self) -> Self:
        if self.role != other.role:
            raise ValueError(f"Can't merge {other}, {self.role=}, {other.role=}")
        self.content.extend(other.content)
        return self

    def asdict(self) -> StrDict:
        # TODO: exclude UNSET values
        return asdict(self)

    @classmethod
    def build(cls, role: MessageRole, content: SupportedValueType):
        if isinstance(content, str):
            return cls(role=role, content=[value_to_msgpart(content)])
        elif isinstance(content, dict):
            return cls(role=role, content=[content])
        else:
            return cls(role=role, content=content)


class LLMToolCallMessage(LLMMessage, kw_only=True):
    """Assistant message variant that batches tool call declarations."""

    type: Literal["function_call"] = "function_call"
    """Assistant subtype emitted when tool calls are present."""

    role: Literal["system", "user", "assistant", "tool"] = "assistant"
    """Fixed assistant role for compatibility with chat providers."""

    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    """Always a list; empty list indicates a bug in the caller."""

    def asdict(self) -> StrDict:
        res = super().asdict()
        res["tool_calls"] = [tc.asdict() for tc in self.tool_calls]
        return res

    @classmethod
    def from_content(
        cls,
        content: SupportedValueType,
        tool_calls: list[LLMToolCall],
    ) -> "LLMToolCallMessage":
        content = parts_factory(content)
        return cls(content=content, tool_calls=tool_calls)


class LLMToolUseMessage(LLMMessage, kw_only=True):
    """Synthetic tool response message sent back to the model."""

    role: Literal["system", "user", "assistant", "tool"] = "tool"
    """Marks the message as tool output when relayed to the model."""

    name: str
    """tool/function label associated with the message."""

    call_id: str
    """Identifier of the `LLMToolCall` whose output is being returned."""

    @classmethod
    def from_content(
        cls, content: SupportedValueType, name: str, call_id: str
    ) -> "LLMToolUseMessage":
        return cls(content=parts_factory(content), name=name, call_id=call_id)


class LLMResponseFormat(Record):
    """Response format hint shared between adapters and providers."""

    type: Literal["json_object", "text", "json_schema"] = "text"
    """Desired response style: plain text, generic JSON, or schema-constrained JSON."""

    schema: Unset[StrDict] = UNSET
    """JSON Schema payload for providers that expect a `schema` field."""

    json_schema: Unset[StrDict] = UNSET
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


class LLMReasoningConfigSnapshot(Record, kw_only=True):
    """Provider-reported reasoning configuration applied to a response."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    """Requested reasoning depth tier (provider-specific semantics)."""

    summary: Literal["auto", "concise", "detailed"] | None = None
    """Whether the provider emitted a reasoning summary."""

    generate_summary: Literal["auto", "concise", "detailed"] | None = None
    """Alias used by some providers; preserved for compatibility."""

    encrypted_content: bool | None = None
    """Whether encrypted reasoning blobs were included by the provider."""


class LLMReasoningMeta(Record, kw_only=True):
    """Reasoning-related metadata surfaced by the provider."""

    config: LLMReasoningConfigSnapshot | None = None
    """Provider-reported reasoning config for this response."""

    tokens: int | None = None
    """Provider-reported reasoning token count (if available)."""


class LLMProviderModality(Record, kw_only=True):
    """Simple capability flags describing provider modality support."""

    text_in: bool = True
    text_out: bool = True
    image_in: bool = False
    image_out: bool = False
    audio_in: bool = False
    audio_out: bool = False
    file_in: bool = False
    file_out: bool = False


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

    tools: list[IToolSpec]
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

    provider_name: str | None = None
    """Optional provider identifier for this citation payload."""


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

    response_id: str | None = None
    """Optional provider-native response identifier for this attempt."""


class LLMGeneratedMedia(Record, kw_only=True):
    """Descriptor for media emitted by the model."""

    type: Literal["image", "audio", "file"]
    """Modality of the generated asset."""

    mime_type: str
    """MIME type of the generated asset."""

    url: str | None = None
    """Optional URL referencing the media."""

    data: bytes | None = None
    """Inline media bytes when immediately consumable."""


class LLMToolCallSegmentMeta(Record, kw_only=True):
    """Metadata for a tool call segment."""

    call_id: str
    tool_name: str | None = None
    is_delta: bool = False


class LLMImageSegmentMeta(Record, kw_only=True):
    """Metadata for an image segment."""

    item_id: str
    status: Literal["in_progress", "completed", "generating", "failed"] | None = None
    output_index: int | None = None
    partial_index: int | None = None
    sequence_number: int | None = None


class LLMReasoningSegmentMeta(Record, kw_only=True):
    """Metadata for a reasoning segment."""

    item_id: str
    kind: Literal["summary", "content"]
    index: int
    status: Literal["in_progress", "completed", "incomplete"] | None = None


type LLMSegmentMeta = LLMToolCallSegmentMeta | LLMImageSegmentMeta | LLMReasoningSegmentMeta


class LLMSegment(Record):
    """Structured slice of provider output with annotations."""

    type: Literal[
        "text",
        "reasoning",
        "tool_call",
        "citation",
        "error",
        "image",
        "audio",
        "file",
        "other",
    ]
    """Semantic segment type to help consumers interpret content."""

    content: str
    """Raw content payload for this segment."""

    citations: list[LLMCitationRef] = field(default_factory=list[LLMCitationRef])
    """Structured citation metadata attached to this segment."""

    safety: list[LLMSafetyAnnotation] = field(default_factory=list[LLMSafetyAnnotation])
    """Structured safety annotations attached to this segment."""

    meta: LLMSegmentMeta | None = None
    """Optional typed metadata for this segment."""

    media: LLMGeneratedMedia | None = None
    """Optional generated media payload for non-text segments."""

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

    status: str | None = None
    """Optional provider status label (e.g., completed, incomplete)."""

    reasoning: LLMReasoningMeta | None = None
    """Optional reasoning-related metadata (usage/config), separate from segments."""


class LLMStreamChunk(Record):
    """Streaming delta emitted by adapters."""

    text_delta: Unset[str] = UNSET
    """Partial text token(s) produced during streaming."""

    tool_call_delta: Unset[LLMToolCallDelta] = UNSET
    """Incremental tool call arguments emitted during streaming."""

    media: Unset[LLMGeneratedMedia] = UNSET
    """Media payload emitted during streaming (e.g., partial image)."""

    response: Unset[LLMResponse] = UNSET
    """Optional final response snapshot delivered when the stream completes."""

    error: Unset[str] = UNSET
    """Error message surfaced by the provider mid-stream."""


class LLMStreamEvent(Record):
    """Provider-agnostic streaming event with structured metadata."""

    event_type: Literal[
        "response.output_text.delta",
        "response.function_call_arguments.delta",
        "response.completed",
        "response.error",
        "response.media",
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


class LLMProviderBase(ABC):
    """Interface for LLM providers that bridge vendor SDKs to this contract."""

    @abstractmethod
    async def complete(
        self, request: LLMRequest, *, trace_ctx: Context | None = None
    ) -> LLMResponse:
        """Complete a chat conversation with a unified request."""
        raise AceAIImplementationError(
            f"{self.__class__.__name__} must implement complete()"
        )

    @abstractmethod
    def stream(
        self, request: LLMRequest, *, trace_ctx: Context | None = None
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Preferred streaming API returning structured events."""
        raise AceAIImplementationError(
            f"{self.__class__.__name__} must implement stream()"
        )

    @property
    def modality(self) -> LLMProviderModality:
        """Return the provider's supported modalities (defaults to text-only)."""
        return LLMProviderModality()

    @abstractmethod
    async def stt(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str,
        prompt: str | None = None,
        trace_ctx: Context | None = None,
    ) -> str:
        """Speech-to-text for an audio file. Default impl not provided."""
        raise NotImplementedError
