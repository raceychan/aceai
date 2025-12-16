"""Shared LLM data models and provider contracts."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, BinaryIO, Literal, Required, Self, TypedDict

from msgspec import UNSET, field
from msgspec.structs import asdict

from aceai.interface import MessageRole, Record, Struct, Unset
from aceai.tools.interface import ToolSpec


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

    metadata: dict[str, Any]
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
    elif isinstance(val, list):
        return [value_to_msgpart(v) for v in val]
    else:
        raise ValueError(f"Unsupported value {val}")


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

    content: list[LLMMessagePart] = field(default_factory=list)
    """Structured message content; plain strings are not accepted."""

    def __ior__(self, other: "LLMMessage") -> Self:
        if self.role != other.role:
            raise ValueError(f"Can't merge {other}, {self.role=}, {other.role=}")
        self.content.extend(other.content)
        return self

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def build(cls, role: MessageRole, content: SupportedValueType):
        if isinstance(content, str):
            return cls(role=role, content=[value_to_msgpart(content)])
        elif isinstance(content, dict):
            return cls(role=role, content=[content])
        elif isinstance(content, list):
            return cls(role=role, content=content)
        else:
            raise NotImplementedError


class LLMToolCallMessage(LLMMessage, kw_only=True):
    """Assistant message variant that batches tool call declarations."""

    type: Literal["function_call"] = "function_call"
    """Assistant subtype emitted when tool calls are present."""

    role: Literal["assistant"] = "assistant"
    """Fixed assistant role for compatibility with chat providers."""

    tool_calls: list[LLMToolCall] | None = None
    """Structured tool invocations emitted within this turn."""

    def asdict(self) -> dict[str, Any]:
        res = super().asdict()
        if self.tool_calls is not None:
            res["tool_calls"] = [tc.asdict() for tc in self.tool_calls]
        return res

    @classmethod
    def build(
        cls,
        val: SupportedValueType,
        tool_calls: list[LLMToolCall],
    ):
        content = parts_factory(val)
        return cls(content=content, tool_calls=tool_calls)


class LLMToolUseMessage(LLMMessage, kw_only=True):
    """Synthetic tool response message sent back to the model."""

    role: Literal["tool"] = "tool"
    """Marks the message as tool output when relayed to the model."""

    name: str
    """tool/function label associated with the message."""

    call_id: str
    """Identifier of the `LLMToolCall` whose output is being returned."""

    def asdict(self) -> dict[str, Any]:
        res = super().asdict()
        if self.call_id is not None:
            res["tool_call_id"] = self.call_id
        return res

    @classmethod
    def build(cls, val: SupportedValueType, name: str, call_id: str):
        return cls(content=parts_factory(val), name=name, call_id=call_id)


class LLMResponseFormat(Record):
    """Response format hint shared between adapters and providers."""

    type: Literal["json_object", "text", "json_schema"] = "text"
    """Desired response style: plain text, generic JSON, or schema-constrained JSON."""

    schema: Unset[dict[str, Any]] = UNSET
    """JSON Schema payload for providers that expect a `schema` field."""

    json_schema: Unset[dict[str, Any]] = UNSET
    """Alternate schema slot for providers that expect `json_schema`."""


class LLMRequestMeta(TypedDict, total=False):
    """Adapter-specific metadata attached to a request."""

    model: str
    """Explicit provider model identifier overriding adapter defaults."""


class LLMRequest(TypedDict, total=False):
    """Unified request bag accepted by all LLM adapters."""

    messages: Required[list[LLMMessage]]
    """Ordered chat history supplied to the provider."""

    temperature: float
    """Randomness control applied by providers that support it."""

    top_p: float
    """Nucleus sampling parameter when available."""

    top_k: int
    """Highest-ranked token shortlist size for providers that expose it."""

    max_tokens: int
    """Upper bound on tokens the provider may generate for this call."""

    stop: list[str]
    """List of stop sequences requested by the caller."""

    tools: list[ToolSpec]
    """All tools/functions the model is allowed to call."""

    tool_choice: Literal["auto", "none"] | str
    """Tool selection policy or explicit tool name."""

    response_format: LLMResponseFormat
    """Structured response preferences (text vs. JSON schema)."""

    stream: bool
    """Flag toggling streaming vs. one-shot completion."""

    metadata: LLMRequestMeta
    """Adapter-specific overrides such as the concrete model name."""


class LLMUsage(Record):
    """Token accounting for a single completion."""

    input_tokens: int | None = None
    """Prompt-side token count reported by the provider."""

    output_tokens: int | None = None
    """Completion-side token count reported by the provider."""

    total_tokens: int | None = None
    """Convenience sum when the provider supplies a precomputed total."""


class LLMCitationRef(Record):
    """Minimal citation/ref grounding metadata attached to a segment."""

    id: str | None = None
    """Provider-supplied identifier (e.g., citation footnote)."""

    source: str | None = None
    """Source label or document identifier when available."""

    start: int | None = None
    """Inclusive character start offset within the source, if provided."""

    end: int | None = None
    """Exclusive character end offset within the source, if provided."""

    url: str | None = None
    """Optional public URL pointing at the referenced material."""

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
    """Elapsed wall time for this attempt in milliseconds."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata kept for debugging/telemetry."""


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

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form provider-specific metadata for forward compatibility."""

    media: "LLMGeneratedMedia | None" = None
    """Optional generated media payload for non-text segments."""


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


class LLMResponse(Record):
    """Provider-agnostic completion payload."""

    id: str | None = None
    """Opaque provider-generated response identifier."""

    model: str | None = None
    """Model name actually used to produce the completion."""

    text: str = ""
    """Aggregated assistant text returned by the provider."""

    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    """Tool/function invocations extracted from the provider output."""

    usage: Unset[LLMUsage] = UNSET
    """Token accounting info when supplied by the provider."""

    segments: list[LLMSegment] = field(default_factory=list[LLMSegment])
    """Structured segments capturing provider output with annotations."""

    provider_meta: list[LLMProviderMeta] = field(default_factory=list[LLMProviderMeta])
    """All provider attempts (including retries/failovers) for this request."""

    raw_events: list[dict[str, Any]] = field(default_factory=list)
    """Provider-native event payloads preserved for debugging."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Adapter-defined extension fields for advanced consumers."""


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

    chunk: LLMStreamChunk
    """Legacy chunk payload for callers that only need text/tool deltas."""

    segments: list[LLMSegment] = field(default_factory=list[LLMSegment])
    """Structured segments associated with this event."""

    provider_meta: list[LLMProviderMeta] = field(default_factory=list[LLMProviderMeta])
    """Provider attempt metadata as of this event emission."""

    raw_event: dict[str, Any] | None = None
    """Provider-native event payload preserved for debugging."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Adapter-defined extension fields for advanced consumers."""


class LLMProviderBase(ABC):
    """Interface for LLM providers that bridge vendor SDKs to this contract."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a chat conversation with a unified request."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Preferred streaming API returning structured events."""
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

    @property
    def modality(self) -> "LLMProviderModality":
        """Return the provider's supported modalities (defaults to text-only)."""
        return LLMProviderModality()

    @abstractmethod
    async def stt(self, filename: str, file: BinaryIO, *, model: str) -> str:
        """Speech-to-text for an audio file. Default impl not provided."""
        raise NotImplementedError


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
