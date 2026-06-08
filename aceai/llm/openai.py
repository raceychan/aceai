import base64
from functools import singledispatchmethod
import json
from importlib.util import find_spec
import time
from typing import Any, AsyncGenerator, BinaryIO, Literal, TypedDict, cast
from warnings import warn

from msgspec import Struct, convert, field, to_builtins
from opentelemetry import trace
from opentelemetry.trace import SpanKind

if find_spec("openai") is None:
    raise RuntimeError(
        "openai provider requires the `openai` package. "
        "Install with `pip install 'aceai[openai]'`."
    )

from openai import APIError, APIStatusError, AsyncOpenAI
from openai.types.responses import FunctionToolParam
from openai.types.responses.response import Response
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_image_gen_call_completed_event import (
    ResponseImageGenCallCompletedEvent,
)
from openai.types.responses.response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    ResponseOutputMessage,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_web_search_call_completed_event import (
    ResponseWebSearchCallCompletedEvent,
)
from openai.types.responses.response_web_search_call_in_progress_event import (
    ResponseWebSearchCallInProgressEvent,
)
from openai.types.responses.response_web_search_call_searching_event import (
    ResponseWebSearchCallSearchingEvent,
)

from aceai.llm.errors import (
    AceAIConfigurationError,
    AceAIRuntimeError,
    AceAIValidationError,
    LLMProviderError,
)
from aceai.llm.interface import UNSET, StrDict, Unset, is_set
from aceai.llm.tracing import get_trace_ctx

from .models import (
    LLMGeneratedMedia,
    LLMHostedToolAction,
    LLMHostedToolSegmentMeta,
    LLMHostedToolSpec,
    LLMImageSegmentMeta,
    LLMInput,
    LLMMessage,
    LLMMessagePart,
    LLMProviderBase,
    LLMProviderMeta,
    LLMProviderModality,
    ReasoningConfig,
    LLMReasoningConfigSnapshot,
    LLMReasoningMeta,
    LLMReasoningSegmentMeta,
    LLMResponse,
    LLMResponseFormat,
    LLMSegment,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallDelta,
    LLMToolCallMessage,
    LLMToolCallSegmentMeta,
    LLMToolSpec,
    LLMToolUseMessage,
    LLMUsage,
)
from .tool_spec import IToolSpec


OpenAIModel = str
WEB_SEARCH_SOURCES_INCLUDE = "web_search_call.action.sources"


class OpenAIMeta(TypedDict, total=False):
    model: OpenAIModel
    stream_model: OpenAIModel
    reasoning: ReasoningConfig


class OpenAIPayload(Struct, kw_only=True):
    messages: list[LLMMessage]
    temperature: Unset[float] = UNSET
    top_p: Unset[float] = UNSET
    top_k: Unset[int] = UNSET
    max_tokens: Unset[int] = UNSET
    stop: Unset[list[str]] = UNSET
    tools: Unset[list[LLMToolSpec]] = UNSET
    tool_choice: Unset[Literal["auto", "none"] | str] = UNSET
    response_format: Unset[LLMResponseFormat] = UNSET
    stream: Unset[bool] = UNSET
    metadata: Unset[OpenAIMeta] = UNSET

    @classmethod
    def from_input(cls, llm_input: LLMInput) -> "OpenAIPayload":
        messages = llm_input["messages"]
        if not isinstance(messages, list):
            raise TypeError("OpenAIPayload.messages must be a list[LLMMessage]")
        messages = cls._validate_messages(messages)
        payload: dict[str, Any] = {"messages": messages}

        if "temperature" in llm_input:
            temperature = llm_input["temperature"]
            if type(temperature) is not float:
                raise TypeError("OpenAIPayload.temperature must be float")
            payload["temperature"] = temperature
        if "top_p" in llm_input:
            top_p = llm_input["top_p"]
            if type(top_p) is not float:
                raise TypeError("OpenAIPayload.top_p must be float")
            payload["top_p"] = top_p
        if "top_k" in llm_input:
            top_k = llm_input["top_k"]
            if type(top_k) is not int:
                raise TypeError("OpenAIPayload.top_k must be int")
            payload["top_k"] = top_k
        if "max_tokens" in llm_input:
            max_tokens = llm_input["max_tokens"]
            if type(max_tokens) is not int:
                raise TypeError("OpenAIPayload.max_tokens must be int")
            payload["max_tokens"] = max_tokens
        if "stop" in llm_input:
            stop = llm_input["stop"]
            if not isinstance(stop, list) or not all(
                type(item) is str for item in stop
            ):
                raise TypeError("OpenAIPayload.stop must be list[str]")
            payload["stop"] = stop
        if "tools" in llm_input:
            tools = llm_input["tools"]
            if not isinstance(tools, list):
                raise TypeError("OpenAIPayload.tools must be list[LLMToolSpec]")
            for tool in tools:
                if not isinstance(tool, IToolSpec) and not isinstance(
                    tool, LLMHostedToolSpec
                ):
                    raise TypeError(
                        "OpenAIPayload.tools must be IToolSpec or LLMHostedToolSpec instances"
                    )
            payload["tools"] = tools
        if "tool_choice" in llm_input:
            tool_choice = llm_input["tool_choice"]
            if type(tool_choice) is not str:
                raise TypeError("OpenAIPayload.tool_choice must be str")
            payload["tool_choice"] = tool_choice
        if "response_format" in llm_input:
            response_format = llm_input["response_format"]
            if not isinstance(response_format, LLMResponseFormat):
                raise TypeError(
                    "OpenAIPayload.response_format must be LLMResponseFormat"
                )
            payload["response_format"] = response_format
        if "stream" in llm_input:
            stream = llm_input["stream"]
            if type(stream) is not bool:
                raise TypeError("OpenAIPayload.stream must be bool")
            payload["stream"] = stream
        if "metadata" in llm_input:
            payload["metadata"] = convert(llm_input["metadata"], type=OpenAIMeta)

        return cls(**payload)

    @staticmethod
    def _validate_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
        validated: list[LLMMessage] = []
        for message in messages:
            if isinstance(message, LLMToolCallMessage):
                validated.append(convert(message.asdict(), type=LLMToolCallMessage))
                continue
            if isinstance(message, LLMToolUseMessage):
                validated.append(convert(message.asdict(), type=LLMToolUseMessage))
                continue
            if isinstance(message, LLMMessage):
                validated.append(convert(message.asdict(), type=LLMMessage))
                continue
            raise TypeError("OpenAIPayload.messages must be LLMMessage instances")
        return validated

    @property
    def tool_names(self) -> list[str]:
        if is_set(self.tools):
            names: list[str] = []
            for tool in self.tools:
                if isinstance(tool, IToolSpec):
                    names.append(tool.name)
                elif isinstance(tool, LLMHostedToolSpec):
                    names.append(f"{tool.provider_name}:{tool.native_name}")
            return names
        return []

    def asdict(self) -> StrDict:
        def _enc_hook(obj: object) -> object:
            if isinstance(obj, IToolSpec):
                return obj.generate_schema()
            if isinstance(obj, LLMHostedToolSpec):
                return obj.asdict()
            raise TypeError(
                f"Encoding objects of type {type(obj).__name__} is unsupported"
            )

        return cast(StrDict, to_builtins(self, enc_hook=_enc_hook))

    def build_response_kwargs(self) -> dict[str, Any]:
        payload = self.asdict()
        if not self.messages:
            raise AceAIValidationError("OpenAIPayload.messages is required")

        input_messages = self._format_messages_for_responses(self.messages)
        kwargs: dict[str, Any] = {"input": input_messages}
        request_metadata: OpenAIMeta = {}
        if is_set(self.metadata):
            request_metadata = self.metadata

        model_name = request_metadata.get("model")
        if not model_name:
            raise AceAIConfigurationError(
                "OpenAI request metadata must include a model identifier"
            )

        kwargs["model"] = model_name

        if "reasoning" in request_metadata:
            reasoning_cfg = request_metadata["reasoning"]
            if not self._supports_reasoning_summary(model_name):
                raise AceAIConfigurationError(
                    f"Model {model_name} does not support reasoning summaries"
                )
            kwargs["reasoning"] = reasoning_cfg
        elif self._supports_reasoning_summary(model_name):
            kwargs["reasoning"] = {"summary": "auto"}

        if "max_tokens" in payload:
            kwargs["max_output_tokens"] = self.max_tokens

        if "temperature" in payload and not model_name.startswith("gpt-5"):
            kwargs["temperature"] = self.temperature

        if "top_p" in payload:
            kwargs["top_p"] = self.top_p

        if "stop" in payload:
            warn(
                "OpenAI Responses API does not support stop sequences; ignoring request.stop"
            )

        if is_set(self.response_format):
            text_config = self._build_text_config(self.response_format)
            if text_config:
                kwargs["text"] = text_config

        if is_set(self.tools):
            kwargs["tools"] = [self._format_tool(tool) for tool in self.tools]
            if "openai:web_search" in self.tool_names:
                kwargs["include"] = [WEB_SEARCH_SOURCES_INCLUDE]

        if "tool_choice" in payload:
            kwargs["tool_choice"] = self.tool_choice

        return kwargs

    def _format_messages_for_responses(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, LLMToolUseMessage):
                formatted.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.call_id,
                        "output": self._coerce_text_content(
                            message.content, context="tool output"
                        ),
                    }
                )
            elif isinstance(message, LLMToolCallMessage):
                if message.content:
                    formatted.append(
                        {
                            "role": message.role,
                            "content": self._format_content_parts(
                                message.content, role=message.role
                            ),
                        }
                    )
                for tc in message.tool_calls or []:
                    formatted.append(tc.asdict())
            else:
                formatted.append(
                    {
                        "role": message.role,
                        "content": self._format_content_parts(
                            message.content, role=message.role
                        ),
                    }
                )

        return formatted

    def _coerce_text_content(
        self, content: list[LLMMessagePart], *, context: str
    ) -> str:
        text_parts: list[str] = []
        for part in content:
            if part["type"] != "text":
                raise ValueError(f"{context} only supports text parts")
            if "data" not in part:
                raise ValueError(f"{context} text parts must include data")
            data = part["data"]
            if not isinstance(data, str):
                raise TypeError(f"{context} text parts must be str")
            text_parts.append(data)
        return "".join(text_parts)

    def _format_content_parts(
        self, content: list[LLMMessagePart], *, role: str = "user"
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for part in content:
            match part["type"]:
                case "text":
                    text_type = "output_text" if role == "assistant" else "input_text"
                    if "data" not in part:
                        raise ValueError("Text message parts must include data")
                    data = part["data"]
                    if not isinstance(data, str):
                        raise TypeError("Text message parts must be str")
                    payload.append({"type": text_type, "text": data})
                case "image":
                    payload.append(self._format_image_part(part))
                case "audio":
                    raise ValueError("OpenAI Responses does not support audio input")
                case "file":
                    if role != "user":
                        raise ValueError(
                            "OpenAI file parts are only supported for user input"
                        )
                    payload.append(self._format_file_part(part))
                case _:
                    raise ValueError(f"Unsupported message part: {part['type']}")
        return payload

    def _format_image_part(self, part: LLMMessagePart) -> dict[str, Any]:
        image_url = part.get("url")
        binary = part.get("binary")
        if image_url is None and isinstance(binary, bytes):
            mime = part.get("mime_type", "image/png")
            b64 = base64.b64encode(binary).decode("ascii")
            image_url = f"data:{mime};base64,{b64}"
        if image_url is None:
            raise ValueError("Image parts must include `url` or `data`")
        return {
            "type": "input_image",
            "image_url": image_url,
            "detail": "auto",
        }

    def _format_file_part(self, part: LLMMessagePart) -> dict[str, Any]:
        if "url" in part:
            return {"type": "input_file", "file_url": part["url"]}
        metadata = part.get("metadata", {})
        file_id = metadata.get("file_id") if isinstance(metadata, dict) else None
        if isinstance(file_id, str) and file_id:
            return {"type": "input_file", "file_id": file_id}
        binary = part.get("binary")
        if not isinstance(binary, bytes):
            raise ValueError(
                "File parts must include `url`, `binary`, or metadata.file_id"
            )
        mime = part.get("mime_type", "application/octet-stream")
        filename = metadata.get("filename") if isinstance(metadata, dict) else None
        if not isinstance(filename, str) or filename == "":
            filename = "attachment"
        b64 = base64.b64encode(binary).decode("ascii")
        return {
            "type": "input_file",
            "filename": filename,
            "file_data": f"data:{mime};base64,{b64}",
        }

    def _build_text_config(
        self,
        response_format: LLMResponseFormat,
    ) -> ResponseTextConfigParam | None:
        match response_format.type:
            case "json_object":
                return {"format": {"type": "json_object"}}
            case "json_schema":
                assert is_set(response_format.schema)
                return {
                    "format": {
                        "type": "json_schema",
                        "schema": response_format.schema,
                        "name": "response_schema",
                    }
                }
            case "text":
                return None
            case _:
                raise AceAIValidationError(
                    f"Unsupported OpenAI response format type: {response_format.type}"
                )

    def _format_tool(self, tool: LLMToolSpec) -> dict[str, Any]:
        if isinstance(tool, IToolSpec):
            schema = tool.generate_schema()
            return cast(
                dict[str, Any],
                cast(
                    FunctionToolParam,
                    {
                        "type": "function",
                        **schema,
                    },
                ),
            )
        if tool.provider_name != "openai":
            raise AceAIConfigurationError(
                f"OpenAI cannot serialize hosted tool for provider {tool.provider_name}"
            )
        return {"type": tool.native_name, **tool.native_config}

    def _supports_reasoning_summary(self, model_name: str) -> bool:
        name = model_name.lower()
        return name.startswith(("o3", "o4", "gpt-5"))


class OpenAIParsedResponse(Struct, kw_only=True):
    text: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    reasoning_items: list[ResponseReasoningItem] = field(
        default_factory=list[ResponseReasoningItem]
    )
    segments: list[LLMSegment] = field(default_factory=list[LLMSegment])


class OpenAIOutputItemContext(Struct, kw_only=True):
    output_index: int
    parsed: OpenAIParsedResponse


class OpenAIResponseParser:
    def __init__(
        self,
        *,
        response: Response,
        provider_name: str,
        raise_on_unsupported_output_item: bool = True,
    ):
        self._response = response
        self._provider_name = provider_name
        self._raise_on_unsupported_output_item = raise_on_unsupported_output_item

    def parse(self) -> OpenAIParsedResponse:
        parsed = OpenAIParsedResponse(text=self._response_output_text())
        if parsed.text:
            parsed.segments.append(LLMSegment(type="text", content=parsed.text))

        for output_index, item in enumerate(self._response_output_items()):
            self._handle_output_item(
                item,
                context=OpenAIOutputItemContext(
                    output_index=output_index,
                    parsed=parsed,
                ),
            )

        return parsed

    @singledispatchmethod
    def _handle_output_item(
        self,
        item: object,
        *,
        context: OpenAIOutputItemContext,
    ) -> None:
        if not self._raise_on_unsupported_output_item:
            return
        raise AceAIRuntimeError(
            "Unsupported OpenAI response output item "
            f"{type(item).__name__} at output index {context.output_index}"
        )

    @_handle_output_item.register(ResponseOutputMessage)
    def _handle_output_message(
        self,
        item: ResponseOutputMessage,
        *,
        context: OpenAIOutputItemContext,
    ) -> None:
        if item.content and not context.parsed.text:
            raise AceAIRuntimeError(
                "OpenAI response output message was not reflected in output_text "
                f"at output index {context.output_index}"
            )

    @_handle_output_item.register(ResponseFunctionToolCall)
    def _handle_function_tool_call(
        self,
        item: ResponseFunctionToolCall,
        *,
        context: OpenAIOutputItemContext,
    ) -> None:
        call = self.tool_call_from_response_item(item)
        context.parsed.tool_calls.append(call)
        context.parsed.segments.append(
            LLMSegment(
                type="tool_call",
                content=call.arguments or "",
                meta=LLMToolCallSegmentMeta(
                    call_id=call.call_id,
                    tool_name=call.name,
                ),
            )
        )

    @_handle_output_item.register(ImageGenerationCall)
    def _handle_image_generation_call(
        self,
        item: ImageGenerationCall,
        *,
        context: OpenAIOutputItemContext,
    ) -> None:
        context.parsed.segments.append(
            LLMSegment(
                type="image",
                content="",
                media=self._image_call_to_media(item),
                meta=LLMImageSegmentMeta(
                    item_id=item.id,
                    status=item.status,
                    output_index=context.output_index,
                ),
            )
        )

    @_handle_output_item.register(ResponseFunctionWebSearch)
    def _handle_web_search_call(
        self,
        item: ResponseFunctionWebSearch,
        *,
        context: OpenAIOutputItemContext,
    ) -> None:
        context.parsed.segments.append(
            self.hosted_tool_segment_from_web_search_item(
                self._provider_name,
                item,
                output_index=context.output_index,
            )
        )

    @_handle_output_item.register(ResponseReasoningItem)
    def _handle_reasoning_item(
        self,
        item: ResponseReasoningItem,
        *,
        context: OpenAIOutputItemContext,
    ) -> None:
        context.parsed.reasoning_items.append(item)
        context.parsed.segments.extend(self._reasoning_segments_from_item(item))

    def _reasoning_segments_from_item(
        self, item: ResponseReasoningItem
    ) -> list[LLMSegment]:
        segments: list[LLMSegment] = []
        for idx, summary in enumerate(item.summary or []):
            segments.append(
                LLMSegment(
                    type="reasoning",
                    content=summary.text,
                    meta=LLMReasoningSegmentMeta(
                        item_id=item.id,
                        status=item.status,
                        kind="summary",
                        index=idx,
                    ),
                )
            )
        for idx, content in enumerate(item.content or []):
            segments.append(
                LLMSegment(
                    type="reasoning",
                    content=content.text,
                    meta=LLMReasoningSegmentMeta(
                        item_id=item.id,
                        status=item.status,
                        kind="content",
                        index=idx,
                    ),
                )
            )
        return segments

    def _image_call_to_media(self, item: ImageGenerationCall) -> LLMGeneratedMedia:
        data: bytes | None = None
        if item.result:
            data = base64.b64decode(item.result)
        return LLMGeneratedMedia(type="image", mime_type="image/png", data=data)

    @staticmethod
    def hosted_tool_segment_from_web_search_item(
        provider_name: str,
        item: ResponseFunctionWebSearch,
        *,
        output_index: int | None = None,
        sequence_number: int | None = None,
    ) -> LLMSegment:
        action = LLMHostedToolAction.from_payload(item.action.to_dict())
        return LLMSegment(
            type="hosted_tool",
            content=json.dumps({"action": action.asdict()}, separators=(",", ":")),
            meta=LLMHostedToolSegmentMeta(
                provider_name=provider_name,
                tool_name="web_search",
                item_id=item.id,
                status=item.status,
                output_index=output_index,
                sequence_number=sequence_number,
                action=action,
            ),
        )

    def _response_output_items(self) -> list[Any]:
        return list(self._response.output or [])

    def _response_output_text(self) -> str:
        if self._response.output is None:
            return ""
        return self._response.output_text or ""

    @staticmethod
    def tool_call_from_response_item(
        item: ResponseFunctionToolCall,
    ) -> LLMToolCall:
        call_id = item.call_id or item.id
        if call_id is None:
            raise AceAIRuntimeError(
                "OpenAI function call response did not include a call identifier"
            )
        return LLMToolCall(
            name=item.name,
            arguments=item.arguments,
            call_id=call_id,
        )


class OpenAIStreamEventMapper:
    def __init__(self, *, provider_name: str):
        self._provider_name = provider_name
        self.streamed_tool_calls: list[LLMToolCall] = []

    def map_event(
        self,
        event: ResponseStreamEvent,
        *,
        model_name: str,
    ) -> LLMStreamEvent | None:
        self._record_stream_state(event)

        text_delta: Unset[str] = UNSET
        tool_call_delta: Unset[LLMToolCallDelta] = UNSET
        error_value: Unset[str] = UNSET
        segments: list[LLMSegment] = []
        event_type: str | None = None

        match event:
            case ResponseTextDeltaEvent(delta=delta) if delta:
                text_delta = delta
                segments = [LLMSegment(type="text", content=delta)]
                event_type = "response.output_text.delta"
            case ResponseFunctionCallArgumentsDeltaEvent(
                delta=delta, item_id=item_id
            ) if delta and item_id:
                tool_call_delta = LLMToolCallDelta(
                    id=item_id,
                    arguments_delta=delta,
                )
                segments = [
                    LLMSegment(
                        type="tool_call",
                        content=delta,
                        meta=LLMToolCallSegmentMeta(
                            call_id=item_id,
                            is_delta=True,
                        ),
                    )
                ]
                event_type = "response.function_call_arguments.delta"
            case ResponseReasoningSummaryTextDeltaEvent(
                delta=delta,
                item_id=item_id,
                summary_index=summary_index,
            ) if delta:
                segments = [
                    LLMSegment(
                        type="reasoning",
                        content=delta,
                        meta=LLMReasoningSegmentMeta(
                            item_id=item_id,
                            kind="summary",
                            index=summary_index,
                            status="in_progress",
                            is_delta=True,
                        ),
                    )
                ]
                event_type = "response.reasoning.delta"
            case ResponseWebSearchCallInProgressEvent(
                item_id=item_id,
                output_index=output_index,
                sequence_number=sequence_number,
            ):
                segments = [
                    self._hosted_tool_segment(
                        item_id=item_id,
                        output_index=output_index,
                        sequence_number=sequence_number,
                        status="in_progress",
                    )
                ]
                event_type = "response.hosted_tool"
            case ResponseWebSearchCallSearchingEvent(
                item_id=item_id,
                output_index=output_index,
                sequence_number=sequence_number,
            ):
                segments = [
                    self._hosted_tool_segment(
                        item_id=item_id,
                        output_index=output_index,
                        sequence_number=sequence_number,
                        status="searching",
                    )
                ]
                event_type = "response.hosted_tool"
            case ResponseWebSearchCallCompletedEvent(
                item_id=item_id,
                output_index=output_index,
                sequence_number=sequence_number,
            ):
                segments = [
                    self._hosted_tool_segment(
                        item_id=item_id,
                        output_index=output_index,
                        sequence_number=sequence_number,
                        status="completed",
                    )
                ]
                event_type = "response.hosted_tool"
            case ResponseOutputItemDoneEvent(
                item=item,
                output_index=output_index,
                sequence_number=sequence_number,
            ) if isinstance(item, ResponseFunctionWebSearch):
                segments = [
                    OpenAIResponseParser.hosted_tool_segment_from_web_search_item(
                        self._provider_name,
                        item,
                        output_index=output_index,
                        sequence_number=sequence_number,
                    )
                ]
                event_type = "response.hosted_tool"
            case ResponseImageGenCallPartialImageEvent(partial_image_b64=partial_b64):
                media = self._media_from_base64(partial_b64)
                segments = [
                    LLMSegment(
                        type="image",
                        content="",
                        media=media,
                        meta=LLMImageSegmentMeta(
                            item_id=event.item_id,
                            output_index=event.output_index,
                            partial_index=event.partial_image_index,
                            sequence_number=event.sequence_number,
                        ),
                    )
                ]
                event_type = "response.media"
            case ResponseImageGenCallCompletedEvent():
                return None
            case ResponseErrorEvent(message=message):
                error_msg = message or "LLM stream error"
                error_value = error_msg
                segments = [LLMSegment(type="error", content=error_msg)]
                event_type = "response.error"
            case _:
                return None

        provider_meta = [
            LLMProviderMeta(
                provider_name=self._provider_name,
                model=model_name,
            )
        ]
        return LLMStreamEvent(
            event_type=event_type,
            text_delta=text_delta,
            tool_call_delta=tool_call_delta,
            error=error_value,
            segments=segments,
            provider_meta=provider_meta,
        )

    def _record_stream_state(self, event: ResponseStreamEvent) -> None:
        if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(
            event.item,
            ResponseFunctionToolCall,
        ):
            self.streamed_tool_calls.append(
                OpenAIResponseParser.tool_call_from_response_item(event.item)
            )

    def _hosted_tool_segment(
        self,
        *,
        item_id: str,
        output_index: int,
        sequence_number: int,
        status: Literal["in_progress", "searching", "completed", "failed"],
    ) -> LLMSegment:
        return LLMSegment(
            type="hosted_tool",
            content=_hosted_tool_content(status),
            meta=LLMHostedToolSegmentMeta(
                provider_name=self._provider_name,
                tool_name="web_search",
                item_id=item_id,
                status=status,
                output_index=output_index,
                sequence_number=sequence_number,
            ),
        )

    def _media_from_base64(self, payload: str) -> LLMGeneratedMedia:
        data = base64.b64decode(payload)
        return LLMGeneratedMedia(type="image", mime_type="image/png", data=data)


class OpenAI(LLMProviderBase):
    """OpenAI provider for LLM completions."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        default_meta: OpenAIMeta,
        provider_name: str = "openai",
    ):
        self._client = client
        self._default_metadata = convert(default_meta, type=OpenAIMeta)
        self._provider_name = provider_name
        self._tracer = trace.get_tracer("aceai.llm.openai")

    @property
    def modality(self) -> LLMProviderModality:
        return LLMProviderModality(image_in=True, image_out=True, file_in=True)

    async def stt(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str,
        prompt: str | None = None,
    ) -> str:
        """Transcribe audio using OpenAI Whisper (async)."""
        attributes = {
            "llm.provider": self.__class__.__name__,
            "llm.model": model,
            "llm.audio.filename": filename,
        }
        trace_ctx = get_trace_ctx()
        span = self._tracer.start_span(
            "openai.audio.transcriptions.create",
            kind=SpanKind.CLIENT,
            context=trace_ctx,
            attributes=attributes,
        )
        kwargs = {
            "model": model,
            "file": (filename, file),
        }
        if prompt is not None:
            kwargs["prompt"] = prompt
        try:
            result = await self._client.audio.transcriptions.create(**kwargs)
            return result.text
        finally:
            span.end()

    def _provider_meta_entry(
        self,
        *,
        model: str,
        latency_ms: float | None = None,
        response_id: str | None = None,
    ) -> LLMProviderMeta:
        return LLMProviderMeta(
            provider_name=self._provider_name,
            model=model,
            latency_ms=latency_ms,
            response_id=response_id,
        )

    def _patch_response_tool_calls(
        self,
        response: LLMResponse,
        tool_calls: list[LLMToolCall],
    ) -> LLMResponse:
        if response.tool_calls or not tool_calls:
            return response
        segments = list(response.segments)
        for call in tool_calls:
            segments.append(
                LLMSegment(
                    type="tool_call",
                    content=call.arguments or "",
                    meta=LLMToolCallSegmentMeta(
                        call_id=call.call_id,
                        tool_name=call.name,
                    ),
                )
            )
        return LLMResponse(
            id=response.id,
            model=response.model,
            text=response.text,
            tool_calls=tool_calls,
            usage=response.usage,
            segments=segments,
            provider_meta=response.provider_meta,
            status=response.status,
            reasoning=response.reasoning,
            reasoning_content=response.reasoning_content,
        )

    def _to_llm_response(
        self, response: Response, *, latency_ms: float | None = None
    ) -> LLMResponse:
        usage = response.usage
        usage_block: Unset[LLMUsage] = UNSET
        if usage:
            usage_block = self.build_usage(
                input_tokens=usage.input_tokens,
                cached_input_tokens=_cached_input_tokens(usage),
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        parser = OpenAIResponseParser(
            response=response,
            provider_name=self._provider_name,
        ).parse()
        response_model = str(response.model)
        reasoning_meta: LLMReasoningMeta | None = None
        reasoning_config = response.reasoning
        reasoning_config_snapshot: LLMReasoningConfigSnapshot | None = None
        if reasoning_config is not None:
            reasoning_config_snapshot = LLMReasoningConfigSnapshot(
                effort=reasoning_config.effort,
                summary=reasoning_config.summary,
                generate_summary=reasoning_config.generate_summary,
            )
        if usage and usage.output_tokens_details:
            details = usage.output_tokens_details
            reasoning_tokens = details.reasoning_tokens
            reasoning_meta = LLMReasoningMeta(
                config=reasoning_config_snapshot,
                tokens=reasoning_tokens,
            )
        if reasoning_meta is None and reasoning_config_snapshot is not None:
            reasoning_meta = LLMReasoningMeta(config=reasoning_config_snapshot)
        provider_meta = [
            self._provider_meta_entry(
                model=response_model,
                latency_ms=latency_ms,
                response_id=response.id,
            )
        ]
        return LLMResponse(
            id=response.id,
            model=str(response.model),
            text=parser.text,
            tool_calls=parser.tool_calls,
            usage=usage_block,
            segments=parser.segments,
            provider_meta=provider_meta,
            status=response.status,
            reasoning=reasoning_meta,
        )

    def _map_stream_event(
        self,
        event: ResponseStreamEvent,
        *,
        model_name: str,
    ) -> LLMStreamEvent | None:
        return OpenAIStreamEventMapper(provider_name=self._provider_name).map_event(
            event,
            model_name=model_name,
        )

    def _apply_default_meta(self, payload: OpenAIPayload) -> OpenAIPayload:
        """If no metadata, set defaults; if partial, fill in missing keys."""
        if not is_set(payload.metadata) or not payload.metadata:
            payload.metadata = self._default_metadata
        else:
            request_meta = payload.metadata
            for key, value in self._default_metadata.items():
                if key not in request_meta:
                    request_meta[key] = value
        return payload

    def _build_response_kwargs(self, payload: OpenAIPayload) -> dict[str, Any]:
        return payload.build_response_kwargs()

    async def complete(self, request: LLMInput) -> LLMResponse:
        """Complete using OpenAI Responses API."""
        payload = self._apply_default_meta(OpenAIPayload.from_input(request))
        params = self._build_response_kwargs(payload)
        start = time.perf_counter()
        tool_names = payload.tool_names
        attributes = {
            "llm.provider": self.__class__.__name__,
            "llm.model": params["model"],
            "llm.stream": False,
            "llm.tool_count": len(tool_names),
            "llm.tool_names": tool_names,
        }
        trace_ctx = get_trace_ctx()
        span = self._tracer.start_span(
            "openai.responses.create",
            kind=SpanKind.CLIENT,
            context=trace_ctx,
            attributes=attributes,
        )
        try:
            try:
                response: Response = await self._client.responses.create(**params)
            except APIError as err:
                raise _provider_error_from_openai(err) from err
            latency_ms = (time.perf_counter() - start) * 1000.0
            return self._to_llm_response(response, latency_ms=latency_ms)
        finally:
            span.end()

    async def stream(self, request: LLMInput) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream tokens and tool calls using OpenAI Responses streaming API."""
        payload = self._apply_default_meta(OpenAIPayload.from_input(request))
        kwargs = self._build_response_kwargs(payload)
        start = time.perf_counter()
        tool_names = payload.tool_names
        attributes = {
            "llm.provider": self.__class__.__name__,
            "llm.model": kwargs["model"],
            "llm.stream": True,
            "llm.tool_count": len(tool_names),
            "llm.tool_names": tool_names,
        }
        trace_ctx = get_trace_ctx()
        span = self._tracer.start_span(
            "openai.responses.stream",
            kind=SpanKind.CLIENT,
            context=trace_ctx,
            attributes=attributes,
        )
        stream_event_mapper = OpenAIStreamEventMapper(provider_name=self._provider_name)
        try:
            try:
                stream = await self._client.responses.create(**kwargs, stream=True)
                final_response: Response | None = None
                async with stream:
                    async for event in stream:
                        if isinstance(event, ResponseCompletedEvent):
                            final_response = event.response
                        mapped = stream_event_mapper.map_event(
                            event, model_name=kwargs["model"]
                        )
                        if mapped is None:
                            continue
                        yield mapped

                    if final_response is None:
                        raise AceAIRuntimeError(
                            "OpenAI stream did not include response.completed"
                        )
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    final_llm_response = self._to_llm_response(
                        final_response, latency_ms=latency_ms
                    )
                    final_llm_response = self._patch_response_tool_calls(
                        final_llm_response,
                        stream_event_mapper.streamed_tool_calls,
                    )
                    yield LLMStreamEvent(
                        event_type="response.completed",
                        response=final_llm_response,
                        segments=final_llm_response.segments,
                        provider_meta=final_llm_response.provider_meta,
                    )
            except APIError as err:
                raise _provider_error_from_openai(err) from err
        finally:
            span.end()


def _provider_error_from_openai(err: APIError) -> LLMProviderError:
    message = f"{type(err).__name__}: {err}"
    body = err.body if isinstance(err.body, dict) else {}
    error = body["error"] if isinstance(body.get("error"), dict) else body
    code = error.get("code") if isinstance(error, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    text = str(err)
    context_window = (
        code == "context_length_exceeded"
        or error_type == "context_length_exceeded"
        or "context_length_exceeded" in text
        or "Your input exceeds the context window of this model" in text
    )
    status_code = err.status_code if isinstance(err, APIStatusError) else None
    retryable = True
    if status_code is not None:
        retryable = status_code == 429 or 500 <= status_code
    return LLMProviderError(
        message,
        retryable=retryable,
        context_window=context_window,
        status_code=status_code,
    )


def _cached_input_tokens(usage: Any) -> int | None:
    details = getattr(usage, "input_tokens_details", None)
    if details is None:
        return None
    cached_tokens = getattr(details, "cached_tokens", None)
    if cached_tokens is None:
        return None
    if type(cached_tokens) is not int:
        raise TypeError("OpenAI cached input token usage must be int")
    return cached_tokens


def _hosted_tool_content(
    status: Literal["in_progress", "searching", "completed", "failed"],
) -> str:
    match status:
        case "in_progress":
            return "Preparing web search"
        case "searching":
            return "Searching the web"
        case "completed":
            return "Web search completed"
        case "failed":
            return "Web search failed"
        case _:
            raise ValueError("Unsupported hosted tool status")
