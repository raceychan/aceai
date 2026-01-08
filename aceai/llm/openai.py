import base64
import time
from typing import Any, AsyncGenerator, BinaryIO, cast
from warnings import warn

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind

try:
    import openai  # type: ignore[unused-import]
except ImportError as exc:
    raise RuntimeError(
        "openai provider requires the `openai` package. "
        "Install with `uv add openai` or `pip install openai`."
    ) from exc

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam
from openai.types.responses.response import Response
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_image_gen_call_completed_event import (
    ResponseImageGenCallCompletedEvent,
)
from openai.types.responses.response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from aceai.errors import (
    AceAIConfigurationError,
    AceAIRuntimeError,
    AceAIValidationError,
)
from aceai.interface import MISSING, UNSET, Unset, is_present, is_set
from aceai.tools import IToolSpec

from .models import (
    LLMGeneratedMedia,
    LLMImageSegmentMeta,
    LLMMessage,
    LLMMessagePart,
    LLMProviderBase,
    LLMProviderMeta,
    LLMProviderModality,
    LLMReasoningConfigSnapshot,
    LLMReasoningMeta,
    LLMReasoningSegmentMeta,
    LLMRequest,
    LLMRequestMeta,
    LLMResponse,
    LLMResponseFormat,
    LLMSegment,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallDelta,
    LLMToolCallMessage,
    LLMToolCallSegmentMeta,
    LLMToolUseMessage,
    LLMUsage,
)


class OpenAI(LLMProviderBase):
    """OpenAI provider for LLM completions."""

    def __init__(self, client: AsyncOpenAI, *, default_meta: LLMRequestMeta):
        self._client = client
        self._default_metadata: LLMRequestMeta = default_meta
        self._tracer = trace.get_tracer("aceai.llm.openai")

    @property
    def modality(self) -> LLMProviderModality:
        return LLMProviderModality(image_in=True, image_out=True)

    async def stt(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str,
        prompt: str | None = None,
        trace_ctx: Context | None = None,
    ) -> str:
        """Transcribe audio using OpenAI Whisper (async)."""
        attributes = {
            "llm.provider": self.__class__.__name__,
            "llm.model": model,
            "llm.audio.filename": filename,
        }
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

    def _tool_names(self, request: LLMRequest) -> list[str]:
        if "tools" not in request:
            return []
        return [tool.name for tool in request["tools"]]

    def _build_base_response_kwargs(
        self,
        request: LLMRequest,
    ) -> dict[str, Any]:
        """Translate LLMRequest into OpenAI Responses kwargs."""
        if "messages" not in request:
            raise AceAIValidationError("LLMRequest.messages is required")

        input_messages = self._format_messages_for_responses(request["messages"])
        kwargs: dict[str, Any] = {"input": input_messages}
        request_metadata = request.get("metadata", {})

        model_name = request_metadata.get("model")
        if not model_name:
            raise AceAIConfigurationError(
                "OpenAI request metadata must include a model identifier"
            )

        kwargs["model"] = model_name

        if is_present(reasoning_cfg := request_metadata.get("reasoning", MISSING)):
            if not self._supports_reasoning_summary(model_name):
                raise AceAIConfigurationError(
                    f"Model {model_name} does not support reasoning summaries"
                )
            kwargs["reasoning"] = reasoning_cfg

        if is_present(max_tokens := request.get("max_tokens", MISSING)):
            kwargs["max_output_tokens"] = max_tokens

        if is_present(temperature := request.get("temperature", MISSING)):
            if not model_name.startswith("gpt-5"):
                kwargs["temperature"] = temperature

        if is_present(top_p := request.get("top_p", MISSING)):
            kwargs["top_p"] = top_p

        if is_present(request.get("stop", MISSING)):
            warn(
                "OpenAI Responses API does not support stop sequences; ignoring request.stop"
            )

        if is_present(response_format := request.get("response_format", MISSING)):
            text_config = self._build_text_config(response_format)
            if text_config:
                kwargs["text"] = text_config

        if is_present(tools := request.get("tools", MISSING)):
            kwargs["tools"] = [self._format_tool(tool) for tool in tools]

        if is_present(tool_choice := request.get("tool_choice", MISSING)):
            kwargs["tool_choice"] = tool_choice

        return kwargs

    def _format_messages_for_responses(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, Any]]:
        """Project internal chat messages into Responses API input items."""
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
                    data = part["data"]
                    if not isinstance(data, str):
                        raise TypeError("Text message parts must be str")
                    payload.append({"type": text_type, "text": data})
                case "image":
                    payload.append(self._format_image_part(part))
                case "audio":
                    raise ValueError("OpenAI Responses does not support audio input")
                case "file":
                    raise ValueError("OpenAI Responses does not support file input")
                case _:
                    raise ValueError(f"Unsupported message part: {part.type}")
        return payload

    def _format_image_part(self, part: LLMMessagePart) -> dict[str, Any]:
        image_url = part.get("url")
        if image_url is None and isinstance(part["data"], bytes):
            mime = part.get("mime_type", "image/png")
            b64 = base64.b64encode(part["data"]).decode("ascii")
            image_url = f"data:{mime};base64,{b64}"
        if image_url is None:
            raise ValueError("Image parts must include `url` or `data`")
        return {
            "type": "input_image",
            "image_url": image_url,
            "detail": "auto",
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

    def _format_tool(self, tool: IToolSpec) -> FunctionToolParam:
        schema = tool.generate_schema()
        return cast(FunctionToolParam, schema)

    def _provider_meta_entry(
        self,
        *,
        model: str,
        latency_ms: float | None = None,
        response_id: str | None = None,
    ) -> LLMProviderMeta:
        return LLMProviderMeta(
            provider_name="openai",
            model=model,
            latency_ms=latency_ms,
            response_id=response_id,
        )

    def _supports_reasoning_summary(self, model_name: str) -> bool:
        name = model_name.lower()
        return name.startswith(("o3", "o4", "gpt-5"))

    def _extract_reasoning_items(
        self, response: Response
    ) -> list[ResponseReasoningItem]:
        items: list[ResponseReasoningItem] = []
        for output_item in response.output:
            if isinstance(output_item, ResponseReasoningItem):
                items.append(output_item)
        return items

    def _build_reasoning_segments(
        self, reasoning_items: list[ResponseReasoningItem]
    ) -> list[LLMSegment]:
        segments: list[LLMSegment] = []
        for item in reasoning_items:
            for idx, summary in enumerate(item.summary):
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
            if item.content is not None:
                for idx, content in enumerate(item.content):
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

    def _build_segments_from_response(
        self,
        *,
        response: Response,
        tool_calls: list[LLMToolCall],
        reasoning_items: list[ResponseReasoningItem] | None = None,
    ) -> list[LLMSegment]:
        segments: list[LLMSegment] = []
        reasoning_items = reasoning_items or []
        if response.output_text:
            segments.append(LLMSegment(type="text", content=response.output_text))
        segments.extend(self._build_image_segments(response))
        segments.extend(self._build_reasoning_segments(reasoning_items))
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
        return segments

    def _build_image_segments(self, response: Response) -> list[LLMSegment]:
        segments: list[LLMSegment] = []
        for idx, item in enumerate(response.output or []):
            if isinstance(item, ImageGenerationCall):
                media = self._image_call_to_media(item)
                segments.append(
                    LLMSegment(
                        type="image",
                        content="",
                        media=media,
                        meta=LLMImageSegmentMeta(
                            item_id=item.id,
                            status=item.status,
                            output_index=idx,
                        ),
                    )
                )
        return segments

    def _image_call_to_media(self, item: ImageGenerationCall) -> LLMGeneratedMedia:
        data: bytes | None = None
        if item.result:
            data = base64.b64decode(item.result)
        return LLMGeneratedMedia(type="image", mime_type="image/png", data=data)

    def _extract_tool_calls(self, response: Response) -> list[LLMToolCall]:
        calls: list[LLMToolCall] = []
        for item in response.output:
            if isinstance(item, ResponseFunctionToolCall):
                call_id = item.call_id or item.id
                if call_id is None:
                    raise AceAIRuntimeError(
                        "OpenAI function call response did not include a call identifier"
                    )
                calls.append(
                    LLMToolCall(
                        name=item.name,
                        arguments=item.arguments,
                        call_id=call_id,
                    )
                )
        return calls

    def _to_llm_response(
        self, response: Response, *, latency_ms: float | None = None
    ) -> LLMResponse:
        usage = response.usage
        usage_block: Unset[LLMUsage] = UNSET
        if usage:
            usage_block = LLMUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        tool_calls = self._extract_tool_calls(response)
        reasoning_items = self._extract_reasoning_items(response)
        segments = self._build_segments_from_response(
            response=response,
            tool_calls=tool_calls,
            reasoning_items=reasoning_items,
        )
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
            text=response.output_text or "",
            tool_calls=tool_calls,
            usage=usage_block,
            segments=segments,
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
            ) if (delta and item_id):
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

        provider_meta = [self._provider_meta_entry(model=model_name)]
        return LLMStreamEvent(
            event_type=event_type,
            text_delta=text_delta,
            tool_call_delta=tool_call_delta,
            error=error_value,
            segments=segments,
            provider_meta=provider_meta,
        )

    def _apply_default_meta(self, request: LLMRequest) -> LLMRequest:
        """if not request no metadata, set to default. if partial, fill in missing."""
        if "metadata" not in request or not request["metadata"]:
            request["metadata"] = self._default_metadata
        else:
            request_meta = request["metadata"]
            for key, value in self._default_metadata.items():
                if key not in request_meta:
                    request_meta[key] = value
        return request

    async def complete(
        self, request: LLMRequest, *, trace_ctx: Context | None = None
    ) -> LLMResponse:
        """Complete using OpenAI Responses API."""
        request = self._apply_default_meta(request)
        params = self._build_base_response_kwargs(request)
        start = time.perf_counter()
        tool_names = self._tool_names(request)
        attributes = {
            "llm.provider": self.__class__.__name__,
            "llm.model": params["model"],
            "llm.stream": False,
            "llm.tool_count": len(tool_names),
            "llm.tool_names": tool_names,
        }
        span = self._tracer.start_span(
            "openai.responses.create",
            kind=SpanKind.CLIENT,
            context=trace_ctx,
            attributes=attributes,
        )
        try:
            response: Response = await self._client.responses.create(**params)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return self._to_llm_response(response, latency_ms=latency_ms)
        finally:
            span.end()

    async def stream(
        self, request: LLMRequest, *, trace_ctx: Context | None = None
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream tokens and tool calls using OpenAI Responses streaming API."""
        request = self._apply_default_meta(request)
        kwargs = self._build_base_response_kwargs(request)
        start = time.perf_counter()
        tool_names = self._tool_names(request)
        attributes = {
            "llm.provider": self.__class__.__name__,
            "llm.model": kwargs["model"],
            "llm.stream": True,
            "llm.tool_count": len(tool_names),
            "llm.tool_names": tool_names,
        }
        span = self._tracer.start_span(
            "openai.responses.stream",
            kind=SpanKind.CLIENT,
            context=trace_ctx,
            attributes=attributes,
        )
        stream_manager = self._client.responses.stream(**kwargs)
        try:
            async with stream_manager as stream:
                async for event in stream:
                    mapped = self._map_stream_event(event, model_name=kwargs["model"])
                    if mapped is None:
                        continue
                    yield mapped

                parsed = await stream.get_final_response()
                latency_ms = (time.perf_counter() - start) * 1000.0
                final_llm_response = self._to_llm_response(
                    parsed, latency_ms=latency_ms
                )
                yield LLMStreamEvent(
                    event_type="response.completed",
                    response=final_llm_response,
                    segments=final_llm_response.segments,
                    provider_meta=final_llm_response.provider_meta,
                )
        finally:
            span.end()

    def _media_from_base64(self, payload: str) -> LLMGeneratedMedia:
        data = base64.b64decode(payload)
        return LLMGeneratedMedia(type="image", mime_type="image/png", data=data)
