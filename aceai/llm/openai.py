import time
from typing import Any, AsyncIterator, BinaryIO, cast
from warnings import warn

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam
from openai.types.responses.response import Response
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from aceai.interface import MISSING, UNSET, Unset, is_present, is_set

from .models import (
    LLMMessage,
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    LLMSegment,
    LLMStreamChunk,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallDelta,
    LLMToolCallMessage,
    LLMToolUseMessage,
    LLMUsage,
    LLMProviderMeta,
    LLMResponseFormat,
    ToolSpec,
)


class OpenAI(LLMProviderBase):
    """OpenAI provider for LLM completions."""

    def __init__(
        self, client: AsyncOpenAI, *, default_model: str, default_stream_model: str
    ):
        self._client = client
        self._default_model = default_model
        self._default_stream_model = default_stream_model

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_stream_model(self) -> str:
        return self._default_stream_model

    async def stt(self, filename: str, file: BinaryIO, *, model: str) -> str:
        """Transcribe audio using OpenAI Whisper (async)."""
        result = await self._client.audio.transcriptions.create(
            model=model,
            file=(filename, file),
        )
        text = result.text
        return text

    def _build_base_response_kwargs(
        self,
        request: LLMRequest,
        *,
        default_model: str,
    ) -> dict[str, Any]:
        """Translate LLMRequest into OpenAI Responses kwargs."""
        input_messages = self._format_messages_for_responses(request["messages"])
        kwargs: dict[str, Any] = {"input": input_messages}
        metadata = request.get("metadata", {})
        model_name = metadata.get("model", default_model)

        kwargs["model"] = model_name

        if is_present(max_tokens := request.get("max_tokens", MISSING)):
            kwargs["max_output_tokens"] = max_tokens

        if is_present(temperature := request.get("temperature", MISSING)):
            if model_name.startswith("gpt-5"):
                pass
            else:
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
                        "output": message.content,
                    }
                )
            elif isinstance(message, LLMToolCallMessage):
                if message.content:
                    formatted.append({"role": message.role, "content": message.content})
                for tc in message.tool_calls or []:
                    formatted.append(tc.asdict())
            else:
                formatted.append({"role": message.role, "content": message.content})

        return formatted

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
                raise RuntimeError(
                    f"Unsupported OpenAI response format type: {response_format.type}"
                )

    def _format_tool(self, tool: ToolSpec) -> FunctionToolParam:
        description = tool["description"]
        parameters = tool["parameters"]
        return {
            "type": "function",
            "name": tool["name"],
            "description": description,
            "parameters": parameters,
            "strict": False,
        }

    def _provider_meta_entry(
        self,
        *,
        model: str,
        latency_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMProviderMeta:
        return LLMProviderMeta(
            provider_name="openai",
            model=model,
            latency_ms=latency_ms,
            extra=extra or {},
        )

    def _build_segments_from_response(
        self,
        *,
        response: Response,
        tool_calls: list[LLMToolCall],
    ) -> list[LLMSegment]:
        segments: list[LLMSegment] = []
        if response.output_text:
            segments.append(LLMSegment(type="text", content=response.output_text))
        for call in tool_calls:
            segments.append(
                LLMSegment(
                    type="tool_call",
                    content=call.arguments or "",
                    metadata={
                        "call_id": call.call_id,
                        "tool_name": call.name,
                    },
                )
            )
        return segments

    def _safe_model_dump(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        dump = getattr(payload, "model_dump", None)
        if callable(dump):
            try:
                data = dump()
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def _extract_tool_calls(self, response: Response) -> list[LLMToolCall]:
        calls: list[LLMToolCall] = []
        for item in response.output:
            if isinstance(item, ResponseFunctionToolCall):
                call_id = item.call_id or item.id
                if call_id is None:
                    raise RuntimeError(
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
        segments = self._build_segments_from_response(
            response=response, tool_calls=tool_calls
        )
        response_model = str(response.model)
        extras: dict[str, Any] = {}
        if response.status is not None:
            extras["status"] = response.status
        raw_event = self._safe_model_dump(response)
        provider_meta = [
            self._provider_meta_entry(
                model=response_model,
                latency_ms=latency_ms,
                extra={"response_id": response.id} if response.id else None,
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
            raw_events=[raw_event] if raw_event else [],
            extras=extras,
        )

    def _map_stream_event(
        self,
        event: ResponseStreamEvent,
        *,
        model_name: str,
    ) -> LLMStreamEvent | None:
        chunk: LLMStreamChunk | None = None
        segments: list[LLMSegment] = []
        event_type: str | None = None

        match event:
            case ResponseTextDeltaEvent(delta=delta) if delta:
                chunk = LLMStreamChunk(text_delta=delta)
                segments = [LLMSegment(type="text", content=delta)]
                event_type = "response.output_text.delta"
            case ResponseFunctionCallArgumentsDeltaEvent(
                delta=delta, item_id=item_id
            ) if (delta and item_id):
                chunk = LLMStreamChunk(
                    tool_call_delta=LLMToolCallDelta(
                        id=item_id,
                        arguments_delta=delta,
                    )
                )
                segments = [
                    LLMSegment(
                        type="tool_call",
                        content=delta,
                        metadata={"call_id": item_id, "is_delta": True},
                    )
                ]
                event_type = "response.function_call_arguments.delta"
            case ResponseErrorEvent(message=message):
                error_msg = message or "LLM stream error"
                chunk = LLMStreamChunk(error=error_msg)
                segments = [LLMSegment(type="error", content=error_msg)]
                event_type = "response.error"
            case _:
                match event.type:
                    case "response.output_text.delta" if event.delta:
                        chunk = LLMStreamChunk(text_delta=event.delta)
                        segments = [LLMSegment(type="text", content=event.delta)]
                        event_type = "response.output_text.delta"
                    case "response.function_call_arguments.delta" if (
                        event.delta and event.item_id
                    ):
                        chunk = LLMStreamChunk(
                            tool_call_delta=LLMToolCallDelta(
                                id=event.item_id,
                                arguments_delta=event.delta,
                            )
                        )
                        segments = [
                            LLMSegment(
                                type="tool_call",
                                content=event.delta,
                                metadata={
                                    "call_id": event.item_id,
                                    "is_delta": True,
                                },
                            )
                        ]
                        event_type = "response.function_call_arguments.delta"
                    case "response.error":
                        error_msg = (
                            getattr(event, "message", None) or "LLM stream error"
                        )
                        chunk = LLMStreamChunk(error=error_msg)
                        segments = [LLMSegment(type="error", content=error_msg)]
                        event_type = "response.error"
                    case _:
                        return None

        raw_event = self._safe_model_dump(event)
        provider_meta = [self._provider_meta_entry(model=model_name)]
        return LLMStreamEvent(
            event_type=event_type,
            chunk=chunk,
            segments=segments,
            provider_meta=provider_meta,
            raw_event=raw_event or None,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete using OpenAI Responses API."""
        params = self._build_base_response_kwargs(
            request, default_model=self._default_model
        )
        start = time.perf_counter()
        response: Response = await self._client.responses.create(**params)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._to_llm_response(response, latency_ms=latency_ms)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """Stream tokens and tool calls using OpenAI Responses streaming API."""
        kwargs = self._build_base_response_kwargs(
            request, default_model=self._default_stream_model
        )
        model_name = kwargs.get("model", self._default_stream_model)
        start = time.perf_counter()
        stream_manager = self._client.responses.stream(**kwargs)
        final_response: Response | None = None
        async with stream_manager as stream:
            async for event in stream:
                mapped = self._map_stream_event(event, model_name=model_name)
                if mapped is None:
                    continue
                yield mapped
            parsed = await stream.get_final_response()
            final_response = cast(Response, parsed)
        assert final_response is not None
        latency_ms = (time.perf_counter() - start) * 1000.0
        final_llm_response = self._to_llm_response(
            final_response, latency_ms=latency_ms
        )
        yield LLMStreamEvent(
            event_type="response.completed",
            chunk=LLMStreamChunk(response=final_llm_response),
            segments=final_llm_response.segments,
            provider_meta=final_llm_response.provider_meta,
            raw_event=self._safe_model_dump(final_response) or None,
            extras=final_llm_response.extras,
        )
