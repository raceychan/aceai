from typing import Any, AsyncIterator, BinaryIO, cast
from warnings import warn

import openai
from openai.types.responses import FunctionToolParam
from openai.types.responses.response import Response
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from aceai.interface import MISSING, UNSET, Unset, is_present, is_set

from .interface import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    LLMToolCall,
    LLMToolCallDelta,
    LLMUsage,
    ResponseFormat,
    ToolSpec,
)


class OpenAI(LLMProviderBase):
    """OpenAI provider for LLM completions."""

    def __init__(self, api_key: str, default_model: str, default_stream_model: str):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._default_model = default_model
        self._default_stream_model = default_stream_model
        self._client = openai.AsyncOpenAI(api_key=self._api_key)

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
        input_messages = [m.asdict() for m in request["messages"]]
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

    def _build_completion_kwargs(
        self,
        request: LLMRequest,
    ) -> ResponseCreateParamsNonStreaming:
        kwargs = self._build_base_response_kwargs(
            request,
            default_model=self._default_model,
        )
        return cast(ResponseCreateParamsNonStreaming, kwargs)

    def _build_stream_completion_kwargs(
        self,
        request: LLMRequest,
    ) -> dict[str, Any]:
        kwargs = self._build_base_response_kwargs(
            request,
            default_model=self._default_stream_model,
        )
        return kwargs

    def _build_text_config(
        self,
        response_format: ResponseFormat,
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

    def _extract_tool_calls(self, response: Response) -> list[LLMToolCall]:
        calls: list[LLMToolCall] = []
        for item in response.output:
            if isinstance(item, ResponseFunctionToolCall):
                calls.append(
                    LLMToolCall(
                        type="function",
                        name=item.name,
                        arguments=item.arguments,
                        call_id=item.call_id,
                        id=item.id,
                    )
                )
        return calls

    def _to_llm_response(self, response: Response) -> LLMResponse:
        usage = response.usage
        usage_block: Unset[LLMUsage] = UNSET
        if usage:
            usage_block = LLMUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        return LLMResponse(
            id=response.id,
            model=str(response.model),
            text=response.output_text or "",
            tool_calls=self._extract_tool_calls(response),
            usage=usage_block,
        )

    def _map_stream_event(
        self,
        event: ResponseStreamEvent,
    ) -> LLMStreamChunk | None:
        match event:
            case ResponseTextDeltaEvent(delta=delta) if delta:
                return LLMStreamChunk(text_delta=delta)
            case ResponseFunctionCallArgumentsDeltaEvent(
                delta=delta, item_id=item_id
            ) if (delta and item_id):
                return LLMStreamChunk(
                    tool_call_delta=LLMToolCallDelta(
                        id=item_id,
                        arguments_delta=delta,
                    )
                )
            case ResponseErrorEvent(message=message):
                return LLMStreamChunk(error=message or "LLM stream error")
            case _:
                match event.type:
                    case "response.output_text.delta":
                        return LLMStreamChunk(text_delta=event.delta)
                    case "response.function_call_arguments.delta":
                        return LLMStreamChunk(
                            tool_call_delta=LLMToolCallDelta(
                                id=event.item_id,
                                arguments_delta=event.delta,
                            )
                        )
                    case _:
                        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete using OpenAI Responses API."""
        params = self._build_completion_kwargs(request)
        response: Response = await self._client.responses.create(**params)
        return self._to_llm_response(response)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream tokens and tool calls using OpenAI Responses streaming API."""
        kwargs = self._build_stream_completion_kwargs(request)
        stream_manager = self._client.responses.stream(**kwargs)
        final_response: Response | None = None
        async with stream_manager as stream:
            async for event in stream:
                chunk = self._map_stream_event(event)
                if chunk is None:
                    continue
                yield chunk
            parsed = await stream.get_final_response()
            final_response = cast(Response, parsed)
        yield LLMStreamChunk(response=self._to_llm_response(final_response))
