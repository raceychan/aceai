"""
OpenAI adapter and LLM provider abstraction.

This module centralizes interactions with the OpenAI SDK and exposes a small
provider interface so upstream application code doesn't depend on the SDK
directly.

Notes
- We re-export OpenAI exception types so callers can catch them via this module
  without importing the SDK directly.
"""

from typing import Any, Sequence, Unpack, cast
from warnings import warn

from openai import AsyncOpenAI
from openai.types.responses.response import Response as ResponsesResponse
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from openai.types.shared_params.response_format_text import ResponseFormatText

from .interface import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    LLMRequest,
    ToolCall,
    ToolCallFunction,
)

ResponseFormat = (
    ResponseFormatJSONObject | ResponseFormatJSONSchema | ResponseFormatText
)

TEMPERATURELESS_MODELS: set[str] = {"gpt-5"}


def _build_message_content(message: ChatMessage) -> dict[str, str]:
    kind = "output_text" if message.role == "assistant" else "input_text"
    return {
        "type": kind,
        "text": message.content,
    }


def _collect_response_text(response: ResponsesResponse) -> str:
    if not response.output:
        return ""
    parts: list[str] = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for content in item.content:
                if isinstance(content, ResponseOutputText):
                    parts.append(content.text)
                else:
                    parts.append(content.refusal)
    return "".join(parts)


def _collect_tool_calls(response: ResponsesResponse) -> list[ToolCall]:
    calls: list[ToolCall] = []
    if not response.output:
        return calls
    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall):
            calls.append(
                ToolCall(
                    id=item.call_id,
                    function=ToolCallFunction(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                    type="function",
                )
            )
    return calls


class OpenAIProvider(LLMProvider):
    """Concrete LLM provider backed by the OpenAI SDK."""

    def __init__(self, *, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)

    async def get_response(
        self,
        **request: Unpack[LLMRequest],
    ) -> ChatResponse:
        kwargs = self._build_response_kwargs(
            **request,
        )
        resp = await self._client.responses.create(**kwargs)
        return self.convert_resp(resp)

    def convert_resp(self, resp: ResponsesResponse) -> ChatResponse:
        content = _collect_response_text(resp)
        tool_calls = _collect_tool_calls(resp)
        return ChatResponse(content=content, tool_calls=tool_calls)

    def _build_response_kwargs(
        self,
        **request: Unpack[LLMRequest],
    ) -> ResponseCreateParamsNonStreaming:
        input_items = self._build_input_items(request["messages"])

        if (
            request["model"] in TEMPERATURELESS_MODELS
            and (temperature := request.get("temperature")) is not None
        ):
            warn(
                f"Model {request['model']} does not support temperature; ignoring temperature={temperature}"
            )
            request.pop("temperature")

        kwargs: dict[str, Any] = {
            "model": request["model"],
            "input": list(input_items),
        }

        text_config: dict[str, Any] = {}
        if request.get("json_schema") is not None:
            text_config["format"] = {
                "type": "json_schema",
                "json_schema": request.get("json_schema"),
            }
        kwargs["text"] = cast(ResponseTextConfigParam, text_config)

        if max_output_token := kwargs.get("max_output_tokens"):
            kwargs["max_output_tokens"] = max_output_token

        if temperature := kwargs.get("temperature") is not None:
            kwargs["temperature"] = temperature
        if top_p := kwargs.get("top_p") is not None:
            kwargs["top_p"] = top_p

        if tools := request.get("tools"):
            kwargs["tools"] = tools
        if tool_choice := request.get("tool_choice"):
            kwargs["tool_choice"] = tool_choice

        return cast(ResponseCreateParamsNonStreaming, kwargs)

    def _build_input_items(
        self, messages: Sequence[ChatMessage]
    ) -> list[ResponseInputParam]:
        items: list[ResponseInputParam] = []
        for msg in messages:
            if msg.role == "tool":
                if not msg.tool_call_id:
                    raise ValueError("Tool response missing tool_call_id")
                items.append(
                    cast(
                        ResponseInputParam,
                        {
                            "type": "function_call_output",
                            "call_id": msg.tool_call_id,
                            "output": msg.content,
                        },
                    )
                )
                continue

            content_item = _build_message_content(msg)

            message_item: ResponseInputParam = cast(
                ResponseInputParam,
                {
                    "type": "message",
                    "role": msg.role,
                    "content": [content_item],
                },
            )

            items.append(message_item)
            if msg.role == "assistant" and msg.tool_calls:
                for call in msg.tool_calls:
                    items.append(
                        cast(
                            ResponseInputParam,
                            {
                                "type": "function_call",
                                "call_id": call.id,
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        )
                    )
        return items
