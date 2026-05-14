"""DeepSeek provider using the OpenAI-compatible Chat Completions API."""

import time
from importlib.util import find_spec
from typing import Any, AsyncGenerator

if find_spec("openai") is None:
    raise RuntimeError(
        "DeepSeek provider requires the deepseek extra. "
        "Install with `pip install 'aceai[deepseek]'`."
    )

from openai import AsyncOpenAI

from aceai.llm.errors import AceAIConfigurationError
from aceai.llm.interface import UNSET, Unset
from aceai.llm.models import (
    LLMHostedToolSpec,
    LLMInput,
    LLMMessage,
    LLMProviderModality,
    LLMResponse,
    LLMSegment,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallDelta,
    LLMToolCallMessage,
    LLMToolCallSegmentMeta,
    LLMToolSpec,
    LLMToolUseMessage,
    LLMUsage,
    LLMReasoningSegmentMeta,
)
from aceai.llm.openai import OpenAI, OpenAIMeta, OpenAIPayload


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CLIENT_TIMEOUT_SECONDS = 300.0


class DeepSeek(OpenAI):
    """DeepSeek provider backed by OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        default_meta: OpenAIMeta,
        base_url: str = DEEPSEEK_BASE_URL,
    ):
        super().__init__(
            client=AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=DEEPSEEK_CLIENT_TIMEOUT_SECONDS,
            ),
            default_meta=default_meta,
            provider_name="deepseek",
        )

    @property
    def modality(self) -> LLMProviderModality:
        return LLMProviderModality()

    def _coerce_text_content(
        self,
        content: list[dict[str, Any]],
        *,
        context: str,
    ) -> str:
        text_parts: list[str] = []
        for part in content:
            if part["type"] != "text":
                raise ValueError(f"{context} only supports text parts")
            data = part["data"]
            if type(data) is not str:
                raise TypeError(f"{context} text parts must be str")
            text_parts.append(data)
        return "".join(text_parts)

    def _format_messages_for_chat(
        self, messages: list[LLMMessage]
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, LLMToolUseMessage):
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.call_id,
                        "name": message.name,
                        "content": self._coerce_text_content(
                            message.content,
                            context="DeepSeek tool output",
                        ),
                    }
                )
                continue
            if isinstance(message, LLMToolCallMessage):
                assistant_message = {
                    "role": "assistant",
                    "content": self._coerce_text_content(
                        message.content,
                        context="DeepSeek assistant message",
                    ),
                    "tool_calls": [
                        self._format_tool_call_message(call)
                        for call in message.tool_calls
                    ],
                }
                if message.reasoning_content is not None:
                    assistant_message["reasoning_content"] = message.reasoning_content
                formatted.append(assistant_message)
                continue
            formatted.append(
                {
                    "role": message.role,
                    "content": self._coerce_text_content(
                        message.content,
                        context="DeepSeek message",
                    ),
                }
            )
        return formatted

    def _format_tool_call_message(self, call: LLMToolCall) -> dict[str, Any]:
        return {
            "id": call.call_id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }

    def _format_tool_for_chat(self, tool: LLMToolSpec) -> dict[str, Any]:
        if isinstance(tool, LLMHostedToolSpec):
            raise AceAIConfigurationError("DeepSeek does not support hosted tools")
        schema = tool.generate_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
                "strict": schema["strict"],
            },
        }

    def _build_chat_kwargs(self, payload: OpenAIPayload) -> dict[str, Any]:
        data = payload.asdict()
        metadata = payload.metadata if "metadata" in data else {}
        if "model" not in metadata:
            raise AceAIConfigurationError(
                "DeepSeek request metadata must include a model identifier"
            )
        model_name = metadata["model"]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._format_messages_for_chat(payload.messages),
        }
        if "temperature" in data:
            kwargs["temperature"] = payload.temperature
        if "top_p" in data:
            kwargs["top_p"] = payload.top_p
        if "max_tokens" in data:
            kwargs["max_tokens"] = payload.max_tokens
        if "stop" in data:
            kwargs["stop"] = payload.stop
        if "tools" in data:
            kwargs["tools"] = [
                self._format_tool_for_chat(tool) for tool in payload.tools
            ]
        if "tool_choice" in data:
            kwargs["tool_choice"] = payload.tool_choice
        if "response_format" in data:
            kwargs["response_format"] = {"type": payload.response_format.type}
        if "reasoning" in metadata:
            reasoning = metadata["reasoning"]
            kwargs["reasoning_effort"] = reasoning["effort"]
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return kwargs

    def _usage_from_chat(self, usage: Any) -> Unset[LLMUsage]:
        if usage is None:
            return UNSET
        return self.build_usage(
            input_tokens=usage.prompt_tokens,
            cached_input_tokens=usage.prompt_cache_hit_tokens,
            cache_miss_input_tokens=usage.prompt_cache_miss_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _reasoning_content_from_extra(self, value: Any) -> str | None:
        extra = value.model_extra
        if extra is None:
            return None
        if "reasoning_content" not in extra:
            return None
        reasoning_content = extra["reasoning_content"]
        if reasoning_content is None:
            return None
        if type(reasoning_content) is not str:
            raise TypeError("DeepSeek reasoning_content must be str")
        return reasoning_content

    def _tool_calls_from_chat(self, message: Any) -> list[LLMToolCall]:
        calls: list[LLMToolCall] = []
        if message.tool_calls is None:
            return calls
        for call in message.tool_calls:
            calls.append(
                LLMToolCall(
                    name=call.function.name,
                    arguments=call.function.arguments,
                    call_id=call.id,
                )
            )
        return calls

    def _to_chat_llm_response(
        self,
        response: Any,
        *,
        latency_ms: float | None = None,
    ) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        content = message.content
        if content is None:
            content = ""
        reasoning_content = self._reasoning_content_from_extra(message)
        tool_calls = self._tool_calls_from_chat(message)
        segments = [LLMSegment(type="text", content=content)] if content != "" else []
        if reasoning_content is not None:
            segments.append(
                LLMSegment(
                    type="reasoning",
                    content=reasoning_content,
                )
            )
        for call in tool_calls:
            segments.append(
                LLMSegment(
                    type="tool_call",
                    content=call.arguments,
                    meta=LLMToolCallSegmentMeta(
                        call_id=call.call_id,
                        tool_name=call.name,
                    ),
                )
            )
        return LLMResponse(
            id=response.id,
            model=response.model,
            text=content,
            tool_calls=tool_calls,
            usage=self._usage_from_chat(response.usage),
            segments=segments,
            provider_meta=[
                self._provider_meta_entry(
                    model=response.model,
                    latency_ms=latency_ms,
                    response_id=response.id,
                )
            ],
            status=choice.finish_reason,
            reasoning_content=reasoning_content,
        )

    def request_to_payload(self, request: LLMInput) -> OpenAIPayload:
        payload = OpenAIPayload.from_input(request)
        return self._apply_default_meta(payload)

    async def complete(self, request: LLMInput) -> LLMResponse:
        payload = self.request_to_payload(request)
        kwargs = self._build_chat_kwargs(payload)
        start = time.perf_counter()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._to_chat_llm_response(response, latency_ms=latency_ms)

    async def stream(self, request: LLMInput) -> AsyncGenerator[LLMStreamEvent, None]:
        payload = self.request_to_payload(request)
        kwargs = self._build_chat_kwargs(payload)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        start = time.perf_counter()
        stream = await self._client.chat.completions.create(**kwargs)
        text = ""
        reasoning_content = ""
        model_name = kwargs["model"]
        usage: Unset[LLMUsage] = UNSET
        tool_names_by_index: dict[int, str] = {}
        tool_ids_by_index: dict[int, str] = {}
        tool_arguments_by_index: dict[int, str] = {}

        async for chunk in stream:
            if chunk.usage is not None:
                usage = self._usage_from_chat(chunk.usage)
            if len(chunk.choices) == 0:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            delta_reasoning = self._reasoning_content_from_extra(delta)
            if delta_reasoning is not None:
                reasoning_content += delta_reasoning
                yield LLMStreamEvent(
                    event_type="response.reasoning.delta",
                    segments=[
                        LLMSegment(
                            type="reasoning",
                            content=delta_reasoning,
                            meta=LLMReasoningSegmentMeta(
                                item_id="deepseek-reasoning",
                                kind="content",
                                index=0,
                                status="in_progress",
                                is_delta=True,
                            ),
                        )
                    ],
                    provider_meta=[self._provider_meta_entry(model=model_name)],
                )
                continue
            if delta.content:
                text += delta.content
                yield LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta=delta.content,
                    segments=[LLMSegment(type="text", content=delta.content)],
                    provider_meta=[self._provider_meta_entry(model=model_name)],
                )
            if delta.tool_calls is None:
                continue
            for tool_delta in delta.tool_calls:
                if tool_delta.id is not None:
                    tool_ids_by_index[tool_delta.index] = tool_delta.id
                if tool_delta.function.name is not None:
                    tool_names_by_index[tool_delta.index] = tool_delta.function.name
                if tool_delta.function.arguments is None:
                    continue
                existing = (
                    tool_arguments_by_index[tool_delta.index]
                    if tool_delta.index in tool_arguments_by_index
                    else ""
                )
                tool_arguments_by_index[tool_delta.index] = (
                    existing + tool_delta.function.arguments
                )
                call_id = tool_ids_by_index[tool_delta.index]
                yield LLMStreamEvent(
                    event_type="response.function_call_arguments.delta",
                    tool_call_delta=LLMToolCallDelta(
                        id=call_id,
                        arguments_delta=tool_delta.function.arguments,
                    ),
                    segments=[
                        LLMSegment(
                            type="tool_call",
                            content=tool_delta.function.arguments,
                            meta=LLMToolCallSegmentMeta(
                                call_id=call_id,
                                tool_name=(
                                    tool_names_by_index[tool_delta.index]
                                    if tool_delta.index in tool_names_by_index
                                    else None
                                ),
                                is_delta=True,
                            ),
                        )
                    ],
                    provider_meta=[self._provider_meta_entry(model=model_name)],
                )

        latency_ms = (time.perf_counter() - start) * 1000.0
        tool_calls = [
            LLMToolCall(
                name=tool_names_by_index[index],
                arguments=tool_arguments_by_index[index],
                call_id=tool_ids_by_index[index],
            )
            for index in sorted(tool_arguments_by_index)
        ]
        segments = [LLMSegment(type="text", content=text)] if text != "" else []
        if reasoning_content != "":
            segments.append(
                LLMSegment(
                    type="reasoning",
                    content=reasoning_content,
                )
            )
        for call in tool_calls:
            segments.append(
                LLMSegment(
                    type="tool_call",
                    content=call.arguments,
                    meta=LLMToolCallSegmentMeta(
                        call_id=call.call_id,
                        tool_name=call.name,
                    ),
                )
            )
        final_response = LLMResponse(
            model=model_name,
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            segments=segments,
            provider_meta=[
                self._provider_meta_entry(
                    model=model_name,
                    latency_ms=latency_ms,
                )
            ],
            status="completed",
            reasoning_content=reasoning_content if reasoning_content != "" else None,
        )
        yield LLMStreamEvent(
            event_type="response.completed",
            response=final_response,
            segments=final_response.segments,
            provider_meta=final_response.provider_meta,
        )
