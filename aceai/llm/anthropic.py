"""Anthropic provider using the Messages API."""

import base64
import json
import time
from importlib.util import find_spec
from typing import Any, AsyncGenerator, BinaryIO, Literal, TypedDict

if find_spec("httpx") is None:
    raise RuntimeError(
        "Anthropic provider requires the anthropic extra. "
        "Install with `pip install 'aceai[anthropic]'`."
    )

import httpx
from msgspec import Struct, convert, to_builtins

from aceai.llm.errors import (
    AceAIConfigurationError,
    AceAIValidationError,
    LLMProviderError,
)
from aceai.llm.interface import UNSET, StrDict, Unset, is_set
from aceai.llm.models import (
    LLMHostedToolSpec,
    LLMInput,
    LLMMessage,
    LLMMessagePart,
    LLMProviderBase,
    LLMProviderMeta,
    LLMProviderModality,
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
from aceai.llm.tool_spec import IToolSpec


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_CLIENT_TIMEOUT_SECONDS = 300.0
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_PROVIDER_NAME = "anthropic"
ANTHROPIC_OAUTH_PROVIDER_NAME = "anthropic-oauth"
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_CACHE_CONTROL = {"type": "ephemeral"}


class AnthropicMeta(TypedDict, total=False):
    model: str


class AnthropicPayload(Struct, kw_only=True):
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
    metadata: Unset[AnthropicMeta] = UNSET

    @classmethod
    def from_input(cls, llm_input: LLMInput) -> "AnthropicPayload":
        messages = llm_input["messages"]
        if not isinstance(messages, list):
            raise TypeError("AnthropicPayload.messages must be list[LLMMessage]")
        payload: dict[str, Any] = {"messages": cls._validate_messages(messages)}
        if "temperature" in llm_input:
            temperature = llm_input["temperature"]
            if type(temperature) is not float:
                raise TypeError("AnthropicPayload.temperature must be float")
            payload["temperature"] = temperature
        if "top_p" in llm_input:
            top_p = llm_input["top_p"]
            if type(top_p) is not float:
                raise TypeError("AnthropicPayload.top_p must be float")
            payload["top_p"] = top_p
        if "top_k" in llm_input:
            top_k = llm_input["top_k"]
            if type(top_k) is not int:
                raise TypeError("AnthropicPayload.top_k must be int")
            payload["top_k"] = top_k
        if "max_tokens" in llm_input:
            max_tokens = llm_input["max_tokens"]
            if type(max_tokens) is not int:
                raise TypeError("AnthropicPayload.max_tokens must be int")
            payload["max_tokens"] = max_tokens
        if "stop" in llm_input:
            stop = llm_input["stop"]
            if not isinstance(stop, list) or not all(type(item) is str for item in stop):
                raise TypeError("AnthropicPayload.stop must be list[str]")
            payload["stop"] = stop
        if "tools" in llm_input:
            tools = llm_input["tools"]
            if not isinstance(tools, list):
                raise TypeError("AnthropicPayload.tools must be list[LLMToolSpec]")
            for tool in tools:
                if not isinstance(tool, IToolSpec) and not isinstance(
                    tool, LLMHostedToolSpec
                ):
                    raise TypeError(
                        "AnthropicPayload.tools must contain LLMToolSpec instances"
                    )
            payload["tools"] = tools
        if "tool_choice" in llm_input:
            tool_choice = llm_input["tool_choice"]
            if type(tool_choice) is not str:
                raise TypeError("AnthropicPayload.tool_choice must be str")
            payload["tool_choice"] = tool_choice
        if "response_format" in llm_input:
            response_format = llm_input["response_format"]
            if not isinstance(response_format, LLMResponseFormat):
                raise TypeError(
                    "AnthropicPayload.response_format must be LLMResponseFormat"
                )
            payload["response_format"] = response_format
        if "stream" in llm_input:
            stream = llm_input["stream"]
            if type(stream) is not bool:
                raise TypeError("AnthropicPayload.stream must be bool")
            payload["stream"] = stream
        if "metadata" in llm_input:
            payload["metadata"] = convert(llm_input["metadata"], type=AnthropicMeta)
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
            raise TypeError("AnthropicPayload.messages must be LLMMessage instances")
        return validated

    def asdict(self) -> StrDict:
        def _enc_hook(obj: object) -> object:
            if isinstance(obj, IToolSpec):
                return obj.generate_schema()
            if isinstance(obj, LLMHostedToolSpec):
                return obj.asdict()
            raise TypeError(
                f"Encoding objects of type {type(obj).__name__} is unsupported"
            )

        return to_builtins(self, enc_hook=_enc_hook)  # type: ignore[return-value]


class Anthropic(LLMProviderBase):
    """Anthropic Messages provider for API-key or OAuth bearer-token auth."""

    def __init__(
        self,
        *,
        api_key: str,
        default_meta: AnthropicMeta,
        provider_name: str = ANTHROPIC_PROVIDER_NAME,
        api_url: str = ANTHROPIC_API_URL,
        auth_mode: Literal["api_key", "oauth"] = "api_key",
        client: httpx.AsyncClient | None = None,
    ):
        if api_key == "":
            raise AceAIConfigurationError("Anthropic credential is required")
        self._credential = api_key
        self._default_metadata = convert(default_meta, type=AnthropicMeta)
        self._provider_name = provider_name
        self._api_url = api_url
        self._auth_mode = auth_mode
        self._client = client or httpx.AsyncClient(timeout=ANTHROPIC_CLIENT_TIMEOUT_SECONDS)

    @property
    def modality(self) -> LLMProviderModality:
        return LLMProviderModality(image_in=True)

    async def stt(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str,
        prompt: str | None = None,
    ) -> str:
        raise NotImplementedError("Anthropic provider does not support speech-to-text")

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

    def _headers(self) -> dict[str, str]:
        headers = {
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        if self._auth_mode == "oauth":
            headers["authorization"] = f"Bearer {self._credential}"
            return headers
        headers["x-api-key"] = self._credential
        return headers

    def _apply_default_meta(self, payload: AnthropicPayload) -> AnthropicPayload:
        if not is_set(payload.metadata) or not payload.metadata:
            payload.metadata = self._default_metadata
        else:
            request_meta = payload.metadata
            for key, value in self._default_metadata.items():
                if key not in request_meta:
                    request_meta[key] = value
        return payload

    def request_to_payload(self, request: LLMInput) -> AnthropicPayload:
        payload = AnthropicPayload.from_input(request)
        return self._apply_default_meta(payload)

    def _build_messages_request(self, payload: AnthropicPayload) -> dict[str, Any]:
        data = payload.asdict()
        metadata = payload.metadata if "metadata" in data else {}
        model_name = metadata.get("model")
        if not model_name:
            raise AceAIConfigurationError(
                "Anthropic request metadata must include a model identifier"
            )
        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": (
                payload.max_tokens
                if "max_tokens" in data
                else ANTHROPIC_DEFAULT_MAX_TOKENS
            ),
        }
        system, messages = self._format_messages(payload.messages)
        if system:
            kwargs["system"] = system
        kwargs["messages"] = messages
        if "temperature" in data:
            kwargs["temperature"] = payload.temperature
        if "top_p" in data:
            kwargs["top_p"] = payload.top_p
        if "top_k" in data:
            kwargs["top_k"] = payload.top_k
        if "stop" in data:
            kwargs["stop_sequences"] = payload.stop
        if "response_format" in data and payload.response_format.type != "text":
            raise AceAIValidationError(
                "Anthropic provider only supports text response_format"
            )
        if is_set(payload.tools):
            kwargs["tools"] = [self._format_tool(tool) for tool in payload.tools]
        if "tool_choice" in data:
            kwargs["tool_choice"] = self._format_tool_choice(payload.tool_choice)
        self._apply_prompt_cache(kwargs)
        return kwargs

    def _apply_prompt_cache(self, kwargs: dict[str, Any]) -> None:
        if self._mark_last_cacheable_system_block(kwargs):
            return
        if self._mark_last_cacheable_tool(kwargs):
            return
        self._mark_last_cacheable_message_block(kwargs)

    def _mark_last_cacheable_system_block(self, kwargs: dict[str, Any]) -> bool:
        if "system" not in kwargs:
            return False
        system = kwargs["system"]
        if not isinstance(system, list):
            raise TypeError("Anthropic system payload must be a list")
        for block in reversed(system):
            if self._mark_cacheable_block(block):
                return True
        return False

    def _mark_last_cacheable_tool(self, kwargs: dict[str, Any]) -> bool:
        if "tools" not in kwargs:
            return False
        tools = kwargs["tools"]
        if not isinstance(tools, list):
            raise TypeError("Anthropic tools payload must be a list")
        if len(tools) == 0:
            return False
        tool = tools[-1]
        if not isinstance(tool, dict):
            raise TypeError("Anthropic tool payload must be a mapping")
        tool["cache_control"] = ANTHROPIC_CACHE_CONTROL
        return True

    def _mark_last_cacheable_message_block(self, kwargs: dict[str, Any]) -> bool:
        messages = kwargs["messages"]
        if not isinstance(messages, list):
            raise TypeError("Anthropic messages payload must be a list")
        for message in reversed(messages[:-1]):
            content = message["content"]
            if not isinstance(content, list):
                raise TypeError("Anthropic message content must be a list")
            for block in reversed(content):
                if self._mark_cacheable_block(block):
                    return True
        return False

    def _mark_cacheable_block(self, block: dict[str, Any]) -> bool:
        block_type = block["type"]
        if block_type == "text" and block["text"] == "":
            return False
        block["cache_control"] = ANTHROPIC_CACHE_CONTROL
        return True

    def _format_messages(
        self,
        messages: list[LLMMessage],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system: list[dict[str, Any]] = []
        formatted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system.extend(self._format_content_parts(message.content))
                continue
            if isinstance(message, LLMToolUseMessage):
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.call_id,
                                "content": self._coerce_text_content(
                                    message.content,
                                    context="Anthropic tool output",
                                ),
                            }
                        ],
                    }
                )
                continue
            if isinstance(message, LLMToolCallMessage):
                content = self._format_content_parts(message.content)
                for call in message.tool_calls:
                    content.append(self._format_tool_call_message(call))
                formatted.append({"role": "assistant", "content": content})
                continue
            if message.role == "tool":
                raise AceAIValidationError(
                    "Anthropic tool messages must be LLMToolUseMessage"
                )
            formatted.append(
                {
                    "role": message.role,
                    "content": self._format_content_parts(message.content),
                }
            )
        return system, formatted

    def _coerce_text_content(
        self,
        content: list[LLMMessagePart],
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

    def _format_content_parts(self, content: list[LLMMessagePart]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for part in content:
            match part["type"]:
                case "text":
                    data = part["data"]
                    if type(data) is not str:
                        raise TypeError("Anthropic text parts must be str")
                    if data == "":
                        continue
                    payload.append({"type": "text", "text": data})
                case "image":
                    payload.append(self._format_image_part(part))
                case "audio":
                    raise ValueError("Anthropic provider does not support audio input")
                case "file":
                    raise ValueError("Anthropic provider does not support file input")
                case _:
                    raise ValueError(f"Unsupported message part: {part['type']}")
        return payload

    def _format_image_part(self, part: LLMMessagePart) -> dict[str, Any]:
        if "binary" in part:
            mime = part["mime_type"] if "mime_type" in part else "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(part["binary"]).decode("ascii"),
                },
            }
        if "url" in part:
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": part["url"],
                },
            }
        raise ValueError("Image parts must include `url` or `binary`")

    def _format_tool_call_message(self, call: LLMToolCall) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": json.loads(call.arguments),
        }

    def _format_tool(self, tool: LLMToolSpec) -> dict[str, Any]:
        if isinstance(tool, LLMHostedToolSpec):
            raise AceAIConfigurationError("Anthropic does not support hosted tools")
        schema = tool.generate_schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }

    def _format_tool_choice(self, tool_choice: str) -> dict[str, str]:
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice}

    def _usage_from_payload(self, usage: dict[str, Any] | None) -> Unset[LLMUsage]:
        if usage is None:
            return UNSET
        input_tokens = usage["input_tokens"] if "input_tokens" in usage else None
        output_tokens = usage["output_tokens"] if "output_tokens" in usage else None
        cached_input_tokens = (
            usage["cache_read_input_tokens"]
            if "cache_read_input_tokens" in usage
            else None
        )
        cache_creation_input_tokens = (
            usage["cache_creation_input_tokens"]
            if "cache_creation_input_tokens" in usage
            else None
        )
        cache_miss_input_tokens = None
        if input_tokens is not None and cached_input_tokens is not None:
            cache_miss_input_tokens = input_tokens - cached_input_tokens
            if cache_creation_input_tokens is not None:
                cache_miss_input_tokens += cache_creation_input_tokens
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return self.build_usage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_miss_input_tokens=cache_miss_input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _response_from_payload(
        self,
        payload: dict[str, Any],
        *,
        latency_ms: float | None = None,
    ) -> LLMResponse:
        text = ""
        segments: list[LLMSegment] = []
        tool_calls: list[LLMToolCall] = []
        for block in payload["content"]:
            block_type = block["type"]
            if block_type == "text":
                text += block["text"]
                segments.append(LLMSegment(type="text", content=block["text"]))
            elif block_type == "tool_use":
                arguments = json.dumps(block["input"], separators=(",", ":"))
                call = LLMToolCall(
                    name=block["name"],
                    arguments=arguments,
                    call_id=block["id"],
                )
                tool_calls.append(call)
                segments.append(
                    LLMSegment(
                        type="tool_call",
                        content=arguments,
                        meta=LLMToolCallSegmentMeta(
                            call_id=call.call_id,
                            tool_name=call.name,
                        ),
                    )
                )
        model = payload["model"]
        return LLMResponse(
            id=payload["id"],
            model=model,
            text=text,
            tool_calls=tool_calls,
            usage=self._usage_from_payload(
                payload["usage"] if "usage" in payload else None
            ),
            segments=segments,
            provider_meta=[
                self._provider_meta_entry(
                    model=model,
                    latency_ms=latency_ms,
                    response_id=payload["id"],
                )
            ],
            status=payload["stop_reason"] if "stop_reason" in payload else None,
        )

    async def complete(self, request: LLMInput) -> LLMResponse:
        payload = self.request_to_payload(request)
        kwargs = self._build_messages_request(payload)
        start = time.perf_counter()
        try:
            response = await self._client.post(
                self._api_url,
                headers=self._headers(),
                json=kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            raise _provider_error_from_httpx(err) from err
        except httpx.TransportError as err:
            raise LLMProviderError(f"{type(err).__name__}: {err}") from err
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._response_from_payload(response.json(), latency_ms=latency_ms)

    async def stream(self, request: LLMInput) -> AsyncGenerator[LLMStreamEvent, None]:
        payload = self.request_to_payload(request)
        kwargs = self._build_messages_request(payload)
        kwargs["stream"] = True
        start = time.perf_counter()
        text = ""
        model = kwargs["model"]
        response_id: str | None = None
        stop_reason: str | None = None
        usage: dict[str, Any] | None = None
        tool_block_ids: dict[int, str] = {}
        tool_names: dict[int, str] = {}
        tool_arguments: dict[int, str] = {}
        try:
            async with self._client.stream(
                "POST",
                self._api_url,
                headers=self._headers(),
                json=kwargs,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    event_type = event["type"]
                    if event_type == "message_start":
                        message = event["message"]
                        response_id = message["id"]
                        model = message["model"]
                        usage = message["usage"] if "usage" in message else usage
                    elif event_type == "content_block_start":
                        index = event["index"]
                        block = event["content_block"]
                        if block["type"] == "tool_use":
                            tool_block_ids[index] = block["id"]
                            tool_names[index] = block["name"]
                            tool_arguments[index] = ""
                    elif event_type == "content_block_delta":
                        index = event["index"]
                        delta = event["delta"]
                        delta_type = delta["type"]
                        if delta_type == "text_delta":
                            text += delta["text"]
                            yield LLMStreamEvent(
                                event_type="response.output_text.delta",
                                text_delta=delta["text"],
                                segments=[LLMSegment(type="text", content=delta["text"])],
                                provider_meta=[self._provider_meta_entry(model=model)],
                            )
                        elif delta_type == "input_json_delta":
                            partial_json = delta["partial_json"]
                            tool_arguments[index] += partial_json
                            call_id = tool_block_ids[index]
                            yield LLMStreamEvent(
                                event_type="response.function_call_arguments.delta",
                                tool_call_delta=LLMToolCallDelta(
                                    id=call_id,
                                    arguments_delta=partial_json,
                                ),
                                segments=[
                                    LLMSegment(
                                        type="tool_call",
                                        content=partial_json,
                                        meta=LLMToolCallSegmentMeta(
                                            call_id=call_id,
                                            tool_name=tool_names[index],
                                            is_delta=True,
                                        ),
                                    )
                                ],
                                provider_meta=[self._provider_meta_entry(model=model)],
                            )
                    elif event_type == "message_delta":
                        delta = event["delta"]
                        stop_reason = (
                            delta["stop_reason"] if "stop_reason" in delta else None
                        )
                        usage = event["usage"] if "usage" in event else usage
        except httpx.HTTPStatusError as err:
            raise _provider_error_from_httpx(err) from err
        except httpx.TransportError as err:
            raise LLMProviderError(f"{type(err).__name__}: {err}") from err

        latency_ms = (time.perf_counter() - start) * 1000.0
        tool_calls = [
            LLMToolCall(
                name=tool_names[index],
                arguments=tool_arguments[index],
                call_id=tool_block_ids[index],
            )
            for index in sorted(tool_arguments)
        ]
        segments = [LLMSegment(type="text", content=text)] if text != "" else []
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
            id=response_id,
            model=model,
            text=text,
            tool_calls=tool_calls,
            usage=self._usage_from_payload(usage),
            segments=segments,
            provider_meta=[
                self._provider_meta_entry(
                    model=model,
                    latency_ms=latency_ms,
                    response_id=response_id,
                )
            ],
            status=stop_reason,
        )
        yield LLMStreamEvent(
            event_type="response.completed",
            response=final_response,
            segments=segments,
            provider_meta=final_response.provider_meta,
        )


def _provider_error_from_httpx(err: httpx.HTTPStatusError) -> LLMProviderError:
    status_code = err.response.status_code
    message = f"{type(err).__name__}: {err}"
    text = err.response.text
    context_window = "context_length_exceeded" in text or (
        "context window" in text.lower()
    )
    retryable = status_code == 429 or 500 <= status_code
    return LLMProviderError(
        message,
        retryable=retryable,
        context_window=context_window,
        status_code=status_code,
    )
