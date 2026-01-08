"""LLM Service - Provider-agnostic LLM interface with clean responsibilities."""

import asyncio
from typing import AsyncGenerator, BinaryIO, Protocol, Type, TypeVar, Unpack

from msgspec import DecodeError, ValidationError
from msgspec.json import decode
from msgspec.json import encode as json_encode
from msgspec.json import schema as get_schema
from opentelemetry.context import Context

from aceai.errors import AceAIConfigurationError, AceAIValidationError, LLMProviderError

from .models import (
    LLMMessage,
    LLMProviderBase,
    LLMProviderModality,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)

JSONDecodeErrors = (ValidationError, DecodeError)

T = TypeVar("T")

ERROR_PROMPT_TEMPLATE = (
    "Error handling notice:\n"
    "Expected JSON schema: {schema_name}\n"
    "Decoder error: {error}\n"
    "Please respond again with ONLY valid JSON that conforms to the schema. "
    "Do not include explanations or surrounding text."
)


class ILLMService(Protocol):
    async def complete(
        self, *, trace_ctx: Context | None = None, **request: Unpack[LLMRequest]
    ) -> LLMResponse: ...

    async def complete_json(
        self,
        *,
        schema: Type[T],
        retries: int | None = None,
        trace_ctx: Context | None = None,
        **request: Unpack[LLMRequest],
    ) -> T: ...

    async def transcribe(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str | None = None,
        prompt: str | None = None,
        trace_ctx: Context | None = None,
    ) -> str: ...

    def stream(
        self, *, trace_ctx: Context | None = None, **request: Unpack[LLMRequest]
    ) -> AsyncGenerator[LLMStreamEvent, None]: ...

    def get_provider_count(self) -> int: ...

    @property
    def has_provider(self) -> bool: ...


class LLMService:
    """
    Clean LLM service that handles provider selection, timeouts, retries, and response parsing.

    Responsibilities:
    - Provider selection and round-robin
    - Timeouts and retries
    - Response parsing helpers (e.g., complete_json)

    Does NOT own:
    - Domain prompts or templates
    - Plan/eval semantics
    - Schema validation into domain models
    """

    def __init__(
        self,
        providers: list[LLMProviderBase],
        timeout_seconds: float,
        max_retries: int = 2,
    ):
        if not providers:
            raise AceAIConfigurationError("At least one provider is required")
        self._providers = providers
        self._current_provider_index = 0
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def _get_current_provider(self) -> LLMProviderBase:
        """Get the current provider (round-robin)."""
        return self._providers[self._current_provider_index]

    def _rotate_provider(self) -> None:
        """Rotate to the next provider."""
        self._current_provider_index = (self._current_provider_index + 1) % len(
            self._providers
        )

    def get_provider_count(self) -> int:
        """Get the number of available providers."""
        return len(self._providers)

    @property
    def has_provider(self) -> bool:
        """Get the current provider for direct access."""
        return self.get_provider_count() > 0

    def _validate_messages(self, request: LLMRequest) -> None:
        messages = request.get("messages")
        if not messages:
            raise ValueError("LLM requests require at least one message")
        modality: LLMProviderModality = self._get_current_provider().modality
        for message in messages:
            if not isinstance(message.content, list):
                raise TypeError("LLMMessage.content must be a list[LLMMessagePart]")
            for part in message.content:
                match part["type"]:
                    case "text":
                        if not modality.text_in:
                            raise LLMProviderError(
                                "Provider does not support text input"
                            )
                    case "image":
                        if not modality.image_in:
                            raise LLMProviderError(
                                "Provider does not support image input"
                            )
                    case "audio":
                        if not modality.audio_in:
                            raise LLMProviderError(
                                "Provider does not support audio input"
                            )
                    case "file":
                        if not modality.file_in:
                            raise LLMProviderError(
                                "Provider does not support file input"
                            )
                    case _:
                        raise ValueError(f"Unsupported message part type: {part.type}")

    async def _complete_json_with_retry(
        self,
        *,
        schema: Type[T],
        retries: int,
        trace_ctx: Context | None = None,
        **request: Unpack[LLMRequest],
    ) -> T:
        attempts = max(1, retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            raw_response = await self.complete(trace_ctx=trace_ctx, **request)
            raw_text = raw_response.text
            try:
                return decode(raw_text, type=schema)
            except JSONDecodeErrors as err:
                last_error = err

            if attempt == attempts - 1:
                assert last_error is not None
                raise last_error

            schema_name = schema.__name__
            error_prompt = ERROR_PROMPT_TEMPLATE.format(
                schema_name=schema_name,
                error=str(last_error),
            )
            error_message = LLMMessage.build(
                content=error_prompt,
                role="system",
            )
            request["messages"].append(error_message)

        assert last_error is not None
        raise last_error

    async def complete(
        self, *, trace_ctx: Context | None = None, **request: Unpack[LLMRequest]
    ) -> LLMResponse:
        """Complete using a unified LLMRequest or raw messages."""
        # Apply default max_tokens based on settings and model if not provided
        provider = self._get_current_provider()
        coro = provider.complete(request, trace_ctx=trace_ctx)
        timeout = self._timeout_seconds
        return await asyncio.wait_for(coro, timeout=timeout)

    async def complete_json(
        self,
        *,
        schema: Type[T],
        retries: int | None = None,
        trace_ctx: Context | None = None,
        **request: Unpack[LLMRequest],
    ) -> T:
        """Complete a request and parse the response into the provided schema.

        On decode failure, appends a system error-handling message with the decoder
        error to help the LLM self-correct, retrying up to `max_retries` times.
        """
        messages = request.get("messages")
        if not messages:
            raise AceAIValidationError("complete_json requires at least one message")

        first_message = messages[0]
        if first_message.role != "system":
            raise AceAIValidationError(
                "complete_json expects the first message to be a system message"
            )

        schema_output = get_schema(schema)
        schema_def = schema_output.get("$defs", schema_output)

        schema_hint = LLMMessage.build(
            role="system",
            content="\n\nReturn Format Advisory:\n"
            f"You must respond with ONLY JSON that matches the `{schema.__name__}` schema.\n"
            f"JSON Schema:\n{json_encode(schema_def).decode()}\n",
        )
        messages.insert(1, schema_hint)

        retries = retries or self._max_retries

        if retries <= 0:
            raw_response = await self.complete(trace_ctx=trace_ctx, **request)
            return decode(raw_response.text, type=schema)

        return await self._complete_json_with_retry(
            schema=schema,
            retries=retries,
            trace_ctx=trace_ctx,
            **request,
        )

    async def transcribe(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str | None = None,
        prompt: str | None = None,
        trace_ctx: Context | None = None,
    ) -> str:
        provider = self._get_current_provider()
        if model is None:
            raise AceAIConfigurationError("STT requests require a model identifier")

        coro = provider.stt(
            filename,
            file,
            model=model,
            prompt=prompt,
            trace_ctx=trace_ctx,
        )
        return await asyncio.wait_for(coro, timeout=self._timeout_seconds)

    async def stream(
        self, *, trace_ctx: Context | None = None, **request: Unpack[LLMRequest]
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Provider passthrough streaming.

        Business logic (e.g., output sanitization/formatting) must be handled at
        higher layers such as ContextManager or agents, not here.
        """
        provider = self._get_current_provider()
        stream_resp = provider.stream(request, trace_ctx=trace_ctx)
        try:
            async for event in stream_resp:
                yield event
        finally:
            await stream_resp.aclose()
