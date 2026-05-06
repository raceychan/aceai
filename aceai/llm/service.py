"""LLM Service - Provider-agnostic LLM interface with clean responsibilities."""

import asyncio
from typing import (
    AsyncGenerator,
    AsyncIterator,
    BinaryIO,
    Protocol,
    Type,
    TypeVar,
    Unpack,
)

import httpx
from msgspec import DecodeError, ValidationError
from msgspec.json import decode
from msgspec.json import encode as json_encode
from msgspec.json import schema as get_schema
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from aceai.llm.errors import (
    AceAIConfigurationError,
    AceAIValidationError,
    LLMProviderError,
)

from .models import (
    LLMInput,
    LLMMessage,
    LLMProviderBase,
    LLMProviderModality,
    LLMResponse,
    LLMStreamEvent,
)

JSONDecodeErrors = (ValidationError, DecodeError)
LLM_RETRY_EXHAUSTED_MESSAGE = "LLM request failed after retries. Please try again later."
RetryableProviderErrors = (
    TimeoutError,
    httpx.TransportError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)

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
        self, **request: Unpack[LLMInput]
    ) -> LLMResponse: ...

    async def complete_json(
        self,
        *,
        schema: Type[T],
        retries: int | None = None,
        **request: Unpack[LLMInput],
    ) -> T: ...

    async def transcribe(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str | None = None,
        prompt: str | None = None,
    ) -> str: ...

    def stream(
        self, **request: Unpack[LLMInput]
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
        max_retries: int = 5,
        retry_initial_delay_seconds: float = 0.5,
        retry_backoff_factor: float = 2.0,
        stream_event_timeout_seconds: float = 3.0,
    ):
        if not providers:
            raise AceAIConfigurationError("At least one provider is required")
        self._providers = providers
        self._current_provider_index = 0
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_initial_delay_seconds = retry_initial_delay_seconds
        self._retry_backoff_factor = retry_backoff_factor
        self._stream_event_timeout_seconds = stream_event_timeout_seconds

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

    def _validate_messages(self, llm_input: LLMInput) -> None:
        if "messages" not in llm_input or not llm_input["messages"]:
            raise ValueError("LLM requests require at least one message")
        modality: LLMProviderModality = self._get_current_provider().modality
        for message in llm_input["messages"]:
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
                        raise ValueError(
                            f"Unsupported message part type: {part['type']}"
                        )

    async def _complete_json_with_retry(
        self,
        *,
        schema: Type[T],
        retries: int,
        llm_input: LLMInput,
    ) -> T:
        attempts = max(1, retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            raw_response = await self._complete_request_with_retry(llm_input)
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
            llm_input["messages"].append(error_message)

        assert last_error is not None
        raise last_error

    async def complete(
        self, **request: Unpack[LLMInput]
    ) -> LLMResponse:
        """Complete using a unified LLMInput or raw messages."""
        return await self._complete_request_with_retry(request)

    async def _complete_request_with_retry(self, request: LLMInput) -> LLMResponse:
        retries_remaining = self._max_retries
        retry_count = 0
        while True:
            try:
                return await self._complete_request(request)
            except RetryableProviderErrors as err:
                if not _is_retryable_provider_error(err) or retries_remaining == 0:
                    raise
                retry_count += 1
                retries_remaining -= 1
                await asyncio.sleep(self._retry_delay_seconds(retry_count))

    async def _complete_request(self, request: LLMInput) -> LLMResponse:
        provider = self._get_current_provider()
        coro = provider.complete(request)
        timeout = self._timeout_seconds
        return await asyncio.wait_for(coro, timeout=timeout)

    async def complete_json(
        self,
        *,
        schema: Type[T],
        retries: int | None = None,
        **request: Unpack[LLMInput],
    ) -> T:
        """Complete a request and parse the response into the provided schema.

        On decode failure, appends a system error-handling message with the decoder
        error to help the LLM self-correct, retrying up to `max_retries` times.
        """
        if "messages" not in request:
            raise AceAIValidationError("LLMInput.messages is required")

        messages = request["messages"]
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
            raw_response = await self._complete_request_with_retry(request)
            return decode(raw_response.text, type=schema)

        return await self._complete_json_with_retry(
            schema=schema,
            retries=retries,
            llm_input=request,
        )

    async def transcribe(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str | None = None,
        prompt: str | None = None,
    ) -> str:
        provider = self._get_current_provider()
        if model is None:
            raise AceAIConfigurationError("STT requests require a model identifier")

        coro = provider.stt(
            filename,
            file,
            model=model,
            prompt=prompt,
        )
        return await asyncio.wait_for(coro, timeout=self._timeout_seconds)

    async def stream(
        self, **request: Unpack[LLMInput]
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Provider passthrough streaming with bounded retry on transport failures.

        Business logic (e.g., output sanitization/formatting) must be handled at
        higher layers such as ContextManager or agents, not here.
        """
        retry_count = 0
        while True:
            provider = self._get_current_provider()
            stream_resp: AsyncGenerator[LLMStreamEvent, None] | None = None
            try:
                stream_resp = provider.stream(request)
                while True:
                    event = await _next_stream_event(
                        stream_resp,
                        timeout_seconds=self._stream_event_timeout_seconds,
                    )
                    yield event
            except StopAsyncIteration:
                return
            except RetryableProviderErrors as err:
                if (
                    not _is_retryable_provider_error(err)
                    or retry_count == self._max_retries
                ):
                    yield LLMStreamEvent(
                        event_type="response.completed",
                        response=LLMResponse(
                            text=LLM_RETRY_EXHAUSTED_MESSAGE,
                            status="failed",
                        ),
                        segments=[],
                    )
                    return
                retry_count += 1
                delay_seconds = self._retry_delay_seconds(retry_count)
                yield LLMStreamEvent(
                    event_type="response.retrying",
                    error=_provider_error_message(err),
                    retry_count=retry_count,
                    retry_max=self._max_retries,
                    retry_delay_seconds=delay_seconds,
                )
                await asyncio.sleep(delay_seconds)
            finally:
                if stream_resp is not None:
                    await stream_resp.aclose()

    def _retry_delay_seconds(self, retry_count: int) -> float:
        return self._retry_initial_delay_seconds * (
            self._retry_backoff_factor ** (retry_count - 1)
        )


def _provider_error_message(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


async def _next_stream_event(
    stream: AsyncIterator[LLMStreamEvent],
    *,
    timeout_seconds: float,
) -> LLMStreamEvent:
    return await asyncio.wait_for(stream.__anext__(), timeout=timeout_seconds)


def _is_retryable_provider_error(err: BaseException) -> bool:
    if isinstance(err, APIStatusError):
        return err.status_code == 429 or 500 <= err.status_code
    return True
