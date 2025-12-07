import pytest
from msgspec import DecodeError, Struct
from openai import OpenAIError

from aceai.errors import (
    AceAIConfigurationError,
    AceAIRuntimeError,
    AceAIValidationError,
)
from aceai.llm.models import (
    LLMMessage,
    LLMProviderBase,
    LLMResponse,
    LLMStreamEvent,
)
from aceai.llm.service import LLMProviderError, LLMService


class Payload(Struct):
    value: int


class RecordingProvider(LLMProviderBase):
    def __init__(
        self,
        *,
        responses: list[LLMResponse] | None = None,
        stream_events: list[LLMStreamEvent] | None = None,
        default_model: str = "gpt-4o",
        default_stream_model: str = "gpt-4o-mini",
    ) -> None:
        self._responses = list(responses or [LLMResponse(text="ok")])
        self._stream_events = list(stream_events or [])
        self._default_model = default_model
        self._default_stream_model = default_stream_model
        self.complete_requests: list[dict] = []
        self.stream_requests: list[dict] = []

    async def complete(self, request: dict) -> LLMResponse:
        self.complete_requests.append(request)
        return self._responses.pop(0)

    def stream(self, request: dict):
        self.stream_requests.append(request)

        async def iterator():
            for event in self._stream_events:
                yield event

        return iterator()

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_stream_model(self) -> str:
        return self._default_stream_model

    async def stt(self, filename, file, *, model: str) -> str:  # pragma: no cover - not used
        return "transcript"


class ErroringProvider(LLMProviderBase):
    def __init__(self) -> None:
        self._default_model = "gpt-4o"
        self._default_stream_model = "gpt-4o-mini"

    async def complete(self, request: dict) -> LLMResponse:
        raise OpenAIError("boom")

    def stream(self, request: dict):
        raise AceAIRuntimeError("unused stream")

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_stream_model(self) -> str:
        return self._default_stream_model

    async def stt(self, filename, file, *, model: str) -> str:  # pragma: no cover - not used
        return ""


def test_llm_service_requires_providers() -> None:
    with pytest.raises(AceAIConfigurationError):
        LLMService([], timeout_seconds=1.0)


@pytest.mark.anyio
async def test_llm_service_complete_applies_default_model() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text="ok")])
    service = LLMService([provider], timeout_seconds=1.0)

    await service.complete(messages=[LLMMessage(role="system", content="Prompt")])

    metadata = provider.complete_requests[0]["metadata"]
    assert metadata["model"] == provider.default_model


@pytest.mark.anyio
async def test_llm_service_stream_uses_default_stream_model() -> None:
    event = LLMStreamEvent(
        event_type="response.output_text.delta",
        text_delta="hello",
    )
    provider = RecordingProvider(stream_events=[event])
    service = LLMService([provider], timeout_seconds=1.0)

    received = []
    async for part in service.stream(messages=[LLMMessage(role="system", content="s")]):
        received.append(part.text_delta)

    assert received == ["hello"]
    assert provider.stream_requests[0]["metadata"]["model"] == provider.default_stream_model


@pytest.mark.anyio
async def test_complete_json_validates_message_structure() -> None:
    service = LLMService([RecordingProvider()], timeout_seconds=1.0)

    with pytest.raises(AceAIValidationError):
        await service.complete_json(schema=Payload)

    with pytest.raises(AceAIValidationError):
        await service.complete_json(schema=Payload, messages=[])

    with pytest.raises(AceAIValidationError):
        await service.complete_json(
            schema=Payload,
            messages=[LLMMessage(role="user", content="hi")],
        )


@pytest.mark.anyio
async def test_complete_json_inserts_hint_and_retries_on_error() -> None:
    responses = [
        LLMResponse(text="not json"),
        LLMResponse(text='{"value":7}'),
    ]
    provider = RecordingProvider(responses=responses)
    service = LLMService([provider], timeout_seconds=1.0, max_retries=2)
    messages = [LLMMessage(role="system", content="start")]

    payload = await service.complete_json(
        schema=Payload,
        retries=2,
        messages=messages,
    )

    assert payload.value == 7
    assert len(provider.complete_requests) == 2
    schema_hint = provider.complete_requests[0]["messages"][1]
    assert "Return Format Advisory" in schema_hint.content
    error_notice = provider.complete_requests[1]["messages"][-1]
    assert error_notice.role == "system"
    assert "Error handling notice" in error_notice.content


@pytest.mark.anyio
async def test_llm_service_wraps_openai_errors() -> None:
    service = LLMService([ErroringProvider()], timeout_seconds=0.1)

    with pytest.raises(LLMProviderError, match="LLM provider error"):
        await service.complete(messages=[LLMMessage(role="system", content="prompt")])


@pytest.mark.anyio
async def test_llm_service_rotation_and_last_response_tracking() -> None:
    provider_one = RecordingProvider(responses=[LLMResponse(text="first")])
    provider_two = RecordingProvider(responses=[LLMResponse(text="second")])
    service = LLMService([provider_one, provider_two], timeout_seconds=1.0)

    assert service.get_provider_count() == 2
    assert service.has_provider
    assert service.last_response is None

    await service.complete(messages=[LLMMessage(role="system", content="start")])
    assert service.last_response is not None
    assert service.last_response.text == "first"

    service._rotate_provider()
    assert service._get_current_provider() is provider_two


@pytest.mark.anyio
async def test_complete_json_retry_failure_raises_decode_error() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text="not json")])
    service = LLMService([provider], timeout_seconds=1.0)
    messages = [LLMMessage(role="system", content="ctx")]

    with pytest.raises(DecodeError):
        await service._complete_json_with_retry(
            schema=Payload,
            retries=1,
            messages=messages,
        )


@pytest.mark.anyio
async def test_complete_json_zero_retries_decodes_once() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text='{"value":5}')])
    service = LLMService([provider], timeout_seconds=1.0)
    result = await service.complete_json(
        schema=Payload,
        retries=0,
        messages=[LLMMessage(role="system", content="ctx")],
    )

    assert result.value == 5
    assert len(provider.complete_requests) == 1
