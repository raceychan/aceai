import pytest
from msgspec import DecodeError, Struct
from openai import OpenAIError
from types import SimpleNamespace

from aceai.errors import (
    AceAIConfigurationError,
    AceAIRuntimeError,
    AceAIValidationError,
    LLMProviderError,
)
from aceai.llm.models import (
    LLMMessage,
    LLMMessagePart,
    LLMProviderBase,
    LLMProviderModality,
    LLMResponse,
    LLMStreamEvent,
)
from aceai.llm.service import LLMService


class AttrMessagePart(dict):
    @property
    def type(self):
        return self["type"]


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
        modality: LLMProviderModality | None = None,
    ) -> None:
        self._responses = list(responses or [LLMResponse(text="ok")])
        self._stream_events = list(stream_events or [])
        self._default_model = default_model
        self._default_stream_model = default_stream_model
        self._modality = modality or LLMProviderModality()
        self.complete_requests: list[dict] = []
        self.stream_requests: list[dict] = []

    async def complete(self, request: dict, *, trace_ctx=None) -> LLMResponse:
        self.complete_requests.append(request)
        return self._responses.pop(0)

    def stream(self, request: dict, *, trace_ctx=None):
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

    @property
    def modality(self) -> LLMProviderModality:
        return self._modality

    async def stt(
        self, filename, file, *, model: str, prompt: str | None = None, trace_ctx=None
    ) -> str:  # pragma: no cover - not used
        return "transcript"


class ErroringProvider(LLMProviderBase):
    def __init__(self) -> None:
        self._default_model = "gpt-4o"
        self._default_stream_model = "gpt-4o-mini"

    async def complete(self, request: dict, *, trace_ctx=None) -> LLMResponse:
        raise OpenAIError("boom")

    def stream(self, request: dict, *, trace_ctx=None):
        raise AceAIRuntimeError("unused stream")

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_stream_model(self) -> str:
        return self._default_stream_model

    async def stt(
        self, filename, file, *, model: str, prompt: str | None = None, trace_ctx=None
    ) -> str:  # pragma: no cover - not used
        return ""


def test_llm_service_requires_providers() -> None:
    with pytest.raises(AceAIConfigurationError):
        LLMService([], timeout_seconds=1.0)


def test_llm_service_rotation_and_counts() -> None:
    providers = [RecordingProvider(), RecordingProvider()]
    service = LLMService(providers, timeout_seconds=1.0)

    assert service.get_provider_count() == 2
    assert service.has_provider is True
    assert service._get_current_provider() is providers[0]

    service._rotate_provider()

    assert service._get_current_provider() is providers[1]


@pytest.mark.anyio
async def test_llm_service_complete_passes_metadata_through() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text="ok")])
    service = LLMService([provider], timeout_seconds=1.0)
    metadata = {"model": provider.default_model}

    await service.complete(
        messages=[LLMMessage.build("system", "Prompt")],
        metadata=metadata,
    )

    assert provider.complete_requests[0]["metadata"] is metadata


@pytest.mark.anyio
async def test_llm_service_complete_updates_last_response() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text="fresh")])
    service = LLMService([provider], timeout_seconds=1.0)

    response = await service.complete(messages=[LLMMessage.build("system", "Prompt")])

    assert response.text == "fresh"


@pytest.mark.anyio
async def test_llm_service_stream_preserves_request_metadata() -> None:
    event = LLMStreamEvent(
        event_type="response.output_text.delta",
        text_delta="hello",
    )
    provider = RecordingProvider(stream_events=[event])
    service = LLMService([provider], timeout_seconds=1.0)
    metadata = {"model": provider.default_stream_model}

    received = []
    async for part in service.stream(
        messages=[LLMMessage.build("system", "s")],
        metadata=metadata,
    ):
        received.append(part.text_delta)

    assert received == ["hello"]
    assert provider.stream_requests[0]["metadata"] is metadata


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
            messages=[LLMMessage.build("user", "hi")],
        )


@pytest.mark.anyio
async def test_complete_json_inserts_hint_and_retries_on_error() -> None:
    responses = [
        LLMResponse(text="not json"),
        LLMResponse(text='{"value":7}'),
    ]
    provider = RecordingProvider(responses=responses)
    service = LLMService([provider], timeout_seconds=1.0, max_retries=2)
    messages = [LLMMessage.build("system", "start")]

    payload = await service.complete_json(
        schema=Payload,
        retries=2,
        messages=messages,
    )

    assert payload.value == 7
    assert len(provider.complete_requests) == 2
    schema_hint = provider.complete_requests[0]["messages"][1]
    assert "Return Format Advisory" in schema_hint.content[0]["data"]
    error_notice = provider.complete_requests[1]["messages"][-1]
    assert error_notice.role == "system"
    assert "Error handling notice" in error_notice.content[0]["data"]


@pytest.mark.anyio
async def test_complete_passes_image_parts_through_without_validation() -> None:
    provider = RecordingProvider(
        modality=LLMProviderModality(image_in=False),
        responses=[LLMResponse(text="ok")],
    )
    service = LLMService([provider], timeout_seconds=1.0)

    image_part = LLMMessagePart(type="image", data=b"bytes", mime_type="image/png")
    message = LLMMessage.build("user", [image_part])

    response = await service.complete(messages=[message])

    assert response.text == "ok"
    forwarded = provider.complete_requests[0]["messages"][0].content[0]
    assert forwarded["type"] == "image"


def test_validate_messages_requires_messages() -> None:
    service = LLMService([RecordingProvider()], timeout_seconds=1.0)
    with pytest.raises(ValueError):
        service._validate_messages({})


def test_validate_messages_requires_list_content() -> None:
    service = LLMService([RecordingProvider()], timeout_seconds=1.0)
    bad_message = SimpleNamespace(role="user", content="oops")
    with pytest.raises(TypeError):
        service._validate_messages({"messages": [bad_message]})


@pytest.mark.parametrize(
    ("part", "modality"),
    [
        (LLMMessagePart(type="text", data="t"), LLMProviderModality(text_in=False)),
        (
            LLMMessagePart(type="image", data=b"raw"),
            LLMProviderModality(image_in=False),
        ),
        (
            LLMMessagePart(type="audio", data=b"pcm"),
            LLMProviderModality(audio_in=False),
        ),
        (
            LLMMessagePart(type="file", data=b"blob"),
            LLMProviderModality(file_in=False),
        ),
    ],
)
def test_validate_messages_enforces_modality_support(
    part: LLMMessagePart, modality: LLMProviderModality
) -> None:
    provider = RecordingProvider(modality=modality)
    service = LLMService([provider], timeout_seconds=1.0)
    message = LLMMessage.build("user", [part])

    with pytest.raises(LLMProviderError):
        service._validate_messages({"messages": [message]})


def test_validate_messages_rejects_unknown_part_type() -> None:
    provider = RecordingProvider()
    service = LLMService([provider], timeout_seconds=1.0)
    message = LLMMessage(
        role="user",
        content=[AttrMessagePart(type="binary", data="payload")],
    )

    with pytest.raises(ValueError):
        service._validate_messages({"messages": [message]})


@pytest.mark.anyio
async def test_complete_json_with_retry_raises_after_failures() -> None:
    responses = [
        LLMResponse(text="nah"),
        LLMResponse(text="still nah"),
    ]
    provider = RecordingProvider(responses=responses)
    service = LLMService([provider], timeout_seconds=1.0)
    messages = [LLMMessage.build("system", "start")]

    with pytest.raises(DecodeError):
        await service._complete_json_with_retry(
            schema=Payload,
            retries=2,
            messages=messages,
        )


@pytest.mark.anyio
async def test_complete_json_without_retries_short_circuits() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text='{"value":3}')])
    service = LLMService([provider], timeout_seconds=1.0)
    messages = [LLMMessage.build("system", "start")]

    payload = await service.complete_json(
        schema=Payload,
        retries=0,
        messages=messages,
    )

    assert payload.value == 3
