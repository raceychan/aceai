import pytest
from msgspec import Struct

from aceai.llm.models import (
    LLMMessage,
    LLMProviderBase,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamEvent,
)
from aceai.llm.service import LLMService


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


@pytest.mark.anyio
async def test_llm_service_complete_applies_default_model() -> None:
    provider = RecordingProvider(responses=[LLMResponse(text="ok")])
    service = LLMService([provider], timeout_seconds=1.0)

    await service.complete(messages=[LLMMessage(role="system", content="Prompt")])

    metadata = provider.complete_requests[0]["metadata"]
    assert metadata["model"] == provider.default_model


@pytest.mark.anyio
async def test_llm_service_stream_uses_default_stream_model() -> None:
    chunk = LLMStreamChunk(text_delta="hello")
    event = LLMStreamEvent(
        event_type="response.output_text.delta",
        chunk=chunk,
    )
    provider = RecordingProvider(stream_events=[event])
    service = LLMService([provider], timeout_seconds=1.0)

    received = []
    async for part in service.stream(messages=[LLMMessage(role="system", content="s")]):
        received.append(part.chunk.text_delta)

    assert received == ["hello"]
    assert provider.stream_requests[0]["metadata"]["model"] == provider.default_stream_model


@pytest.mark.anyio
async def test_complete_json_validates_message_structure() -> None:
    service = LLMService([RecordingProvider()], timeout_seconds=1.0)

    with pytest.raises(ValueError):
        await service.complete_json(schema=Payload, messages=[])

    with pytest.raises(ValueError):
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
