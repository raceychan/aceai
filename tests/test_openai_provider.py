import io
from base64 import b64encode
from types import SimpleNamespace
from typing import cast

import pytest
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_image_gen_call_completed_event import (
    ResponseImageGenCallCompletedEvent,
)
from openai.types.responses.response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary as ReasoningSummary
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from aceai.errors import (
    AceAIConfigurationError,
    AceAIRuntimeError,
    AceAIValidationError,
)
from aceai.llm import LLMMessage
from aceai.llm.models import (
    LLMMessagePart,
    LLMResponseFormat,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
)
from aceai.llm.openai import OpenAI
from aceai.tools import tool
from aceai.tools._tool_sig import Annotated, spec


class NamespaceWithDump(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


class AttrMessagePart(dict):
    @property
    def type(self):
        return self["type"]


@pytest.fixture
def fake_openai_client():
    class FakeStreamManager:
        def __init__(self, events, final_response):
            self._events = list(events)
            self._final_response = final_response

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            return self._events.pop(0)

        async def get_final_response(self):
            return self._final_response

    class FakeResponses:
        def __init__(self):
            self.create_calls: list[dict] = []
            self.stream_calls: list[dict] = []
            self.response_payload = NamespaceWithDump(
                id="fake-response",
                model="fake-model",
                output_text="",
                output=[],
                usage=None,
                status="completed",
                reasoning=None,
            )
            self.stream_events: list = []
            self.final_stream_response = NamespaceWithDump(
                id="fake-stream-response",
                model="fake-stream-model",
                output_text="",
                output=[],
                usage=None,
                status="completed",
                reasoning=None,
            )

        async def create(self, **kwargs):
            self.create_calls.append(kwargs)
            return self.response_payload

        def stream(self, **kwargs):
            self.stream_calls.append(kwargs)
            return FakeStreamManager(self.stream_events, self.final_stream_response)

    class FakeClient:
        def __init__(self):
            self.transcription_calls: list[dict] = []
            self.transcription_text = ""

            async def transcribe(**kwargs):
                self.transcription_calls.append(kwargs)
                return SimpleNamespace(text=self.transcription_text)

            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(create=transcribe)
            )
            self.responses = FakeResponses()

    return FakeClient()


@pytest.fixture
def openai_provider(fake_openai_client) -> OpenAI:
    return OpenAI(
        client=fake_openai_client,
        default_meta={"model": "gpt-4o", "stream_model": "gpt-4o-mini"},
    )


@pytest.fixture
def openai_echo_spec():
    @tool
    def echo(
        message: Annotated[str, spec(description="Echo message")],
    ) -> str:
        return message

    return echo.tool_spec


def _messages_with_attr_parts(messages):
    return messages


def test_build_base_response_kwargs_maps_request_fields(
    openai_provider: OpenAI,
    openai_echo_spec,
) -> None:
    request = {
        "messages": _messages_with_attr_parts(
            [LLMMessage.build("system", "You are helpful.")]
        ),
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stop": ["END"],
        "response_format": LLMResponseFormat(type="json_object"),
        "tools": [openai_echo_spec],
        "tool_choice": "auto",
        "metadata": {"model": "gpt-4o"},
    }

    with pytest.warns(UserWarning):
        params = openai_provider._build_base_response_kwargs(request)

    assert params["model"] == "gpt-4o"
    assert params["max_output_tokens"] == 256
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
    assert params["text"] == {"format": {"type": "json_object"}}
    assert params["tools"][0]["name"] == "echo"
    assert params["tool_choice"] == "auto"


def test_modality_reports_image_support(openai_provider: OpenAI) -> None:
    modality = openai_provider.modality
    assert modality.image_in is True
    assert modality.image_out is True


@pytest.mark.anyio
async def test_stt_delegates_to_openai_client(
    fake_openai_client, openai_provider: OpenAI
) -> None:
    audio_bytes = io.BytesIO(b"pcm")
    fake_openai_client.transcription_text = "transcribed"

    text = await openai_provider.stt(
        "clip.wav",
        audio_bytes,
        model="gpt-whisper",
        prompt="noise profile",
    )

    assert text == "transcribed"
    assert fake_openai_client.transcription_calls[0]["file"][0] == "clip.wav"
    assert fake_openai_client.transcription_calls[0]["prompt"] == "noise profile"


def test_build_base_response_kwargs_requires_messages(
    openai_provider: OpenAI,
) -> None:
    with pytest.raises(AceAIValidationError):
        openai_provider._build_base_response_kwargs({})


def test_build_base_response_kwargs_requires_model(
    openai_provider: OpenAI,
) -> None:
    request = {"messages": _messages_with_attr_parts([LLMMessage.build("system", "x")])}
    with pytest.raises(AceAIConfigurationError):
        openai_provider._build_base_response_kwargs(request)


def test_build_base_response_kwargs_accepts_reasoning_for_supported_model(
    openai_provider: OpenAI,
) -> None:
    request = {
        "messages": _messages_with_attr_parts([LLMMessage.build("system", "start")]),
        "metadata": {
            "model": "gpt-5o",
            "reasoning": {"summary": "auto"},
        },
    }

    params = openai_provider._build_base_response_kwargs(request)

    assert params["reasoning"] == {"summary": "auto"}


def test_build_base_response_kwargs_skips_temperature_for_gpt5(
    openai_provider: OpenAI,
) -> None:
    request = {
        "messages": _messages_with_attr_parts([LLMMessage.build("system", "start")]),
        "temperature": 0.2,
        "metadata": {"model": "gpt-5o-mini"},
    }

    params = openai_provider._build_base_response_kwargs(request)

    assert "temperature" not in params
    assert params["model"] == "gpt-5o-mini"


def test_build_base_response_kwargs_rejects_reasoning_for_unsupported_model(
    openai_provider: OpenAI,
) -> None:
    request = {
        "messages": [LLMMessage.build("system", "start")],
        "metadata": {
            "model": "gpt-4o",
            "reasoning": {"summary": "auto"},
        },
    }

    with pytest.raises(AceAIConfigurationError):
        openai_provider._build_base_response_kwargs(request)


def test_supports_reasoning_summary(openai_provider: OpenAI) -> None:
    assert openai_provider._supports_reasoning_summary("o4-mini") is True
    assert openai_provider._supports_reasoning_summary("o3-large") is True
    assert openai_provider._supports_reasoning_summary("gpt-5o") is True
    assert openai_provider._supports_reasoning_summary("gpt-4o") is False


def test_format_messages_for_responses_includes_tool_outputs(
    openai_provider: OpenAI,
) -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    messages = _messages_with_attr_parts(
        [
            LLMMessage.build("system", "Start"),
            LLMToolCallMessage(tool_calls=[call]),
            LLMToolUseMessage.from_content("result", name="lookup", call_id="call-1"),
        ]
    )

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"][0] == {"type": "input_text", "text": "Start"}
    assert formatted[1]["type"] == "function_call"
    assert formatted[1]["name"] == "lookup"
    assert formatted[2]["type"] == "function_call_output"
    assert formatted[2]["output"] == "result"


def test_format_messages_includes_tool_call_content(
    openai_provider: OpenAI,
) -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    message = LLMToolCallMessage(
        content=[LLMMessagePart(type="text", data="tool prelude")],
        tool_calls=[call],
    )

    formatted = openai_provider._format_messages_for_responses([message])

    assert formatted[0]["content"][0]["text"] == "tool prelude"


def test_format_messages_supports_image_parts(openai_provider: OpenAI) -> None:
    parts = [
        LLMMessagePart(type="text", data="describe this"),
        LLMMessagePart(
            type="image",
            url="https://example.com/image.png",
            data=b"",
        ),
    ]
    messages = _messages_with_attr_parts(
        [
            LLMMessage.build("user", parts),
        ]
    )

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted[0]["content"][0] == {"type": "input_text", "text": "describe this"}
    assert formatted[0]["content"][1]["type"] == "input_image"
    assert formatted[0]["content"][1]["image_url"] == "https://example.com/image.png"


def test_coerce_text_content_rejects_non_text_parts(openai_provider: OpenAI) -> None:
    part = LLMMessagePart(type="image", data="")
    with pytest.raises(ValueError, match="tool output"):
        openai_provider._coerce_text_content([part], context="tool output")


@pytest.mark.parametrize(
    "part",
    [
        LLMMessagePart(type="audio", data=b"wav"),
        LLMMessagePart(type="file", data=b"blob"),
        cast(LLMMessagePart, AttrMessagePart(type="binary", data="x")),
    ],
)
def test_format_content_parts_rejects_unsupported_modalities(
    openai_provider: OpenAI, part: LLMMessagePart
) -> None:
    with pytest.raises(ValueError):
        openai_provider._format_content_parts([part])


def test_format_image_part_accepts_inline_data(openai_provider: OpenAI) -> None:
    part = LLMMessagePart(type="image", data=b"\x01\x02", mime_type="image/jpeg")

    formatted = openai_provider._format_image_part(part)

    assert formatted["image_url"].startswith("data:image/jpeg;base64,")


def test_format_image_part_requires_url_or_bytes(openai_provider: OpenAI) -> None:
    part = LLMMessagePart(type="image", data="not-bytes")
    with pytest.raises(ValueError):
        openai_provider._format_image_part(part)


def test_build_text_config_variants(openai_provider: OpenAI) -> None:
    json_object = LLMResponseFormat(type="json_object")
    json_schema = LLMResponseFormat(type="json_schema", schema={"name": "Payload"})
    plain = LLMResponseFormat()

    assert openai_provider._build_text_config(json_object) == {
        "format": {"type": "json_object"}
    }
    assert openai_provider._build_text_config(json_schema) == {
        "format": {
            "type": "json_schema",
            "schema": {"name": "Payload"},
            "name": "response_schema",
        }
    }
    assert openai_provider._build_text_config(plain) is None


def test_build_text_config_rejects_unknown_type(openai_provider: OpenAI) -> None:
    invalid = LLMResponseFormat(type="binary")  # type: ignore[arg-type]

    with pytest.raises(AceAIValidationError):
        openai_provider._build_text_config(invalid)


def test_extract_tool_calls_and_to_llm_response(openai_provider: OpenAI) -> None:
    image_payload = b64encode(b"image-bytes").decode()
    response = NamespaceWithDump(
        id="resp-1",
        model="gpt-4o",
        output_text="hello",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                name="lookup",
                arguments='{"x":1}',
                call_id="call-1",
                id=None,
                status="completed",
            ),
            ImageGenerationCall(
                id="img-1",
                result=image_payload,
                status="completed",
                type="image_generation_call",
            ),
        ],
        usage=NamespaceWithDump(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            output_tokens_details=None,
        ),
        status="completed",
        reasoning=None,
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.text == "hello"
    assert llm_response.tool_calls[0].name == "lookup"
    assert llm_response.usage is not None
    assert llm_response.usage.input_tokens == 1
    image_segment = next(seg for seg in llm_response.segments if seg.type == "image")
    assert image_segment.media is not None
    assert image_segment.media.data is not None


def test_to_llm_response_includes_reasoning_summary(openai_provider: OpenAI) -> None:
    reasoning_item = ResponseReasoningItem(
        id="rs_1",
        summary=[ReasoningSummary(text="Chain summary", type="summary_text")],
        type="reasoning",
        status="completed",
    )
    usage = NamespaceWithDump(
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        output_tokens_details=NamespaceWithDump(reasoning_tokens=7),
    )
    response = NamespaceWithDump(
        id="resp-5",
        model="o4-mini",
        output_text="final",
        output=[reasoning_item],
        usage=usage,
        status="completed",
        reasoning=NamespaceWithDump(
            effort="medium",
            summary="auto",
            generate_summary=None,
        ),
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.reasoning is not None
    assert llm_response.reasoning.tokens == 7
    assert llm_response.reasoning.config is not None
    assert llm_response.reasoning.config.effort == "medium"
    reasoning_segments = [
        seg for seg in llm_response.segments if seg.type == "reasoning"
    ]
    assert reasoning_segments and reasoning_segments[0].content == "Chain summary"


def test_extract_tool_calls_falls_back_to_item_id(
    openai_provider: OpenAI,
) -> None:
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        name="lookup",
        arguments="{}",
        call_id="temp",
    ).model_copy(update={"call_id": None, "id": "tool-1"})

    response = NamespaceWithDump(
        id="resp-2",
        model="gpt-4o",
        output_text="",
        output=[tool_call],
        usage=None,
        status="in_progress",
        reasoning=None,
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.tool_calls[0].call_id == "tool-1"


def test_extract_tool_calls_requires_identifier(openai_provider: OpenAI) -> None:
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        name="lookup",
        arguments="{}",
        call_id="temp",
    ).model_copy(update={"call_id": None, "id": None})
    response = NamespaceWithDump(
        id="resp-3",
        model="gpt-4o",
        output_text="",
        output=[tool_call],
        usage=None,
        status="errored",
        reasoning=None,
    )

    with pytest.raises(AceAIRuntimeError, match="call identifier"):
        openai_provider._extract_tool_calls(response)


def test_map_stream_event_handles_known_types(openai_provider: OpenAI) -> None:
    text_event = ResponseTextDeltaEvent(
        content_index=0,
        delta="Hi",
        item_id="item-1",
        logprobs=[],
        output_index=0,
        sequence_number=1,
        type="response.output_text.delta",
    )
    text_chunk = openai_provider._map_stream_event(text_event, model_name="gpt-4o")
    assert text_chunk is not None
    assert text_chunk.text_delta == "Hi"

    tool_event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"value":1}',
        item_id="item-2",
        output_index=0,
        sequence_number=2,
        type="response.function_call_arguments.delta",
    )
    tool_chunk = openai_provider._map_stream_event(tool_event, model_name="gpt-4o")
    assert tool_chunk is not None
    assert tool_chunk.tool_call_delta.arguments_delta == '{"value":1}'

    error_event = ResponseErrorEvent(
        code=None,
        message="boom",
        param=None,
        sequence_number=3,
        type="error",
    )
    error_chunk = openai_provider._map_stream_event(error_event, model_name="gpt-4o")
    assert error_chunk is not None
    assert error_chunk.error == "boom"

    fallback_event = NamespaceWithDump(
        type="response.output_text.delta",
        delta="fallback",
        item_id=None,
    )
    assert (
        openai_provider._map_stream_event(fallback_event, model_name="gpt-4o") is None
    )

    image_event = ResponseImageGenCallPartialImageEvent(
        item_id="img-1",
        output_index=0,
        partial_image_b64=b64encode(b"chunk").decode(),
        partial_image_index=0,
        sequence_number=4,
        type="response.image_generation_call.partial_image",
    )
    image_chunk = openai_provider._map_stream_event(image_event, model_name="gpt-4o")
    assert image_chunk is not None
    assert image_chunk.event_type == "response.media"
    assert image_chunk.segments[0].media is not None
    assert image_chunk.segments[0].type == "image"


def test_map_stream_event_returns_none_for_completed_image_events(
    openai_provider: OpenAI,
) -> None:
    event = ResponseImageGenCallCompletedEvent(
        item_id="img-1",
        output_index=0,
        sequence_number=5,
        type="response.image_generation_call.completed",
    )

    assert openai_provider._map_stream_event(event, model_name="gpt-4o") is None


def test_map_stream_event_handles_fallback_tool_and_error_events(
    openai_provider: OpenAI,
) -> None:
    fallback_tool = NamespaceWithDump(
        type="response.function_call_arguments.delta",
        delta="{}",
        item_id="call-1",
    )
    tool_chunk = openai_provider._map_stream_event(fallback_tool, model_name="gpt-4o")
    assert tool_chunk is None

    fallback_error = NamespaceWithDump(
        type="response.error",
        message="fail",
    )
    error_chunk = openai_provider._map_stream_event(fallback_error, model_name="gpt-4o")
    assert error_chunk is None


def test_map_stream_event_handles_fallback_partial_images(
    openai_provider: OpenAI,
) -> None:
    payload = b64encode(b"bytes").decode()
    fallback_image = NamespaceWithDump(
        type="response.image_generation_call.partial_image",
        partial_image_b64=payload,
        item_id="img-2",
        output_index=0,
        partial_image_index=1,
        sequence_number=6,
    )

    assert (
        openai_provider._map_stream_event(fallback_image, model_name="gpt-4o") is None
    )

    missing_payload = NamespaceWithDump(
        type="response.image_generation_call.partial_image",
        item_id="img-3",
        output_index=0,
        partial_image_index=0,
        sequence_number=7,
    )

    assert (
        openai_provider._map_stream_event(missing_payload, model_name="gpt-4o") is None
    )


@pytest.mark.anyio
async def test_complete_uses_default_metadata(
    fake_openai_client, openai_provider: OpenAI
) -> None:
    fake_openai_client.responses.response_payload.output_text = "done"
    request = {"messages": _messages_with_attr_parts([LLMMessage.build("system", "s")])}

    response = await openai_provider.complete(request)

    create_call = fake_openai_client.responses.create_calls[0]
    assert create_call["model"] == "gpt-4o"
    assert response.provider_meta[0].response_id == "fake-response"


@pytest.mark.anyio
async def test_stream_yields_completed_event(
    fake_openai_client, openai_provider: OpenAI
) -> None:
    text_event = ResponseTextDeltaEvent(
        content_index=0,
        delta="chunk",
        item_id="item-1",
        logprobs=[],
        output_index=0,
        sequence_number=1,
        type="response.output_text.delta",
    )
    fake_openai_client.responses.stream_events = [text_event]
    fake_openai_client.responses.final_stream_response.output_text = "final"
    request = {"messages": _messages_with_attr_parts([LLMMessage.build("system", "s")])}

    events: list[LLMStreamEvent] = []
    async for evt in openai_provider.stream(request):
        events.append(evt)

    assert events[-1].event_type == "response.completed"
    assert fake_openai_client.responses.stream_calls[0]["model"] == "gpt-4o"
