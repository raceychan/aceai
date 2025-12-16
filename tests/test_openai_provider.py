from base64 import b64encode
from types import SimpleNamespace

import pytest
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from aceai.llm import LLMMessage
from aceai.llm.models import (
    LLMMessagePart,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
    LLMResponseFormat,
)
from aceai.llm.openai import OpenAI
from aceai.tools.interface import ToolSpec


@pytest.fixture
def fake_openai_client():
    async def transcribe(**_kwargs):
        return SimpleNamespace(text="")

    class FakeStreamManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def get_final_response(self):
            return SimpleNamespace(
                id="fake-response",
                model="fake-model",
                output_text="",
                output=[],
                usage=None,
                status="completed",
            )

    class FakeResponses:
        async def create(self, **kwargs):
            return SimpleNamespace(
                id="fake-response",
                model=kwargs.get("model", "fake-model"),
                output_text="",
                output=[],
                usage=None,
                status="completed",
            )

        def stream(self, **kwargs):
            return FakeStreamManager()

    class FakeClient:
        def __init__(self):
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(create=transcribe)
            )
            self.responses = FakeResponses()

    return FakeClient()


@pytest.fixture
def openai_provider(fake_openai_client) -> OpenAI:
    return OpenAI(
        client=fake_openai_client,
        default_model="gpt-4o",
        default_stream_model="gpt-4o-mini",
    )


def _messages_with_attr_parts(messages):
    normalized = []
    for message in messages:
        if message.content:
            message.content = [
                part if hasattr(part, "type") else SimpleNamespace(**part)
                for part in message.content
            ]
        normalized.append(message)
    return normalized


def test_build_base_response_kwargs_maps_request_fields(
    openai_provider: OpenAI,
) -> None:
    tool_spec: ToolSpec = {
        "type": "function",
        "name": "echo",
        "description": "Echo",
        "parameters": {"type": "object", "properties": {}},
    }
    request = {
        "messages": _messages_with_attr_parts(
            [LLMMessage.build("system", "You are helpful.")]
        ),
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stop": ["END"],
        "response_format": LLMResponseFormat(type="json_object"),
        "tools": [tool_spec],
        "tool_choice": "auto",
        "metadata": {},
    }

    with pytest.warns(UserWarning):
        params = openai_provider._build_base_response_kwargs(
            request, default_model="gpt-4o"
        )

    assert params["model"] == "gpt-4o"
    assert params["max_output_tokens"] == 256
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
    assert params["text"] == {"format": {"type": "json_object"}}
    assert params["tools"][0]["name"] == "echo"
    assert params["tool_choice"] == "auto"


def test_build_base_response_kwargs_skips_temperature_for_gpt5(
    openai_provider: OpenAI,
) -> None:
    request = {
        "messages": _messages_with_attr_parts([LLMMessage.build("system", "start")]),
        "temperature": 0.2,
        "metadata": {"model": "gpt-5o-mini"},
    }

    params = openai_provider._build_base_response_kwargs(
        request, default_model="fallback"
    )

    assert "temperature" not in params
    assert params["model"] == "gpt-5o-mini"


def test_format_messages_for_responses_includes_tool_outputs(
    openai_provider: OpenAI,
) -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    messages = _messages_with_attr_parts(
        [
            LLMMessage.build("system", "Start"),
            LLMToolCallMessage(tool_calls=[call]),
            LLMToolUseMessage.build("result", name="lookup", call_id="call-1"),
        ]
    )

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"][0] == {"type": "input_text", "text": "Start"}
    assert formatted[1]["type"] == "function_call"
    assert formatted[1]["name"] == "lookup"
    assert formatted[2]["type"] == "function_call_output"
    assert formatted[2]["output"] == "result"


def test_format_messages_allows_tool_output_without_call_id(
    openai_provider: OpenAI,
) -> None:
    messages = _messages_with_attr_parts(
        [
            LLMToolUseMessage.build("oops", name="orphan", call_id=None),
        ]
    )

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted == [
        {
            "type": "function_call_output",
            "call_id": None,
            "output": "oops",
        }
    ]


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


def test_extract_tool_calls_and_to_llm_response(openai_provider: OpenAI) -> None:
    image_payload = b64encode(b"image-bytes").decode()
    response = SimpleNamespace(
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
        usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
        status="completed",
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.text == "hello"
    assert llm_response.tool_calls[0].name == "lookup"
    assert llm_response.usage is not None
    assert llm_response.usage.input_tokens == 1
    image_segment = next(seg for seg in llm_response.segments if seg.type == "image")
    assert image_segment.media is not None
    assert image_segment.media.data is not None


def test_extract_tool_calls_falls_back_to_item_id(
    openai_provider: OpenAI,
) -> None:
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        name="lookup",
        arguments="{}",
        call_id="temp",
    ).model_copy(update={"call_id": None, "id": "tool-1"})

    response = SimpleNamespace(
        id="resp-2",
        model="gpt-4o",
        output_text="",
        output=[tool_call],
        usage=None,
        status="in_progress",
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.tool_calls[0].call_id == "tool-1"


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
    text_chunk = openai_provider._map_stream_event(
        text_event, model_name="gpt-4o"
    )
    assert text_chunk is not None
    assert text_chunk.chunk.text_delta == "Hi"

    tool_event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"value":1}',
        item_id="item-2",
        output_index=0,
        sequence_number=2,
        type="response.function_call_arguments.delta",
    )
    tool_chunk = openai_provider._map_stream_event(tool_event, model_name="gpt-4o")
    assert tool_chunk is not None
    assert (
        tool_chunk.chunk.tool_call_delta.arguments_delta == '{"value":1}'
    )

    error_event = ResponseErrorEvent(
        code=None,
        message="boom",
        param=None,
        sequence_number=3,
        type="error",
    )
    error_chunk = openai_provider._map_stream_event(error_event, model_name="gpt-4o")
    assert error_chunk is not None
    assert error_chunk.chunk.error == "boom"

    fallback_event = SimpleNamespace(
        type="response.output_text.delta",
        delta="fallback",
        item_id=None,
    )
    fallback_chunk = openai_provider._map_stream_event(
        fallback_event, model_name="gpt-4o"
    )
    assert fallback_chunk is not None
    assert fallback_chunk.chunk.text_delta == "fallback"

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
    assert image_chunk.chunk.media is not None
    assert image_chunk.segments[0].type == "image"
