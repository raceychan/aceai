from types import SimpleNamespace

import pytest

from aceai.llm import LLMMessage
from aceai.llm.interface import LLMToolCall, ResponseFormat
from aceai.llm.openai import OpenAI
from aceai.tools.interface import ToolSpec
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent


@pytest.fixture
def openai_provider(monkeypatch):
    class DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(create=lambda **_: None)
            )
            self.responses = SimpleNamespace()

    monkeypatch.setattr("aceai.llm.openai.openai.AsyncOpenAI", DummyAsyncOpenAI)
    return OpenAI(api_key="key", default_model="gpt-4o", default_stream_model="gpt-4o-mini")


def test_build_base_response_kwargs_maps_request_fields(openai_provider: OpenAI) -> None:
    tool_spec: ToolSpec = {
        "type": "function",
        "name": "echo",
        "description": "Echo",
        "parameters": {"type": "object", "properties": {}},
    }
    request = {
        "messages": [LLMMessage(role="system", content="You are helpful.")],
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "stop": ["END"],
        "response_format": ResponseFormat(type="json_object"),
        "tools": [tool_spec],
        "tool_choice": "auto",
        "metadata": {},
    }

    with pytest.warns(UserWarning):
        params = openai_provider._build_base_response_kwargs(request, default_model="gpt-4o")

    assert params["model"] == "gpt-4o"
    assert params["max_output_tokens"] == 256
    assert params["temperature"] == 0.3
    assert params["top_p"] == 0.9
    assert params["text"] == {"format": {"type": "json_object"}}
    assert params["tools"][0]["name"] == "echo"
    assert params["tool_choice"] == "auto"


def test_build_base_response_kwargs_skips_temperature_for_gpt5(openai_provider: OpenAI) -> None:
    request = {
        "messages": [LLMMessage(role="system", content="start")],
        "temperature": 0.2,
        "metadata": {"model": "gpt-5o-mini"},
    }

    params = openai_provider._build_base_response_kwargs(request, default_model="fallback")

    assert "temperature" not in params
    assert params["model"] == "gpt-5o-mini"


def test_format_messages_for_responses_includes_tool_outputs(openai_provider: OpenAI) -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    messages = [
        LLMMessage(role="system", content="Start"),
        LLMMessage(role="assistant", content="", tool_calls=[call]),
        LLMMessage(role="tool", content="result", tool_call_id="call-1"),
    ]

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted[0] == {"role": "system", "content": "Start"}
    assert formatted[1]["type"] == "function_call"
    assert formatted[1]["name"] == "lookup"
    assert formatted[2]["type"] == "function_call_output"
    assert formatted[2]["output"] == "result"


def test_build_text_config_variants(openai_provider: OpenAI) -> None:
    json_object = ResponseFormat(type="json_object")
    json_schema = ResponseFormat(type="json_schema", schema={"name": "Payload"})
    plain = ResponseFormat()

    assert openai_provider._build_text_config(json_object) == {"format": {"type": "json_object"}}
    assert openai_provider._build_text_config(json_schema) == {
        "format": {
            "type": "json_schema",
            "schema": {"name": "Payload"},
            "name": "response_schema",
        }
    }
    assert openai_provider._build_text_config(plain) is None


def test_extract_tool_calls_and_to_llm_response(openai_provider: OpenAI) -> None:
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
            )
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.text == "hello"
    assert llm_response.tool_calls[0].name == "lookup"
    assert llm_response.usage is not None
    assert llm_response.usage.input_tokens == 1


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
    text_chunk = openai_provider._map_stream_event(text_event)
    assert text_chunk is not None and text_chunk.text_delta == "Hi"

    tool_event = ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"value":1}',
        item_id="item-2",
        output_index=0,
        sequence_number=2,
        type="response.function_call_arguments.delta",
    )
    tool_chunk = openai_provider._map_stream_event(tool_event)
    assert tool_chunk is not None
    assert tool_chunk.tool_call_delta.arguments_delta == '{"value":1}'

    error_event = ResponseErrorEvent(
        code=None,
        message="boom",
        param=None,
        sequence_number=3,
        type="error",
    )
    error_chunk = openai_provider._map_stream_event(error_event)
    assert error_chunk is not None and error_chunk.error == "boom"

    fallback_event = SimpleNamespace(
        type="response.output_text.delta",
        delta="fallback",
        item_id=None,
    )
    fallback_chunk = openai_provider._map_stream_event(fallback_event)
    assert fallback_chunk is not None and fallback_chunk.text_delta == "fallback"
