import io
from types import SimpleNamespace

import pytest
from openai.types.responses.response_error_event import ResponseErrorEvent
from openai.types.responses.response_function_call_arguments_delta_event import (
    ResponseFunctionCallArgumentsDeltaEvent,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from aceai.errors import AceAIRuntimeError, AceAIValidationError
from aceai.llm import LLMMessage
from aceai.llm.models import (
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
    LLMResponseFormat,
)
from aceai.llm.openai import OpenAI
from aceai.tools.interface import ToolSpec


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
            self.response_payload = SimpleNamespace(
                id="fake-response",
                model="fake-model",
                output_text="",
                output=[],
                usage=None,
                status="completed",
            )
            self.stream_events: list = []
            self.final_stream_response = SimpleNamespace(
                id="fake-stream-response",
                model="fake-stream-model",
                output_text="",
                output=[],
                usage=None,
                status="completed",
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
        default_model="gpt-4o",
        default_stream_model="gpt-4o-mini",
    )


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
        "messages": [LLMMessage(role="system", content="You are helpful.")],
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
        "messages": [LLMMessage(role="system", content="start")],
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
    messages = [
        LLMMessage(role="system", content="Start"),
        LLMToolCallMessage(content="", tool_calls=[call]),
        LLMToolUseMessage(content="result", call_id="call-1"),
    ]

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted[0] == {"role": "system", "content": "Start"}
    assert formatted[1]["type"] == "function_call"
    assert formatted[1]["name"] == "lookup"
    assert formatted[2]["type"] == "function_call_output"
    assert formatted[2]["output"] == "result"


def test_format_messages_allows_tool_output_without_call_id(
    openai_provider: OpenAI,
) -> None:
    messages = [LLMToolUseMessage(content="oops", call_id=None)]

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted == [
        {
            "type": "function_call_output",
            "call_id": None,
            "output": "oops",
        }
    ]


def test_format_messages_includes_tool_call_message_content(
    openai_provider: OpenAI,
) -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    messages = [LLMToolCallMessage(content="prep", tool_calls=[call])]

    formatted = openai_provider._format_messages_for_responses(messages)

    assert formatted[0] == {"role": "assistant", "content": "prep"}
    assert formatted[1]["name"] == "lookup"


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
        status="completed",
    )

    llm_response = openai_provider._to_llm_response(response)

    assert llm_response.text == "hello"
    assert llm_response.tool_calls[0].name == "lookup"
    assert llm_response.usage is not None
    assert llm_response.usage.input_tokens == 1


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


def test_extract_tool_calls_requires_identifier(openai_provider: OpenAI) -> None:
    tool_call = ResponseFunctionToolCall(
        type="function_call",
        name="lookup",
        arguments="{}",
        call_id="temp",
    ).model_copy(update={"call_id": None, "id": None})
    response = SimpleNamespace(
        id="resp-3",
        model="gpt-4o",
        output_text="",
        output=[tool_call],
        usage=None,
        status="errored",
    )

    with pytest.raises(AceAIRuntimeError, match="call identifier"):
        openai_provider._extract_tool_calls(response)


def test_safe_model_dump_handles_none_and_errors(openai_provider: OpenAI) -> None:
    class BadModel:
        def model_dump(self):
            raise AceAIRuntimeError("fail")

    assert openai_provider._safe_model_dump(None) == {}
    assert openai_provider._safe_model_dump(BadModel()) == {}


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

    fallback_event = SimpleNamespace(
        type="response.output_text.delta",
        delta="fallback",
        item_id=None,
    )
    fallback_chunk = openai_provider._map_stream_event(
        fallback_event, model_name="gpt-4o"
    )
    assert fallback_chunk is not None
    assert fallback_chunk.text_delta == "fallback"


def test_map_stream_event_handles_fallback_tool_and_error(openai_provider: OpenAI) -> None:
    tool_event = SimpleNamespace(
        type="response.function_call_arguments.delta",
        delta="{}",
        item_id="tool-3",
    )
    tool_chunk = openai_provider._map_stream_event(tool_event, model_name="gpt-4o")
    assert tool_chunk is not None
    assert tool_chunk.tool_call_delta.id == "tool-3"

    error_event = SimpleNamespace(
        type="response.error",
        delta=None,
        item_id=None,
        message="nope",
    )
    error_chunk = openai_provider._map_stream_event(error_event, model_name="gpt-4o")
    assert error_chunk is not None
    assert error_chunk.error == "nope"

    unknown_event = SimpleNamespace(type="other", delta=None, item_id=None)
    assert openai_provider._map_stream_event(unknown_event, model_name="gpt-4o") is None


@pytest.mark.anyio
async def test_openai_properties_and_stt(fake_openai_client) -> None:
    fake_openai_client.transcription_text = "heard"
    provider = OpenAI(
        client=fake_openai_client,
        default_model="gpt-4o",
        default_stream_model="gpt-4o-mini",
    )

    transcript = await provider.stt(
        "audio.wav",
        io.BytesIO(b"wave"),
        model="whisper-large",
    )

    assert transcript == "heard"
    assert fake_openai_client.transcription_calls[0]["model"] == "whisper-large"
    assert provider.default_model == "gpt-4o"
    assert provider.default_stream_model == "gpt-4o-mini"


@pytest.mark.anyio
async def test_openai_complete_invokes_client(fake_openai_client) -> None:
    fake_openai_client.responses.response_payload = SimpleNamespace(
        id="resp-final",
        model="gpt-4o",
        output_text="done",
        output=[],
        usage=None,
        status="completed",
    )
    provider = OpenAI(
        client=fake_openai_client,
        default_model="gpt-4o",
        default_stream_model="gpt-4o-mini",
    )

    response = await provider.complete(
        {"messages": [LLMMessage(role="system", content="hello")]}
    )

    assert response.text == "done"
    assert fake_openai_client.responses.create_calls


@pytest.mark.anyio
async def test_openai_stream_yields_events_and_completion(fake_openai_client) -> None:
    fake_openai_client.responses.stream_events = [
        ResponseTextDeltaEvent(
            content_index=0,
            delta="Hi",
            item_id="item-1",
            logprobs=[],
            output_index=0,
            sequence_number=1,
            type="response.output_text.delta",
        )
    ]
    fake_openai_client.responses.final_stream_response = SimpleNamespace(
        id="resp-stream",
        model="gpt-4o",
        output_text="final",
        output=[],
        usage=None,
        status="completed",
    )
    provider = OpenAI(
        client=fake_openai_client,
        default_model="gpt-4o",
        default_stream_model="gpt-4o-mini",
    )

    events = []
    async for event in provider.stream(
        {"messages": [LLMMessage(role="system", content="stream")]}
    ):
        events.append(event.event_type)

    assert events[-1] == "response.completed"
    assert "response.output_text.delta" in events
    assert fake_openai_client.responses.stream_calls
