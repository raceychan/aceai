import pytest

from aceai.agent import AgentBase
from aceai.events import (
    AgentEvent,
    LLMOutputDeltaEvent,
    LLMStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    ToolStartedEvent,
)
from aceai.errors import AceAIRuntimeError
from aceai.llm import LLMResponse
from aceai.llm.models import LLMStreamEvent, LLMToolCall


class StubExecutor:
    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.tool_schemas: list[dict] = []
        self._results = results or {}
        self.calls: list[LLMToolCall] = []

    async def execute_tool(self, tool_call: LLMToolCall) -> str:
        self.calls.append(tool_call)
        return self._results[tool_call.name]


class StubLLMService:
    def __init__(self, streams: list[list[LLMStreamEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.calls: list[dict] = []

    async def stream(self, **request):
        if not self._streams:
            raise AssertionError("StubLLMService has no remaining stream fixtures")
        self.calls.append(request)
        events = self._streams.pop(0)
        for event in events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("AgentBase should not call complete() in streaming mode")


def make_stream(
    *,
    response: LLMResponse,
    deltas: list[str] | None = None,
) -> list[LLMStreamEvent]:
    events: list[LLMStreamEvent] = []
    for chunk in deltas or []:
        events.append(
            LLMStreamEvent(
                event_type="response.output_text.delta",
                text_delta=chunk,
            )
        )
    events.append(
        LLMStreamEvent(
            event_type="response.completed",
            response=response,
        )
    )
    return events


async def collect_events(agent: AgentBase, question: str) -> list[AgentEvent]:
    return [event async for event in agent.run(question)]


@pytest.mark.anyio
async def test_agent_allows_whitespace_question_and_calls_llm() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="  answer  "),
            deltas=["  answer  "],
        )
    ]
    llm_service = StubLLMService(streams)
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "   ")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.final_answer == "  answer  "
    assert len(llm_service.calls) == 1
    assert llm_service.calls[0]["messages"][-1].content == "   "


@pytest.mark.anyio
async def test_agent_returns_llm_text_without_tool_calls() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="  hello world  "),
            deltas=["  hello ", "world  "],
        )
    ]
    llm_service = StubLLMService(streams)
    executor = StubExecutor()
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
    )

    events = await collect_events(agent, "What time is it?")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "  hello world  "
    assert len(llm_service.calls) == 1


@pytest.mark.anyio
async def test_agent_handles_tool_call_and_continues_conversation() -> None:
    lookup_call = LLMToolCall(name="lookup", arguments="{}", call_id="tool-1")
    streams = [
        make_stream(
            response=LLMResponse(text="use lookup", tool_calls=[lookup_call]),
            deltas=["use ", "lookup"],
        ),
        make_stream(
            response=LLMResponse(text="answer after tool"),
            deltas=["answer after tool"],
        ),
    ]
    llm_service = StubLLMService(streams)
    executor = StubExecutor({"lookup": '{"value":42}'})
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=3,
    )

    events = await collect_events(agent, "Need data")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "answer after tool"
    assert len(llm_service.calls) == 2
    assert executor.calls[0].name == "lookup"
    tool_events = [event for event in events if isinstance(event, ToolStartedEvent)]
    assert tool_events
    assert tool_events[0].tool_name == "lookup"


@pytest.mark.anyio
async def test_agent_returns_final_answer_from_tool() -> None:
    final_call = LLMToolCall(name="final_answer", arguments="{}", call_id="final-1")
    streams = [
        make_stream(
            response=LLMResponse(text="calling tool", tool_calls=[final_call]),
            deltas=["calling tool"],
        )
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=StubExecutor({"final_answer": '"Done"'}),
    )

    events = await collect_events(agent, "Finish up")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.final_answer == '"Done"'
    assert final_event.step.tool_results
    assert final_event.step.tool_results[0].output == '"Done"'


@pytest.mark.anyio
async def test_agent_raises_after_exceeding_turn_limit() -> None:
    streams = [make_stream(response=LLMResponse(text=""), deltas=[])]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=StubExecutor(),
        max_steps=1,
    )

    events: list[AgentEvent] = []
    with pytest.raises(AceAIRuntimeError, match="exceeded maximum reasoning turns"):
        async for evt in agent.run("try again"):
            events.append(evt)

    assert isinstance(events[-1], RunFailedEvent)
    assert (
        events[-1].error == "Agent exceeded maximum reasoning turns without answering"
    )


@pytest.mark.anyio
async def test_agent_can_return_structured_response() -> None:
    final_call = LLMToolCall(name="final_answer", arguments="{}", call_id="final-1")
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                make_stream(
                    response=LLMResponse(text="calling tool", tool_calls=[final_call]),
                    deltas=["calling tool"],
                )
            ]
        ),
        executor=StubExecutor({"final_answer": '"Done"'}),
    )

    events = await collect_events(agent, "Finish up")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.final_answer == '"Done"'
    assert final_event.step.llm_response.text == "calling tool"
    assert final_event.step.tool_results[0].output == '"Done"'


@pytest.mark.anyio
async def test_agent_stream_emits_run_completed_event() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                make_stream(
                    response=LLMResponse(text="  hello  "),
                    deltas=["  hello  "],
                )
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")

    assert isinstance(events[0], LLMStartedEvent)
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "  hello  "


@pytest.mark.anyio
async def test_agent_emits_text_deltas_and_populates_reasoning_log() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="foo",
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="bar",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="foobar"),
        ),
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
        delta_chunk_size=1,
    )

    events = await collect_events(agent, "Question?")
    deltas = [event for event in events if isinstance(event, LLMOutputDeltaEvent)]

    assert len(deltas) == 2
    assert deltas[0].text_delta == "foo"
    assert deltas[1].text_delta == "bar"
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].step.reasoning_log == "foobar"


@pytest.mark.anyio
async def test_reasoning_log_is_empty_when_no_deltas() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                make_stream(
                    response=LLMResponse(text="final"),
                    deltas=[],
                )
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].step.reasoning_log == ""
    assert events[-1].final_answer == "final"


@pytest.mark.anyio
async def test_delta_chunker_flushes_when_threshold_exceeded() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="ab",
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="cd",
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="ef",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="abcdef"),
        ),
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
        delta_chunk_size=4,
    )

    events = await collect_events(agent, "Question?")
    deltas = [event for event in events if isinstance(event, LLMOutputDeltaEvent)]

    assert len(deltas) == 2
    assert deltas[0].text_delta == "abcd"
    assert deltas[1].text_delta == "ef"


@pytest.mark.anyio
async def test_delta_chunker_flushes_on_completion_when_below_threshold() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="hi",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="hi"),
        ),
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
        delta_chunk_size=256,
    )

    events = await collect_events(agent, "Question?")
    deltas = [event for event in events if isinstance(event, LLMOutputDeltaEvent)]

    assert len(deltas) == 1
    assert deltas[0].text_delta == "hi"


@pytest.mark.anyio
async def test_reasoning_log_ring_buffer_truncates_old_content() -> None:
    stream = [
        LLMStreamEvent(event_type="response.output_text.delta", text_delta=str(i))
        for i in range(10)
    ] + [
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="0123456789"),
        )
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
        delta_chunk_size=1,
        reasoning_log_max_chars=5,
    )

    events = await collect_events(agent, "Question?")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.step.reasoning_log == "56789"
    assert final_event.step.reasoning_log_truncated is True


@pytest.mark.anyio
async def test_reasoning_log_disabled_when_max_zero() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="partial",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="partial"),
        ),
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
        delta_chunk_size=1,
        reasoning_log_max_chars=0,
    )

    events = await collect_events(agent, "Question?")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.step.reasoning_log == ""
    assert final_event.step.reasoning_log_truncated is False


@pytest.mark.anyio
async def test_stream_error_triggers_run_failed_event() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.error",
                        error="provider exploded",
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events: list[AgentEvent] = []
    with pytest.raises(AceAIRuntimeError, match="provider exploded"):
        async for evt in agent.run("boom?"):
            events.append(evt)

    assert isinstance(events[-1], RunFailedEvent)
