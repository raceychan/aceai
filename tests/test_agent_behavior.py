import pytest
from opentelemetry.context import Context

from aceai.agent.base import AgentBase, ToolExecutionFailure
from aceai.errors import AceAIRuntimeError
from aceai.agent.events import (
    AgentEvent,
    LLMMediaEvent,
    LLMOutputDeltaEvent,
    LLMStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    ToolFailedEvent,
    ToolStartedEvent,
)
from aceai.llm import LLMResponse
from aceai.llm.models import (
    LLMGeneratedMedia,
    LLMSegment,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallDelta,
)


class StubExecutor:
    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.tool_specs: list[object] = []
        self._results = results or {}
        self.calls: list[LLMToolCall] = []

    def select_tools(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ) -> list[object]:
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        return self.tool_specs

    async def execute_tool(self, tool_call: LLMToolCall, *, trace_ctx: Context) -> str:
        self.calls.append(tool_call)
        return self._results[tool_call.name]


class StubLLMService:
    def __init__(self, streams: list[list[LLMStreamEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.calls: list[dict] = []

    async def stream(self, *, trace_ctx: Context | None = None, **request):
        if not self._streams:
            raise AssertionError("StubLLMService has no remaining stream fixtures")
        self.calls.append(request)
        events = self._streams.pop(0)
        for event in events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("AgentBase should not call complete() in streaming mode")


class RaisingExecutor(StubExecutor):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def execute_tool(self, tool_call: LLMToolCall, *, trace_ctx: Context) -> str:
        raise self._error


class RaisingStreamLLMService:
    def __init__(self, events: list[LLMStreamEvent], error: Exception) -> None:
        self._events = list(events)
        self._error = error
        self.calls: list[dict] = []

    async def stream(self, *, trace_ctx: Context | None = None, **request):
        self.calls.append(request)
        for event in self._events:
            yield event
        raise self._error

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("AgentBase should not call complete() in streaming mode")


class SimpleLLMService:
    def __init__(self, events: list[LLMStreamEvent]) -> None:
        self._events = list(events)
        self.calls: list[dict] = []

    async def stream(self, *, trace_ctx: Context | None = None, **request):
        self.calls.append(request)
        for event in self._events:
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
    assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "   "


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
async def test_agent_surfaces_media_events() -> None:
    media = LLMGeneratedMedia(type="image", mime_type="image/png", data=b"\x89")
    media_segment = LLMSegment(type="image", content="", media=media)
    streams = [
        [
            LLMStreamEvent(
                event_type="response.media",
                segments=[media_segment],
            ),
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="done", segments=[media_segment]),
            ),
        ]
    ]
    llm_service = StubLLMService(streams)
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Need an image")

    media_events = [event for event in events if isinstance(event, LLMMediaEvent)]
    assert media_events
    assert media_events[0].segments[0].media is media
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


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
async def test_agent_does_not_complete_mid_step_after_tool_calls() -> None:
    lookup_call = LLMToolCall(name="lookup", arguments="{}", call_id="tool-1")
    streams = [
        make_stream(
            response=LLMResponse(text="use lookup", tool_calls=[lookup_call]),
            deltas=["use lookup"],
        ),
        make_stream(
            response=LLMResponse(text="done"),
            deltas=["done"],
        ),
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=StubExecutor({"lookup": '{"value":42}'}),
        max_steps=2,
    )

    events = await collect_events(agent, "Finish up")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


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
    with pytest.raises(AceAIRuntimeError, match="exceeded maximum steps"):
        async for evt in agent.run("try again"):
            events.append(evt)

    assert isinstance(events[-1], RunFailedEvent)
    assert events[-1].error == "Agent exceeded maximum steps: 1 without answering"


@pytest.mark.anyio
async def test_agent_can_return_structured_response() -> None:
    response = LLMResponse(text="final")
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([make_stream(response=response, deltas=["final"])]),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Finish up")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.final_answer == "final"


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
async def test_agent_emits_text_deltas_without_buffering_reasoning_log() -> None:
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
    )

    events = await collect_events(agent, "Question?")
    deltas = [event for event in events if isinstance(event, LLMOutputDeltaEvent)]

    assert len(deltas) == 2
    assert deltas[0].text_delta == "foo"
    assert deltas[1].text_delta == "bar"
    final_event = events[-1]
    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.step.reasoning_log == ""
    assert final_event.step.reasoning_log_truncated is False


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
async def test_stream_error_bubbles_without_run_failed_event() -> None:
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

    assert isinstance(events[-1], LLMStartedEvent)
    assert not any(isinstance(evt, RunFailedEvent) for evt in events)


@pytest.mark.anyio
async def test_agent_skips_function_call_argument_deltas() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.function_call_arguments.delta",
            tool_call_delta=LLMToolCallDelta(id="tool-1", arguments_delta='{"a":'),
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="hello ",
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="world",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="hello world"),
        ),
    ]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=SimpleLLMService(stream),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "hello world"


@pytest.mark.anyio
async def test_agent_uses_default_error_message_when_stream_error_missing_text() -> (
    None
):
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.error",
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events: list[AgentEvent] = []
    with pytest.raises(AceAIRuntimeError, match="LLM streaming error"):
        async for evt in agent.run("Boom?"):
            events.append(evt)

    assert isinstance(events[-1], LLMStartedEvent)
    assert not any(isinstance(evt, RunFailedEvent) for evt in events)


@pytest.mark.anyio
async def test_agent_errors_when_completion_event_lacks_response() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.completed",
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events: list[AgentEvent] = []
    with pytest.raises(
        AceAIRuntimeError, match="LLM stream completed without a response payload"
    ):
        async for evt in agent.run("Question?"):
            events.append(evt)

    assert isinstance(events[-1], LLMStartedEvent)
    assert not any(isinstance(evt, RunFailedEvent) for evt in events)


@pytest.mark.anyio
async def test_reasoning_segments_do_not_populate_reasoning_log() -> None:
    response = LLMResponse(
        text="done",
        segments=[
            LLMSegment(type="reasoning", content="first"),
            LLMSegment(type="reasoning", content="second"),
        ],
    )
    stream = make_stream(response=response, deltas=[])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=SimpleLLMService(stream),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.step.reasoning_log == ""
    assert final_event.step.reasoning_log_truncated is False


@pytest.mark.anyio
async def test_agent_tool_execution_failure_emits_tool_failed_event() -> None:
    call = LLMToolCall(name="calc", arguments="{}", call_id="calc-1")
    stream = [
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="use calc", tool_calls=[call]),
        )
    ]
    executor = RaisingExecutor(ValueError("no calc"))
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=executor,
    )

    events: list[AgentEvent] = []
    with pytest.raises(ToolExecutionFailure) as exc_info:
        async for evt in agent.run("Question?"):
            events.append(evt)

    tool_failed = [evt for evt in events if isinstance(evt, ToolFailedEvent)]
    assert tool_failed and tool_failed[0].error == "no calc"
    assert isinstance(events[-1], RunFailedEvent)
    assert exc_info.value.original_error.__class__ is ValueError


@pytest.mark.anyio
async def test_agent_executes_all_tool_calls_in_step() -> None:
    first_call = LLMToolCall(name="lookup", arguments="{}", call_id="lookup-1")
    second_call = LLMToolCall(name="calc", arguments="{}", call_id="calc-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="tool outputs",
                    tool_calls=[first_call, second_call],
                ),
            )
        ],
        make_stream(
            response=LLMResponse(text="done"),
            deltas=["done"],
        ),
    ]
    executor = StubExecutor({"lookup": '{"value":42}', "calc": "3"})
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=executor,
        max_steps=2,
    )

    events = await collect_events(agent, "Finish?")

    assert executor.calls == [first_call, second_call]
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


class ShortCircuitAgent(AgentBase):
    async def _run_step(
        self,
        *,
        event_builder,
        trace_ctx,
        **request_meta,
    ):
        yield event_builder.llm_started()
        raise RuntimeError("preflight failure")


@pytest.mark.anyio
async def test_agent_run_rethrows_when_no_steps_recorded() -> None:
    agent = ShortCircuitAgent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),
        executor=StubExecutor(),
        max_steps=1,
    )

    with pytest.raises(RuntimeError, match="preflight failure"):
        async for _ in agent.run("Question?"):
            pass
