import pytest

from aceai.agent import AgentBase
from aceai.events import (
    AgentEvent,
    LLMStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    ToolStartedEvent,
)
from aceai.errors import AceAIRuntimeError
from aceai.llm import LLMResponse
from aceai.llm.models import LLMToolCall


class StubExecutor:
    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.tool_schemas: list[dict] = []
        self._results = results or {}
        self.calls: list[LLMToolCall] = []

    async def execute_tool(self, tool_call: LLMToolCall) -> str:
        self.calls.append(tool_call)
        return self._results[tool_call.name]


class StubLLMService:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, **request) -> LLMResponse:
        self.calls.append(request)
        return self._responses.pop(0)


async def collect_events(agent: AgentBase, question: str) -> list[AgentEvent]:
    return [event async for event in agent.run(question)]


@pytest.mark.anyio
async def test_agent_allows_whitespace_question_and_calls_llm() -> None:
    responses = [LLMResponse(text="  answer  ")]
    llm_service = StubLLMService(responses)
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
    responses = [LLMResponse(text="  hello world  ")]
    llm_service = StubLLMService(responses)
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
    responses = [
        LLMResponse(text="use lookup", tool_calls=[lookup_call]),
        LLMResponse(text="answer after tool"),
    ]
    llm_service = StubLLMService(responses)
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
    responses = [LLMResponse(text="calling tool", tool_calls=[final_call])]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(responses),
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
    responses = [LLMResponse(text="")]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(responses),
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
            [LLMResponse(text="calling tool", tool_calls=[final_call])]
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
        llm_service=StubLLMService([LLMResponse(text="  hello  ")]),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")

    assert isinstance(events[0], LLMStartedEvent)
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "  hello  "
