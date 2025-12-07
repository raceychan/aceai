import pytest

from aceai.agent import AgentBase
from aceai.errors import AceAIRuntimeError
from aceai.models import AgentResponse, AgentStepEvent
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

    response = await agent.handle("   ")

    assert response.final_output == "answer"
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

    response = await agent.handle("What time is it?")

    assert response.final_output == "hello world"
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
        max_turns=3,
    )

    response = await agent.handle("Need data")

    assert response.final_output == "answer after tool"
    assert len(llm_service.calls) == 2
    assert executor.calls[0].name == "lookup"


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

    response = await agent.handle("Finish up")

    assert response.final_output == '"Done"'


@pytest.mark.anyio
async def test_agent_raises_after_exceeding_turn_limit() -> None:
    responses = [LLMResponse(text="   ")]
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(responses),
        executor=StubExecutor(),
        max_turns=1,
    )

    with pytest.raises(AceAIRuntimeError, match="exceeded maximum reasoning turns"):
        await agent.handle("try again")


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

    response = await agent.handle("Finish up")
    assert isinstance(response, AgentResponse)
    assert response.final_output == '"Done"'
    assert len(response.turns) == 1
    turn = response.turns[0]
    assert turn.tool_results
    assert turn.tool_results[0].output == '"Done"'
    assert turn.llm_response.text == "calling tool"


@pytest.mark.anyio
async def test_agent_stream_emits_run_completed_event() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([LLMResponse(text="  hello  ")]),
        executor=StubExecutor(),
    )

    events: list[AgentStepEvent] = []
    async for evt in agent.stream("Question?"):
        events.append(evt)

    assert events[0].event_type == "agent.llm.started"
    assert events[-1].event_type == "agent.run.completed"
    assert events[-1].annotations["final_output"] == "hello"
