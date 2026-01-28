import pytest
from ididi import Graph

from aceai.agent.base import AgentBase
from aceai.agent.executor import RunState, ToolExecutor
from aceai.llm import LLMResponse
from aceai.llm.models import LLMStreamEvent, LLMToolCall
from aceai.tools import tool
from aceai.tools._tool_sig import Annotated, spec
from aceai.tools.registry import ToolRegistry


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


def make_stream(*, response: LLMResponse) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(
            event_type="response.completed",
            response=response,
        )
    ]


@pytest.fixture
async def graph():
    g = Graph()
    try:
        yield g
    finally:
        g._workers.shutdown(wait=True)


@pytest.mark.anyio
async def test_registry_tools_injected_by_tag_into_agent_executor(graph: Graph) -> None:
    @tool(tags=["math"])
    def add(
        a: Annotated[int, spec(description="a")],
        b: Annotated[int, spec(description="b")],
    ) -> int:
        return a + b

    @tool(tags=["string"])
    def echo(message: Annotated[str, spec(description="message")]) -> str:
        return message

    registry = ToolRegistry(add, echo)
    executor = ToolExecutor(graph, registry.get_tools("math"))

    add_call = LLMToolCall(
        name="add",
        arguments='{"a":1,"b":2}',
        call_id="tool-1",
    )
    llm_service = StubLLMService(
        [
            make_stream(response=LLMResponse(text="use add", tool_calls=[add_call])),
            make_stream(response=LLMResponse(text="done")),
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=2,
    )

    answer = await agent.ask("Need math")
    assert answer == "done"

    assert len(llm_service.calls) == 2
    assert {spec.name for spec in llm_service.calls[0]["tools"]} == {"add"}

    tool_messages = [m for m in llm_service.calls[1]["messages"] if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "add"
    assert tool_messages[0].content[0]["data"] == "3"
