from typing import AsyncGenerator

import pytest
from ididi import Graph
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aceai.agent import AgentBase
from aceai.events import RunCompletedEvent
from aceai.executor import ToolExecutor
from aceai.llm.models import (
    LLMMessage,
    LLMProviderBase,
    LLMProviderModality,
    LLMResponse,
    LLMStreamEvent,
    LLMToolCall,
)
from aceai.llm.service import LLMService
from aceai.tools import tool
from aceai.tools._tool_sig import Annotated, spec


class StreamingProvider(LLMProviderBase):
    def __init__(self, events: list[LLMStreamEvent]) -> None:
        self._events = list(events)

    async def complete(self, request: dict, *, trace_ctx=None) -> LLMResponse:  # pragma: no cover
        raise AssertionError("StreamingProvider.complete should not be used in this test")

    def stream(self, request: dict, *, trace_ctx=None):
        async def iterator():
            for event in self._events:
                yield event

        return iterator()

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def default_stream_model(self) -> str:
        return "gpt-4o-mini"

    @property
    def modality(self) -> LLMProviderModality:
        return LLMProviderModality()

    async def stt(
        self, filename, file, *, model: str, prompt: str | None = None
    ) -> str:  # pragma: no cover
        raise AssertionError("StreamingProvider.stt should not be used in this test")


def make_stream(*, response: LLMResponse) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="calling tools",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=response,
        ),
    ]


@pytest.fixture
async def graph() -> AsyncGenerator[Graph, None]:
    graph = Graph()
    try:
        yield graph
    finally:
        graph._workers.shutdown(wait=True)


@pytest.mark.anyio
async def test_agent_spans_are_parented_under_single_trace(graph: Graph) -> None:
    def lookup_order(order_id: Annotated[str, spec(description="Order identifier")]) -> str:
        return f"order:{order_id}"

    def get_sku_weight(sku: Annotated[str, spec(description="SKU identifier")]) -> str:
        return "2.5"

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent_tracer = tracer_provider.get_tracer("aceai.agent")
    llm_tracer = tracer_provider.get_tracer("aceai.llm")
    executor_tracer = tracer_provider.get_tracer("aceai.executor")

    tool_calls = [
        LLMToolCall(
            name="lookup_order",
            arguments='{"order_id":"ORD-1"}',
            call_id="call-lookup",
        ),
        LLMToolCall(
            name="get_sku_weight",
            arguments='{"sku":"SKU-1"}',
            call_id="call-weight",
        ),
        LLMToolCall(
            name="final_answer",
            arguments='{"answer":"done"}',
            call_id="call-final",
        ),
    ]
    response = LLMResponse(text="use tools", tool_calls=tool_calls)
    provider = StreamingProvider(make_stream(response=response))
    service = LLMService([provider], timeout_seconds=1.0, tracer=llm_tracer)
    executor = ToolExecutor(
        graph=graph,
        tools=[tool(lookup_order), tool(get_sku_weight)],
        tracer=executor_tracer,
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=service,
        executor=executor,
        max_steps=1,
        tracer=agent_tracer,
    )

    events = [event async for event in agent.run("question")]
    assert isinstance(events[-1], RunCompletedEvent)

    spans = list(exporter.get_finished_spans())
    spans_by_name = {span.name: span for span in spans}

    run_span = spans_by_name["agent.run"]
    step_span = spans_by_name["agent.step"]
    llm_span = spans_by_name["llm.stream"]
    tool_lookup_span = spans_by_name["tool.lookup_order"]
    tool_weight_span = spans_by_name["tool.get_sku_weight"]
    tool_final_span = spans_by_name["tool.final_answer"]

    assert run_span.attributes["langfuse.trace.input"] == "question"
    assert run_span.attributes["langfuse.trace.output"] == '"done"'

    assert step_span.parent is not None
    assert step_span.parent.span_id == run_span.context.span_id
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == step_span.context.span_id
    assert tool_lookup_span.parent is not None
    assert tool_lookup_span.parent.span_id == step_span.context.span_id
    assert tool_weight_span.parent is not None
    assert tool_weight_span.parent.span_id == step_span.context.span_id
    assert tool_final_span.parent is not None
    assert tool_final_span.parent.span_id == step_span.context.span_id

    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1


@pytest.mark.anyio
async def test_agent_run_can_be_closed_early_without_context_detach_errors(
    graph: Graph,
) -> None:
    response = LLMResponse(text="ok")
    provider = StreamingProvider(make_stream(response=response))
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer("aceai.agent")
    service = LLMService([provider], timeout_seconds=1.0, tracer=tracer_provider.get_tracer("aceai.llm"))
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=service,
        executor=ToolExecutor(graph=graph, tools=[], tracer=tracer_provider.get_tracer("aceai.executor")),
        max_steps=1,
        tracer=tracer,
    )

    agen = agent.run("question")
    await anext(agen)
    await agen.aclose()
