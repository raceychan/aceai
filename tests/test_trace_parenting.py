from typing import AsyncGenerator

import pytest
from ididi import Graph
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aceai.agent.base import AgentBase
from aceai.agent.events import RunCompletedEvent
from aceai.agent.executor import ToolExecutor
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
    def __init__(self, streams: list[list[LLMStreamEvent]], *, tracer) -> None:
        self._streams = [list(stream) for stream in streams]
        self._tracer = tracer

    async def complete(self, request: dict, *, trace_ctx=None) -> LLMResponse:  # pragma: no cover
        raise AssertionError("StreamingProvider.complete should not be used in this test")

    def stream(self, request: dict, *, trace_ctx=None):
        if not self._streams:
            raise AssertionError("StreamingProvider has no remaining stream fixtures")
        events = self._streams.pop(0)

        async def iterator():
            tool_names = [tool.name for tool in request.get("tools", [])]
            span = self._tracer.start_span(
                "llm.provider.stream",
                kind=SpanKind.CLIENT,
                context=trace_ctx,
                attributes={
                    "llm.tool_count": len(tool_names),
                    "llm.tool_names": tool_names,
                },
            )
            try:
                for event in events:
                    yield event
            finally:
                span.end()

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
        self, filename, file, *, model: str, prompt: str | None = None, trace_ctx=None
    ) -> str:  # pragma: no cover
        raise AssertionError("StreamingProvider.stt should not be used in this test")


def make_stream(*, delta: str, response: LLMResponse) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta=delta,
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
    provider_tracer = tracer_provider.get_tracer("aceai.llm.provider")
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
    ]
    response_step1 = LLMResponse(text="use tools", tool_calls=tool_calls)
    response_step2 = LLMResponse(text="done")
    provider = StreamingProvider(
        [
            make_stream(delta="calling tools", response=response_step1),
            make_stream(delta="done", response=response_step2),
        ],
        tracer=provider_tracer,
    )
    service = LLMService([provider], timeout_seconds=1.0)
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
        max_steps=2,
        tracer=agent_tracer,
    )

    events = [event async for event in agent.run("question")]
    assert isinstance(events[-1], RunCompletedEvent)

    spans = list(exporter.get_finished_spans())
    run_span = next(span for span in spans if span.name == "agent.run")
    step_spans = [span for span in spans if span.name == "agent.step"]
    llm_spans = [span for span in spans if span.name == "llm.provider.stream"]
    tool_lookup_span = next(span for span in spans if span.name == "tool.lookup_order")
    tool_weight_span = next(span for span in spans if span.name == "tool.get_sku_weight")

    assert run_span.attributes["langfuse.trace.input"] == "question"
    assert run_span.attributes["langfuse.trace.output"] == "done"

    assert len(step_spans) == 2
    for step_span in step_spans:
        assert step_span.parent is not None
        assert step_span.parent.span_id == run_span.context.span_id

    assert len(llm_spans) == 2
    step_ids = {step_span.context.span_id for step_span in step_spans}
    for llm_span in llm_spans:
        assert llm_span.parent is not None
        assert llm_span.parent.span_id in step_ids
        assert set(llm_span.attributes["llm.tool_names"]) == {
            "lookup_order",
            "get_sku_weight",
        }

    first_step_span = next(
        step_span
        for step_span in step_spans
        if tool_lookup_span.parent is not None
        and step_span.context.span_id == tool_lookup_span.parent.span_id
    )
    assert tool_lookup_span.parent is not None
    assert tool_lookup_span.parent.span_id == first_step_span.context.span_id
    assert tool_weight_span.parent is not None
    assert tool_weight_span.parent.span_id == first_step_span.context.span_id

    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1


@pytest.mark.anyio
async def test_agent_run_can_be_closed_early_without_context_detach_errors(
    graph: Graph,
) -> None:
    response = LLMResponse(text="ok")
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider = StreamingProvider(
        [make_stream(delta="ok", response=response)],
        tracer=tracer_provider.get_tracer("aceai.llm.provider"),
    )
    tracer = tracer_provider.get_tracer("aceai.agent")
    service = LLMService([provider], timeout_seconds=1.0)
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=service,
        executor=ToolExecutor(
            graph=graph,
            tools=[],
            tracer=tracer_provider.get_tracer("aceai.executor"),
        ),
        max_steps=1,
        tracer=tracer,
    )

    agen = agent.run("question")
    await anext(agen)
    await agen.aclose()
