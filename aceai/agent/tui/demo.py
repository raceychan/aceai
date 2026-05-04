"""Static event fixtures for the read-only TUI prototype."""

from aceai.core.events import AgentEvent, AgentEventBuilder
from aceai.llm.models import LLMResponse, LLMSegment, LLMToolCall, LLMToolCallDelta
from aceai.core.models import AgentStep, ToolExecutionResult

from .events import TUIEvent


def static_demo_events() -> list[TUIEvent]:
    builder = AgentEventBuilder(step_index=0, step_id="demo-step-1")
    call = LLMToolCall(
        name="search_docs",
        arguments='{"query":"aceai tui"}',
        call_id="call-search-docs",
    )
    response = LLMResponse(
        text="I found the relevant TUI design notes.",
        tool_calls=[call],
        segments=[
            LLMSegment(
                type="reasoning",
                content="Need the design notes before summarizing implementation steps.",
            )
        ],
    )
    step = AgentStep(llm_response=response, tool_results=[])
    result = ToolExecutionResult(
        call=call,
        output='{"matches":["spec/tui.md","docs/tui.md"]}',
    )
    step.tool_results.append(result)

    agent_events: list[AgentEvent] = [
        builder.llm_started(),
        builder.llm_text_delta(text_delta="Checking the TUI plan "),
        builder.llm_tool_call_delta(
            tool_call_delta=LLMToolCallDelta(
                id=call.call_id,
                arguments_delta='{"query":"aceai tui"}',
            )
        ),
        builder.llm_reasoning(segment=response.segments[0]),
        builder.llm_completed(step=step),
        builder.tool_started(tool_call=call),
        builder.tool_completed(tool_call=call, tool_result=result),
        builder.step_completed(step=step),
        builder.run_completed(
            step=step,
            final_answer="Static TUI prototype is ready to inspect.",
        ),
    ]
    return [TUIEvent.from_agent_event(event) for event in agent_events]
