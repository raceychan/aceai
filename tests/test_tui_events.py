from aceai.core.events import (
    AgentEventBuilder,
    LLMCompletedEvent,
    LLMMediaEvent,
    LLMStartedEvent,
    RunCompletedEvent,
    RunSuspendedEvent,
)
from aceai.llm.models import (
    LLMGeneratedMedia,
    LLMReasoningSegmentMeta,
    LLMResponse,
    LLMUsage,
    LLMSegment,
    LLMToolCall,
    LLMToolCallDelta,
)
from aceai.core.models import AgentStep, ToolApprovalRequest, ToolExecutionResult
from agent_core.tui.events import TUIEvent


def test_adapt_text_delta_event() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    event = builder.llm_text_delta(text_delta="hello")

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "assistant_delta"
    assert tui_event.title == "assistant"
    assert tui_event.content == "hello"
    assert tui_event.raw_event is event


def test_adapt_step_started_event() -> None:
    event = LLMStartedEvent(step_index=0, step_id="step-1")

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "step_started"
    assert tui_event.title == "step started"


def test_adapt_llm_completed_event() -> None:
    step = AgentStep(llm_response=LLMResponse(text="intermediate"))
    event = LLMCompletedEvent(step_index=0, step_id="step-1", step=step)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "llm_completed"
    assert tui_event.content == "intermediate"


def test_adapt_llm_completed_event_hides_tool_call_content() -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    step = AgentStep(llm_response=LLMResponse(text="scratchpad", tool_calls=[call]))
    event = LLMCompletedEvent(step_index=0, step_id="step-1", step=step)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "llm_completed"
    assert tui_event.content == ""
    assert tui_event.tool_calls == [call]


def test_adapt_llm_completed_event_includes_usage_and_cost() -> None:
    usage = LLMUsage(
        input_tokens=10,
        cached_input_tokens=6,
        output_tokens=4,
        total_tokens=14,
    )
    step = AgentStep(
        llm_response=LLMResponse(text="done", model="gpt-5.5", usage=usage)
    )
    event = LLMCompletedEvent(step_index=0, step_id="step-1", step=step)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.usage is usage
    assert tui_event.cost is not None
    assert tui_event.cost.model == "gpt-5.5"
    assert round(tui_event.cost.total_cost_usd, 6) == 0.000143
    assert round(tui_event.cost.cached_input_cost_usd, 6) == 0.000003


def test_adapt_tool_call_delta_event() -> None:
    builder = AgentEventBuilder(step_index=1, step_id="step-2")
    delta = LLMToolCallDelta(id="call-1", arguments_delta='{"q":')
    event = builder.llm_tool_call_delta(tool_call_delta=delta)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "tool_call_delta"
    assert tui_event.content == '{"q":'
    assert tui_event.tool_call_delta is delta
    assert tui_event.tool_call_id == "call-1"


def test_adapt_reasoning_event() -> None:
    builder = AgentEventBuilder(step_index=2, step_id="step-3")
    segment = LLMSegment(type="reasoning", content="checked the facts")
    event = builder.llm_reasoning(segment=segment)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "reasoning_summary"
    assert tui_event.content == "checked the facts"
    assert tui_event.segment is segment


def test_adapt_streaming_reasoning_delta_event() -> None:
    builder = AgentEventBuilder(step_index=2, step_id="step-3")
    segment = LLMSegment(
        type="reasoning",
        content="checked",
        meta=LLMReasoningSegmentMeta(
            item_id="reasoning",
            kind="content",
            index=0,
            is_delta=True,
        ),
    )
    event = builder.llm_reasoning(segment=segment)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "thinking_delta"
    assert tui_event.title == "reasoning"


def test_adapt_tool_completed_event() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    result = ToolExecutionResult(call=call, output='{"ok":true}')
    event = builder.tool_completed(tool_call=call, tool_result=result)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "tool_completed"
    assert tui_event.tool_name == "lookup"
    assert tui_event.tool_call_id == "call-1"
    assert tui_event.content == '{"ok":true}'
    assert tui_event.tool_result is result


def test_adapt_tool_output_event() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    event = builder.tool_output(tool_call=call, text_delta="partial")

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "tool_output"
    assert tui_event.tool_name == "lookup"
    assert tui_event.content == "partial"


def test_adapt_tool_approval_requested_event() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(name="write_text_file", arguments="{}", call_id="call-1")
    request = ToolApprovalRequest(
        call=call,
        tool_name="write_text_file",
        reason="Tool 'write_text_file' requires approval",
        policy="filesystem_write",
    )
    event = builder.tool_approval_requested(request=request)

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "tool_approval_requested"
    assert tui_event.tool_name == "write_text_file"
    assert tui_event.tool_call_id == "call-1"
    assert "filesystem_write" in tui_event.content


def test_adapt_run_suspended_event() -> None:
    call = LLMToolCall(name="write_text_file", arguments="{}", call_id="call-1")
    request = ToolApprovalRequest(call=call, tool_name="write_text_file")
    event = RunSuspendedEvent(
        step_index=0,
        step_id="step-1",
        request=request,
    )

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "run_suspended"
    assert tui_event.tool_name == "write_text_file"
    assert tui_event.tool_call_id == "call-1"
    assert "Choose Approve or Reject" in tui_event.content


def test_adapt_run_completed_event() -> None:
    event = RunCompletedEvent(
        step_index=0,
        step_id="step-1",
        step=AgentStep(llm_response=LLMResponse(text="done")),
        final_answer="done",
    )

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "run_completed"
    assert tui_event.content == "done"


def test_adapt_media_event_uses_first_segment() -> None:
    media = LLMGeneratedMedia(type="image", mime_type="image/png", data=b"png")
    segment = LLMSegment(type="image", content="", media=media)
    event = LLMMediaEvent(step_index=0, step_id="step-1", segments=[segment])

    tui_event = TUIEvent.from_agent_event(event)

    assert tui_event.kind == "media"
    assert tui_event.segment is segment
