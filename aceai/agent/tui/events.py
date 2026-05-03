"""TUI-facing event records derived from agent events."""

from typing import Literal

from msgspec import field

from aceai.core.events import (
    AgentEvent,
    LLMCompletedEvent,
    LLMOutputDeltaEvent,
    LLMReasoningEvent,
    LLMStartedEvent,
    LLMToolCallDeltaEvent,
    LLMMediaEvent,
    RunCompletedEvent,
    RunFailedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ToolOutputEvent,
    ToolStartedEvent,
)
from aceai.core.helpers.string import uuid_str
from aceai.llm.interface import Record
from aceai.llm.models import LLMSegment, LLMToolCall, LLMToolCallDelta
from aceai.core.models import ToolExecutionResult

TUIEventKind = Literal[
    "user_message",
    "session_notice",
    "run_completed",
    "run_failed",
    "step_completed",
    "step_failed",
    "step_started",
    "llm_completed",
    "assistant_delta",
    "thinking_delta",
    "reasoning_summary",
    "tool_call_delta",
    "tool_started",
    "tool_output",
    "tool_completed",
    "tool_failed",
    "media",
]


class TUIEvent(Record, kw_only=True):
    """Normalized event shape consumed by future TUI widgets."""

    kind: TUIEventKind
    step_index: int
    step_id: str
    title: str
    raw_event: AgentEvent | None
    event_id: str = field(default_factory=uuid_str)
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_call: LLMToolCall | None = None
    tool_call_delta: LLMToolCallDelta | None = None
    tool_result: ToolExecutionResult | None = None
    segment: LLMSegment | None = None
    error: str | None = None


def user_message_event(question: str) -> TUIEvent:
    """Create a TUI-only event for the submitted user question."""

    return TUIEvent(
        kind="user_message",
        step_index=-1,
        step_id=uuid_str(),
        title="you",
        content=question,
        raw_event=None,
    )


def session_notice_event(content: str) -> TUIEvent:
    return TUIEvent(
        kind="session_notice",
        step_index=-1,
        step_id=uuid_str(),
        title="session",
        content=content,
        raw_event=None,
    )


def adapt_agent_event(event: AgentEvent) -> TUIEvent:
    """Convert an agent event into the stable TUI event shape."""

    if isinstance(event, LLMStartedEvent):
        return TUIEvent(
            kind="step_started",
            step_index=event.step_index,
            step_id=event.step_id,
            title="step started",
            raw_event=event,
        )
    if isinstance(event, LLMOutputDeltaEvent):
        return TUIEvent(
            kind="assistant_delta",
            step_index=event.step_index,
            step_id=event.step_id,
            title="assistant",
            content=event.text_delta,
            raw_event=event,
        )
    if isinstance(event, LLMReasoningEvent):
        return TUIEvent(
            kind="reasoning_summary",
            step_index=event.step_index,
            step_id=event.step_id,
            title="reasoning",
            content=event.segment.content,
            segment=event.segment,
            raw_event=event,
        )
    if isinstance(event, LLMToolCallDeltaEvent):
        return TUIEvent(
            kind="tool_call_delta",
            step_index=event.step_index,
            step_id=event.step_id,
            title="tool arguments",
            content=event.text_delta,
            tool_call_delta=event.tool_call_delta,
            tool_call_id=event.tool_call_delta.id,
            raw_event=event,
        )
    if isinstance(event, LLMMediaEvent):
        return TUIEvent(
            kind="media",
            step_index=event.step_index,
            step_id=event.step_id,
            title="media",
            segment=event.segments[0],
            raw_event=event,
        )
    if isinstance(event, LLMCompletedEvent):
        return TUIEvent(
            kind="llm_completed",
            step_index=event.step_index,
            step_id=event.step_id,
            title="llm completed",
            content=event.step.llm_response.text,
            raw_event=event,
        )
    if isinstance(event, ToolStartedEvent):
        return TUIEvent(
            kind="tool_started",
            step_index=event.step_index,
            step_id=event.step_id,
            title=f"tool {event.tool_name}",
            tool_name=event.tool_name,
            tool_call_id=event.tool_call.call_id,
            tool_call=event.tool_call,
            raw_event=event,
        )
    if isinstance(event, ToolOutputEvent):
        return TUIEvent(
            kind="tool_output",
            step_index=event.step_index,
            step_id=event.step_id,
            title=f"tool {event.tool_name} output",
            content=event.text_delta,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call.call_id,
            tool_call=event.tool_call,
            raw_event=event,
        )
    if isinstance(event, ToolCompletedEvent):
        return TUIEvent(
            kind="tool_completed",
            step_index=event.step_index,
            step_id=event.step_id,
            title=f"tool {event.tool_name} completed",
            content=event.tool_result.output,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call.call_id,
            tool_call=event.tool_call,
            tool_result=event.tool_result,
            raw_event=event,
        )
    if isinstance(event, ToolFailedEvent):
        return TUIEvent(
            kind="tool_failed",
            step_index=event.step_index,
            step_id=event.step_id,
            title=f"tool {event.tool_name} failed",
            content=event.error,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call.call_id,
            tool_call=event.tool_call,
            tool_result=event.tool_result,
            error=event.error,
            raw_event=event,
        )
    if isinstance(event, StepCompletedEvent):
        return TUIEvent(
            kind="step_completed",
            step_index=event.step_index,
            step_id=event.step_id,
            title="step completed",
            raw_event=event,
        )
    if isinstance(event, StepFailedEvent):
        return TUIEvent(
            kind="step_failed",
            step_index=event.step_index,
            step_id=event.step_id,
            title="step failed",
            content=event.error,
            error=event.error,
            raw_event=event,
        )
    if isinstance(event, RunCompletedEvent):
        return TUIEvent(
            kind="run_completed",
            step_index=event.step_index,
            step_id=event.step_id,
            title="run completed",
            content=event.final_answer,
            raw_event=event,
        )
    if isinstance(event, RunFailedEvent):
        return TUIEvent(
            kind="run_failed",
            step_index=event.step_index,
            step_id=event.step_id,
            title="run failed",
            content=event.error,
            error=event.error,
            raw_event=event,
        )
    raise TypeError(f"Unsupported agent event: {event.__class__.__name__}")
