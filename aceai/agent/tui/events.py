"""TUI-facing event records derived from agent events."""

from typing import Literal

from msgspec import field
from typing_extensions import Self

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
    RunSuspendedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    ToolApprovalRequestedEvent,
    ToolApprovalResolvedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ToolOutputEvent,
    ToolStartedEvent,
)
from aceai.core.helpers.string import uuid_str
from aceai.llm.interface import Record, is_set
from aceai.llm.models import (
    LLMReasoningSegmentMeta,
    LLMUsage,
    LLMSegment,
    LLMToolCall,
    LLMToolCallDelta,
)
from aceai.core.models import ToolExecutionResult
from aceai.agent.cost import CostEstimate, estimate_usage_cost
from aceai.agent.session import EventLog, SessionEvent

TUIEventKind = Literal[
    "user_message",
    "session_notice",
    "run_completed",
    "run_failed",
    "run_suspended",
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
    "tool_approval_requested",
    "tool_approval_resolved",
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
    tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
    tool_call_delta: LLMToolCallDelta | None = None
    tool_result: ToolExecutionResult | None = None
    segment: LLMSegment | None = None
    usage: LLMUsage | None = None
    cost: CostEstimate | None = None
    error: str | None = None

    @classmethod
    def user_message(cls, question: str) -> Self:
        return cls(
            kind="user_message",
            step_index=-1,
            step_id=uuid_str(),
            title="you",
            content=question,
            raw_event=None,
        )

    @classmethod
    def session_notice(cls, content: str) -> Self:
        return cls(
            kind="session_notice",
            step_index=-1,
            step_id=uuid_str(),
            title="session",
            content=content,
            raw_event=None,
        )

    @classmethod
    def from_agent_event(cls, event: AgentEvent) -> "TUIEvent":
        return _agent_event_to_tui_event(event)

    @classmethod
    def from_session_event(cls, event: SessionEvent) -> "TUIEvent | None":
        if event.kind == "user_message":
            return cls.user_message(event.payload["content"])
        if event.kind == "assistant_message":
            return cls._from_session_assistant_event(event)
        if event.kind == "assistant_tool_call":
            return None
        if event.kind == "tool_started":
            return cls._from_session_tool_started_event(event)
        if event.kind in ("tool_approval_requested", "tool_approval_resolved"):
            return cls._from_session_tool_approval_event(event)
        if event.kind == "tool_result":
            return cls._from_session_tool_result_event(event)
        if event.kind == "error":
            return cls._from_session_error_event(event)
        if event.kind in ("run_completed", "run_suspended", "step_completed", "step_started"):
            return cls._from_session_control_event(event)
        return None

    @classmethod
    def list_from_event_log(cls, event_log: EventLog) -> list["TUIEvent"]:
        events: list[TUIEvent] = []
        for session_event in event_log.events:
            event = cls.from_session_event(session_event)
            if event is not None:
                events.append(event)
        return events

    @classmethod
    def _from_session_assistant_event(cls, event: SessionEvent) -> Self:
        return cls(
            kind="assistant_delta",
            step_index=_session_step_index(event),
            step_id=event.step_id or uuid_str(),
            title="assistant",
            content=event.payload["content"],
            usage=_session_usage(event),
            cost=_session_cost(event),
            raw_event=None,
        )

    @classmethod
    def _from_session_tool_result_event(cls, event: SessionEvent) -> Self:
        call = _session_tool_call(event)
        status = event.payload["status"]
        return cls(
            kind="tool_failed" if status == "failed" else "tool_completed",
            step_index=_session_step_index(event),
            step_id=event.step_id or uuid_str(),
            title=f"tool {event.payload['tool_name']}",
            content=event.payload["output"],
            tool_name=event.payload["tool_name"],
            tool_call_id=event.payload["tool_call_id"],
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output=event.payload["output"],
                error=event.payload["content"] if status == "failed" else None,
            ),
            error=event.payload["content"] if status == "failed" else None,
            raw_event=None,
        )

    @classmethod
    def _from_session_tool_started_event(cls, event: SessionEvent) -> Self:
        call = _session_tool_call(event)
        return cls(
            kind="tool_started",
            step_index=_session_step_index(event),
            step_id=event.step_id or uuid_str(),
            title=f"tool {event.payload['tool_name']}",
            tool_name=event.payload["tool_name"],
            tool_call_id=event.payload["tool_call_id"],
            tool_call=call,
            raw_event=None,
        )

    @classmethod
    def _from_session_tool_approval_event(cls, event: SessionEvent) -> Self:
        call = _session_tool_call(event)
        kind: TUIEventKind = "tool_approval_requested"
        if event.kind == "tool_approval_resolved":
            kind = "tool_approval_resolved"
        return cls(
            kind=kind,
            step_index=_session_step_index(event),
            step_id=event.step_id or uuid_str(),
            title=f"tool {event.payload['tool_name']}",
            content=event.payload["content"],
            tool_name=event.payload["tool_name"],
            tool_call_id=event.payload["tool_call_id"],
            tool_call=call,
            raw_event=None,
        )

    @classmethod
    def _from_session_error_event(cls, event: SessionEvent) -> Self:
        return cls(
            kind="run_failed",
            step_index=_session_step_index(event),
            step_id=event.step_id or uuid_str(),
            title="run failed",
            content=event.payload["content"],
            error=event.payload["content"],
            raw_event=None,
        )

    @classmethod
    def _from_session_control_event(cls, event: SessionEvent) -> Self:
        kind: TUIEventKind
        if event.kind == "run_completed":
            kind = "run_completed"
        elif event.kind == "run_suspended":
            kind = "run_suspended"
        elif event.kind == "step_completed":
            kind = "step_completed"
        elif event.kind == "step_started":
            kind = "step_started"
        else:
            raise ValueError("unsupported session control event")
        return cls(
            kind=kind,
            step_index=_session_step_index(event),
            step_id=event.step_id or uuid_str(),
            title=event.kind,
            content=event.payload["content"],
            tool_name=_session_payload_str_or_none(event, "tool_name"),
            tool_call_id=_session_payload_str_or_none(event, "tool_call_id"),
            tool_call=_session_optional_tool_call(event),
            raw_event=None,
        )


def _agent_event_to_tui_event(event: AgentEvent) -> TUIEvent:
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
        kind: TUIEventKind = "reasoning_summary"
        if (
            isinstance(event.segment.meta, LLMReasoningSegmentMeta)
            and event.segment.meta.is_delta
        ):
            kind = "thinking_delta"
        return TUIEvent(
            kind=kind,
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
        response = event.step.llm_response
        usage: LLMUsage | None = None
        if is_set(response.usage):
            usage = response.usage
        provider_name = None
        if response.provider_meta:
            provider_name = response.provider_meta[0].provider_name
        cost = estimate_usage_cost(response.model, usage, provider_name=provider_name)
        return TUIEvent(
            kind="llm_completed",
            step_index=event.step_index,
            step_id=event.step_id,
            title="llm completed",
            content="" if response.tool_calls else response.text,
            tool_calls=response.tool_calls,
            usage=usage,
            cost=cost,
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
    if isinstance(event, ToolApprovalRequestedEvent):
        content = event.request.reason
        if event.request.policy != "":
            content = f"{content} ({event.request.policy})"
        return TUIEvent(
            kind="tool_approval_requested",
            step_index=event.step_index,
            step_id=event.step_id,
            title=f"tool {event.tool_name} approval",
            content=content,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call.call_id,
            tool_call=event.tool_call,
            raw_event=event,
        )
    if isinstance(event, ToolApprovalResolvedEvent):
        decision_text = "approved" if event.decision.approved else "rejected"
        content = decision_text
        if event.decision.reason != "":
            content = f"{decision_text}: {event.decision.reason}"
        return TUIEvent(
            kind="tool_approval_resolved",
            step_index=event.step_index,
            step_id=event.step_id,
            title=f"tool {event.tool_name} approval resolved",
            content=content,
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
    if isinstance(event, RunSuspendedEvent):
        return TUIEvent(
            kind="run_suspended",
            step_index=event.step_index,
            step_id=event.step_id,
            title="run suspended",
            content="waiting for approval. Choose Approve or Reject.",
            tool_name=event.request.tool_name,
            tool_call_id=event.request.call.call_id,
            tool_call=event.request.call,
            raw_event=event,
        )
    raise TypeError(f"Unsupported agent event: {event.__class__.__name__}")


def _session_usage(event: SessionEvent) -> LLMUsage | None:
    if "usage" not in event.payload:
        return None
    return LLMUsage.from_payload(event.payload["usage"])


def _session_cost(event: SessionEvent) -> CostEstimate | None:
    if "cost" not in event.payload:
        return None
    return CostEstimate.from_payload(event.payload["cost"])


def _session_step_index(event: SessionEvent) -> int:
    if event.step_index is None:
        return -1
    return event.step_index


def _session_tool_call(event: SessionEvent) -> LLMToolCall:
    if "tool_call" in event.payload:
        return LLMToolCall.from_payload(event.payload["tool_call"])
    return LLMToolCall.from_payload(
        {
            "type": "function_call",
            "name": event.payload["tool_name"],
            "arguments": event.payload["tool_arguments"],
            "call_id": event.payload["tool_call_id"],
        }
    )


def _session_optional_tool_call(event: SessionEvent) -> LLMToolCall | None:
    if "tool_call" not in event.payload:
        return None
    return LLMToolCall.from_payload(event.payload["tool_call"])


def _session_payload_str_or_none(event: SessionEvent, key: str) -> str | None:
    if key not in event.payload:
        return None
    value = event.payload[key]
    if type(value) is not str:
        raise TypeError(f"Session event payload {key} must be str")
    return value
