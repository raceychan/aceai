"""State reducer for the read-only TUI prototype."""

from typing import Literal

from msgspec import field

from aceai.llm.interface import Record
from aceai.llm.models import LLMUsage
from aceai.agent.cost import CostEstimate

from .events import TUIEvent

TUIRunStatus = Literal["idle", "running", "completed", "failed"]
TUIStepStatus = Literal["running", "completed", "failed"]
TUIToolStatus = Literal["pending", "running", "completed", "failed"]


class TUIToolState(Record, kw_only=True):
    call_id: str
    name: str | None = None
    status: TUIToolStatus = "pending"
    arguments: str = ""
    output: str = ""
    error: str | None = None
    events: list[TUIEvent] = field(default_factory=list[TUIEvent])


class TUIStepState(Record, kw_only=True):
    step_index: int
    step_id: str
    status: TUIStepStatus = "running"
    events: list[TUIEvent] = field(default_factory=list[TUIEvent])
    tools: list[TUIToolState] = field(default_factory=list[TUIToolState])


class TUIUsageState(Record, kw_only=True):
    current_context_tokens: int | None = None
    current_cached_input_tokens: int | None = None
    session_input_tokens: int | None = None
    session_cached_input_tokens: int | None = None
    session_output_tokens: int | None = None
    session_total_tokens: int | None = None
    current_cost_usd: float | None = None
    session_cost_usd: float | None = None


class TUIRunState(Record, kw_only=True):
    status: TUIRunStatus = "idle"
    steps: list[TUIStepState] = field(default_factory=list[TUIStepState])
    events: list[TUIEvent] = field(default_factory=list[TUIEvent])
    selected_event_id: str | None = None
    final_answer: str = ""
    error: str | None = None
    usage: TUIUsageState = field(default_factory=TUIUsageState)


def initial_state() -> TUIRunState:
    return TUIRunState()


def reduce_events(events: list[TUIEvent]) -> TUIRunState:
    state = initial_state()
    for event in events:
        state = apply_tui_event(state, event)
    return state


def apply_tui_event(state: TUIRunState, event: TUIEvent) -> TUIRunState:
    steps = _apply_step_event(state.steps, event)
    status = _next_run_status(state.status, event)
    events = _append_event(state.events, event)
    selected_event_id = events[-1].event_id
    return TUIRunState(
        status=status,
        steps=steps,
        events=events,
        selected_event_id=selected_event_id,
        final_answer=event.content if event.kind == "run_completed" else state.final_answer,
        error=event.error if event.kind == "run_failed" else state.error,
        usage=_apply_usage_event(state.usage, event),
    )


def _append_event(events: list[TUIEvent], event: TUIEvent) -> list[TUIEvent]:
    if not events:
        return [event]
    previous = events[-1]
    if not _can_merge_stream_delta(previous, event):
        return [*events, event]
    return [*events[:-1], _merge_stream_delta(previous, event)]


def _can_merge_stream_delta(previous: TUIEvent, event: TUIEvent) -> bool:
    return (
        previous.kind == event.kind
        and event.kind in ("assistant_delta", "thinking_delta")
        and previous.step_id == event.step_id
        and previous.step_index == event.step_index
        and previous.tool_call_id == event.tool_call_id
    )


def _merge_stream_delta(previous: TUIEvent, event: TUIEvent) -> TUIEvent:
    return TUIEvent(
        kind=previous.kind,
        step_index=previous.step_index,
        step_id=previous.step_id,
        title=previous.title,
        raw_event=event.raw_event,
        event_id=previous.event_id,
        content=previous.content + event.content,
        tool_name=previous.tool_name,
        tool_call_id=previous.tool_call_id,
        tool_call=previous.tool_call,
        tool_calls=previous.tool_calls,
        tool_call_delta=previous.tool_call_delta,
        tool_result=previous.tool_result,
        segment=event.segment,
        usage=event.usage,
        cost=event.cost,
        error=event.error,
    )


def _apply_usage_event(usage_state: TUIUsageState, event: TUIEvent) -> TUIUsageState:
    if event.usage is None:
        return usage_state
    usage = event.usage
    input_tokens = _token_count(usage.input_tokens)
    output_tokens = _token_count(usage.output_tokens)
    cached_input_tokens = _token_count(usage.cached_input_tokens)
    total_tokens = _token_count(usage.total_tokens)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return TUIUsageState(
        current_context_tokens=usage.input_tokens,
        current_cached_input_tokens=usage.cached_input_tokens,
        session_input_tokens=_add_tokens(
            usage_state.session_input_tokens,
            input_tokens,
        ),
        session_cached_input_tokens=_add_tokens(
            usage_state.session_cached_input_tokens,
            cached_input_tokens,
        ),
        session_output_tokens=_add_tokens(
            usage_state.session_output_tokens,
            output_tokens,
        ),
        session_total_tokens=_add_tokens(
            usage_state.session_total_tokens,
            total_tokens,
        ),
        current_cost_usd=None if event.cost is None else event.cost.total_cost_usd,
        session_cost_usd=_add_cost(usage_state.session_cost_usd, event.cost),
    )


def _token_count(value: int | None) -> int:
    if value is None:
        return 0
    return value


def _add_tokens(total: int | None, increment: int) -> int | None:
    if total is None and increment == 0:
        return None
    if total is None:
        return increment
    return total + increment


def _add_cost(total: float | None, cost: CostEstimate | None) -> float | None:
    if cost is None:
        return total
    if total is None:
        return cost.total_cost_usd
    return total + cost.total_cost_usd


def _apply_step_event(
    steps: list[TUIStepState],
    event: TUIEvent,
) -> list[TUIStepState]:
    if event.kind in ("user_message", "session_notice"):
        return steps
    if event.raw_event is None and event.step_index == -1:
        return steps

    target = _find_step(steps, event.step_id)
    if target is None:
        target = TUIStepState(step_index=len(steps), step_id=event.step_id)

    updated = _update_step(target, event)
    next_steps: list[TUIStepState] = []
    inserted = False
    for step in steps:
        if step.step_id == event.step_id:
            next_steps.append(updated)
            inserted = True
        else:
            next_steps.append(step)
    if not inserted:
        next_steps.append(updated)
    return next_steps


def _find_step(steps: list[TUIStepState], step_id: str) -> TUIStepState | None:
    for step in steps:
        if step.step_id == step_id:
            return step
    return None


def _update_step(step: TUIStepState, event: TUIEvent) -> TUIStepState:
    return TUIStepState(
        step_index=step.step_index,
        step_id=step.step_id,
        status=_next_step_status(step.status, event),
        events=_append_event(step.events, event),
        tools=_apply_tool_event(step.tools, event),
    )


def _apply_tool_event(
    tools: list[TUIToolState],
    event: TUIEvent,
) -> list[TUIToolState]:
    if event.tool_call_id is None:
        return tools

    target = _find_tool(tools, event.tool_call_id)
    if target is None:
        target = TUIToolState(call_id=event.tool_call_id)

    updated = _update_tool(target, event)
    next_tools: list[TUIToolState] = []
    inserted = False
    for tool_state in tools:
        if tool_state.call_id == event.tool_call_id:
            next_tools.append(updated)
            inserted = True
        else:
            next_tools.append(tool_state)
    if not inserted:
        next_tools.append(updated)
    return next_tools


def _find_tool(
    tools: list[TUIToolState],
    call_id: str,
) -> TUIToolState | None:
    for tool_state in tools:
        if tool_state.call_id == call_id:
            return tool_state
    return None


def _update_tool(tool_state: TUIToolState, event: TUIEvent) -> TUIToolState:
    name = event.tool_name if event.tool_name is not None else tool_state.name
    return TUIToolState(
        call_id=tool_state.call_id,
        name=name,
        status=_next_tool_status(tool_state.status, event),
        arguments=(
            tool_state.arguments + event.content
            if event.kind == "tool_call_delta"
            else tool_state.arguments
        ),
        output=(
            tool_state.output + event.content
            if event.kind == "tool_output"
            else event.content
            if event.kind == "tool_completed"
            else tool_state.output
        ),
        error=event.error if event.kind == "tool_failed" else tool_state.error,
        events=[*tool_state.events, event],
    )


def _next_run_status(status: TUIRunStatus, event: TUIEvent) -> TUIRunStatus:
    if _is_restored_transcript_event(event):
        return status
    if event.kind in ("user_message", "session_notice"):
        return status
    if event.kind == "step_started":
        return "running"
    if event.kind == "run_completed":
        return "completed"
    if event.kind == "run_failed":
        return "failed"
    if status == "idle":
        return "running"
    return status


def _is_restored_transcript_event(event: TUIEvent) -> bool:
    return event.raw_event is None and event.step_index == -1


def _next_step_status(status: TUIStepStatus, event: TUIEvent) -> TUIStepStatus:
    if event.kind == "step_completed":
        return "completed"
    if event.kind == "step_failed":
        return "failed"
    return status


def _next_tool_status(status: TUIToolStatus, event: TUIEvent) -> TUIToolStatus:
    if event.kind == "tool_started":
        return "running"
    if event.kind == "tool_completed":
        return "completed"
    if event.kind == "tool_failed":
        return "failed"
    if status == "pending" and event.kind in ("tool_call_delta", "tool_output"):
        return "running"
    return status
