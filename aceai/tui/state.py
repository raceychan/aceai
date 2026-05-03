"""State reducer for the read-only TUI prototype."""

from typing import Literal

from msgspec import field

from aceai.interface import Record

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


class TUIRunState(Record, kw_only=True):
    status: TUIRunStatus = "idle"
    steps: list[TUIStepState] = field(default_factory=list[TUIStepState])
    events: list[TUIEvent] = field(default_factory=list[TUIEvent])
    selected_event_id: str | None = None
    final_answer: str = ""
    error: str | None = None


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
    return TUIRunState(
        status=status,
        steps=steps,
        events=[*state.events, event],
        selected_event_id=event.event_id,
        final_answer=event.content if event.kind == "run_completed" else state.final_answer,
        error=event.error if event.kind == "run_failed" else state.error,
    )


def _apply_step_event(
    steps: list[TUIStepState],
    event: TUIEvent,
) -> list[TUIStepState]:
    if event.kind == "user_message":
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
        events=[*step.events, event],
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
    if event.kind == "step_started":
        return "running"
    if event.kind == "run_completed":
        return "completed"
    if event.kind == "run_failed":
        return "failed"
    if status == "idle":
        return "running"
    return status


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
