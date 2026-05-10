"""Replay durable AceAI session logs into TUI display events."""

from aceai.agent.session import EventLog

from .events import TUIEvent


def event_log_to_tui_events(event_log: EventLog) -> list[TUIEvent]:
    events: list[TUIEvent] = []
    pending_approvals: dict[str, TUIEvent] = {}
    for session_event in event_log.events:
        if _expires_pending_approval(session_event.kind):
            events.extend(_expired_approval_events(pending_approvals))
            pending_approvals = {}
        event = TUIEvent.from_session_event(session_event)
        if event is None:
            continue
        if (
            event.kind == "run_suspended"
            and event.tool_call_id is None
            and len(pending_approvals) == 1
        ):
            event = _run_suspended_event_for_pending_approval(
                event,
                next(iter(pending_approvals.values())),
            )
        events.append(event)
        if event.kind == "tool_approval_requested" and event.tool_call_id is not None:
            pending_approvals[event.tool_call_id] = event
        elif event.kind in (
            "tool_approval_resolved",
            "tool_completed",
            "tool_failed",
        ):
            if event.tool_call_id is not None and event.tool_call_id in pending_approvals:
                del pending_approvals[event.tool_call_id]
    events.extend(_expired_approval_events(pending_approvals))
    return events


def _run_suspended_event_for_pending_approval(
    event: TUIEvent,
    approval: TUIEvent,
) -> TUIEvent:
    return TUIEvent(
        kind=event.kind,
        step_index=event.step_index,
        step_id=event.step_id,
        title=event.title,
        content=event.content,
        raw_event=event.raw_event,
        event_id=event.event_id,
        tool_name=approval.tool_name,
        tool_call_id=approval.tool_call_id,
        tool_call=approval.tool_call,
    )


def _expires_pending_approval(kind: str) -> bool:
    return kind in (
        "user_message",
        "user_steer",
        "assistant_message",
        "assistant_tool_call",
    )


def _expired_approval_events(pending_approvals: dict[str, TUIEvent]) -> list[TUIEvent]:
    events: list[TUIEvent] = []
    for approval in pending_approvals.values():
        tool_name = approval.tool_name or "tool call"
        events.append(
            TUIEvent.session_notice(
                f"approval expired: {tool_name} was not resolved in this run. Ask again to create a fresh approval."
            )
        )
    return events
