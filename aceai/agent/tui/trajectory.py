"""Full trajectory screen for the AceAI TUI."""

import json

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, RichLog

from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.theme import EVENT_LABELS, EVENT_STYLES


class TrajectoryScreen(ModalScreen[None]):
    """Show the full chronological event trajectory."""

    DEFAULT_CSS = """
    TrajectoryScreen {
        align: center middle;
    }

    #trajectory-panel {
        width: 128;
        height: 42;
        max-height: 42;
        border: solid #88c0d0;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #trajectory-body {
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
    }

    #trajectory-actions {
        height: 3;
        margin-top: 1;
    }

    Button {
        width: auto;
        min-width: 10;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, events: list[TUIEvent]) -> None:
        super().__init__()
        self._events = events

    def compose(self) -> ComposeResult:
        with Container(id="trajectory-panel"):
            body = RichLog(id="trajectory-body", wrap=True, auto_scroll=False)
            for renderable in _trajectory_renderables(self._events):
                body.write(renderable)
            yield body
            with Container(id="trajectory-actions"):
                yield Button("Close", id="trajectory-close")

    def on_mount(self) -> None:
        self.query_one("#trajectory-body", RichLog).focus()

    def on_key(self, event: Key) -> None:
        body = self.query_one("#trajectory-body", RichLog)
        if event.key == "up":
            body.scroll_up(animate=False)
            event.stop()
            return
        if event.key == "down":
            body.scroll_down(animate=False)
            event.stop()
            return
        if event.key == "pageup":
            body.scroll_page_up(animate=False)
            event.stop()
            return
        if event.key == "pagedown":
            body.scroll_page_down(animate=False)
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "trajectory-close":
            self.dismiss(None)


def _trajectory_renderables(events: list[TUIEvent]) -> list[RenderableType]:
    if not events:
        return [Text("No events yet", style="#d8dee9")]

    groups = _events_by_turn(events)
    renderables: list[RenderableType] = [
        Text("Trajectory", style="bold #eceff4"),
        _summary(events, groups),
    ]
    turn_index = 1
    for group in groups:
        renderables.append(Text(""))
        if group[0].kind == "user_message":
            renderables.extend(_turn_renderables(turn_index, group))
            turn_index += 1
        else:
            renderables.extend(_session_renderables(group))
    return renderables


def _events_by_turn(events: list[TUIEvent]) -> list[list[TUIEvent]]:
    groups: list[list[TUIEvent]] = []
    for event in events:
        if event.kind == "user_message" or len(groups) == 0:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def _summary(events: list[TUIEvent], groups: list[list[TUIEvent]]) -> Table:
    turn_count = sum(1 for group in groups if group[0].kind == "user_message")
    step_count = len({event.step_id for event in events if event.step_index >= 0})
    rejected_call_ids = _rejected_tool_call_ids(events)
    tool_call_count = len(
        {
            event.tool_call_id
            for event in events
            if event.tool_call_id is not None
        }
    )
    approval_count = sum(
        1
        for event in events
        if event.kind in ("tool_approval_requested", "tool_approval_resolved")
    )
    failure_count = sum(
        1
        for event in events
        if event.kind in ("run_failed", "step_failed", "tool_failed")
        and not _is_rejected_tool_failure(event, rejected_call_ids)
    )
    table = Table.grid(expand=True)
    table.add_column(width=12, style="#9aa3b2")
    table.add_column(width=8, style="#eceff4")
    table.add_column(width=12, style="#9aa3b2")
    table.add_column(ratio=1, style="#eceff4")
    table.add_row("turns", str(turn_count), "tool calls", str(tool_call_count))
    table.add_row("steps", str(step_count), "approvals", str(approval_count))
    table.add_row("events", str(len(events)), "rejected", str(len(rejected_call_ids)))
    table.add_row("", "", "failures", str(failure_count))
    return table


def _rejected_tool_call_ids(events: list[TUIEvent]) -> set[str]:
    return {
        event.tool_call_id
        for event in events
        if event.kind == "tool_approval_resolved"
        and event.tool_call_id is not None
        and event.content.startswith("rejected")
    }


def _is_rejected_tool_failure(event: TUIEvent, rejected_call_ids: set[str]) -> bool:
    return event.kind == "tool_failed" and event.tool_call_id in rejected_call_ids


def _turn_renderables(turn_index: int, events: list[TUIEvent]) -> list[RenderableType]:
    question = events[0]
    renderables: list[RenderableType] = [_turn_header(turn_index, question.content)]
    timeline_events = events[1:]
    if len(timeline_events) == 0:
        return renderables
    step_events, outcome_events = _split_outcome_events(timeline_events)
    renderables.extend(
        _step_renderables(
            step_events,
            suspended_step_ids=_suspended_step_ids(outcome_events),
        )
    )
    outcome_events = _dedupe_outcome_events(step_events, outcome_events)
    if len(outcome_events) > 0:
        renderables.append(_outcome_table(outcome_events))
    return renderables


def _session_renderables(events: list[TUIEvent]) -> list[RenderableType]:
    step_events, outcome_events = _split_outcome_events(events)
    renderables: list[RenderableType] = [_session_header()]
    renderables.extend(
        _step_renderables(
            step_events,
            suspended_step_ids=_suspended_step_ids(outcome_events),
        )
    )
    outcome_events = _dedupe_outcome_events(step_events, outcome_events)
    if len(outcome_events) > 0:
        renderables.append(_outcome_table(outcome_events))
    return renderables


def _turn_header(turn_index: int, content: str) -> Text:
    label = f" {turn_index}  {content}"
    padded = label if len(label) >= 96 else label + (" " * (96 - len(label)))
    return Text(padded, style="bold #2e3440 on #88c0d0")


def _session_header() -> Text:
    return Text(" session" + (" " * 89), style="bold #2e3440 on #88c0d0")


def _split_outcome_events(events: list[TUIEvent]) -> tuple[list[TUIEvent], list[TUIEvent]]:
    step_events: list[TUIEvent] = []
    outcome_events: list[TUIEvent] = []
    for event in events:
        if event.kind in ("run_completed", "run_failed", "run_suspended"):
            outcome_events.append(event)
        else:
            step_events.append(event)
    return step_events, outcome_events


def _suspended_step_ids(events: list[TUIEvent]) -> set[str]:
    return {event.step_id for event in events if event.kind == "run_suspended"}


def _dedupe_outcome_events(
    step_events: list[TUIEvent],
    outcome_events: list[TUIEvent],
) -> list[TUIEvent]:
    last_assistant = _last_assistant_content(step_events)
    if last_assistant == "":
        return outcome_events
    return [
        event
        for event in outcome_events
        if event.kind != "run_completed" or event.content != last_assistant
    ]


def _last_assistant_content(events: list[TUIEvent]) -> str:
    for event in reversed(events):
        if event.kind == "assistant_delta":
            return event.content
    return ""


def _step_renderables(
    events: list[TUIEvent],
    *,
    suspended_step_ids: set[str],
) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    for group in _events_by_step(events):
        renderables.extend(
            _single_step_renderables(
                group,
                is_suspended=group[0].step_id in suspended_step_ids,
            )
        )
    return renderables


def _events_by_step(events: list[TUIEvent]) -> list[list[TUIEvent]]:
    groups: list[list[TUIEvent]] = []
    for event in events:
        if _starts_step_group(groups, event):
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def _starts_step_group(groups: list[list[TUIEvent]], event: TUIEvent) -> bool:
    if len(groups) == 0:
        return True
    previous = groups[-1][0]
    return event.step_index != previous.step_index or event.step_id != previous.step_id


def _single_step_renderables(
    events: list[TUIEvent],
    *,
    is_suspended: bool,
) -> list[RenderableType]:
    step_event = events[0]
    body_events = [
        event
        for event in events
        if event.kind not in ("step_started", "step_completed", "step_failed")
    ]
    renderables: list[RenderableType] = [_step_header(events, is_suspended=is_suspended)]
    if len(body_events) > 0:
        renderables.append(_event_table(body_events))
    elif step_event.kind not in ("step_started", "step_completed", "step_failed"):
        renderables.append(_event_table([step_event]))
    return renderables


def _step_header(events: list[TUIEvent], *, is_suspended: bool) -> Text:
    first = events[0]
    status = _step_status(events, is_suspended=is_suspended)
    step = "-" if first.step_index < 0 else str(first.step_index + 1)
    text = Text()
    text.append("  ▌", style=_step_status_style(status))
    text.append(f" {step}", style=_step_status_style(status))
    if status != "completed":
        text.append("  ")
        text.append(status, style=_step_status_style(status))
    text.append("  ")
    text.append(_short_id(first.step_id), style="#9aa3b2")
    return text


def _step_status(events: list[TUIEvent], *, is_suspended: bool) -> str:
    if is_suspended:
        return "waiting"
    for event in reversed(events):
        if event.kind == "step_failed":
            return "failed"
        if event.kind == "step_completed":
            return "completed"
    return "running"


def _step_status_style(status: str) -> str:
    if status == "failed":
        return "bold #bf616a"
    if status == "waiting":
        return "bold #ebcb8b"
    if status == "running":
        return "bold #88c0d0"
    return "bold #a3be8c"


def _event_table(events: list[TUIEvent]) -> Table:
    rejected_call_ids = _rejected_tool_call_ids(events)
    table = Table.grid(expand=True)
    table.add_column(width=6, style="#4c566a")
    table.add_column(width=12, style="bold")
    table.add_column(ratio=1, overflow="fold")
    for index, event in enumerate(events, start=1):
        connector = "└" if index == len(events) else "│"
        table.add_row(
            f"    {connector}",
            Text(
                _event_label(event, rejected_call_ids),
                style=_event_style(event, rejected_call_ids),
            ),
            _event_summary(event),
        )
    return table


def _outcome_table(events: list[TUIEvent]) -> Table:
    table = Table.grid(expand=True)
    table.add_column(width=6, style="#a3be8c")
    table.add_column(width=12, style="bold")
    table.add_column(ratio=1, overflow="fold")
    for event in events:
        table.add_row(
            "  ◆",
            Text(_event_label(event, set()), style=EVENT_STYLES[event.kind]),
            _event_summary(event),
        )
    return table


def _event_label(event: TUIEvent, rejected_call_ids: set[str]) -> str:
    if _is_rejected_tool_failure(event, rejected_call_ids):
        return "rejected"
    if event.kind == "run_completed":
        return "answer"
    if event.kind == "tool_started":
        return "call"
    if event.kind == "tool_approval_resolved":
        if event.content.startswith("rejected"):
            return "rejected"
        return "approved"
    return EVENT_LABELS[event.kind]


def _event_style(event: TUIEvent, rejected_call_ids: set[str]) -> str:
    if _is_rejected_tool_failure(event, rejected_call_ids):
        return "bold #d08770"
    if event.kind == "tool_approval_resolved":
        if event.content.startswith("rejected"):
            return "bold #d08770"
        return "bold #a3be8c"
    return EVENT_STYLES[event.kind]


def _event_summary(event: TUIEvent) -> str:
    parts: list[str] = []
    subject = _event_subject(event)
    if subject != "":
        parts.append(subject)
    body = _event_body(event)
    if body != "":
        parts.append(body)
    return " - ".join(parts)


def _event_body(event: TUIEvent) -> str:
    if event.error is not None:
        return _preview_block(event.error)
    if event.tool_result is not None and event.tool_result.output != "":
        return _tool_result_body(event)
    if event.content != "":
        return _preview_block(event.content)
    if event.tool_call is not None and event.tool_call.arguments != "":
        return _tool_call_body(event)
    return ""


def _tool_call_body(event: TUIEvent) -> str:
    if event.tool_call is None:
        return ""
    payload = json.loads(event.tool_call.arguments)
    if event.tool_name == "run_shell_command":
        return f"$ {payload['command']}"
    if event.tool_name in ("read_text_file", "write_text_file"):
        return payload["path"]
    if event.tool_name == "search_text":
        return f"{payload['query']} in {payload['path']}"
    return _preview(event.tool_call.arguments)


def _tool_result_body(event: TUIEvent) -> str:
    if event.tool_result is None:
        return ""
    payload = json.loads(event.tool_result.output)
    if event.tool_name == "run_shell_command":
        return _command_result_body(payload)
    if event.tool_name == "read_text_file":
        return _preview_block(payload["content"])
    if event.tool_name == "write_text_file":
        return f"wrote {payload['bytes_written']} bytes"
    if event.tool_name == "search_text":
        return _preview_block(payload["matches"])
    return _preview(event.tool_result.output)


def _command_result_body(payload) -> str:
    if payload["stdout"] != "":
        return _preview_block(payload["stdout"])
    if payload["stderr"] != "":
        return _preview_block(payload["stderr"])
    return f"exit {payload['exit_code']}"


def _event_subject(event: TUIEvent) -> str:
    if event.tool_name is not None:
        return event.tool_name
    if event.kind == "tool_call_delta":
        return _short_id(event.tool_call_id)
    if event.kind in ("assistant_delta", "thinking_delta", "reasoning_summary"):
        return ""
    if event.kind in ("run_completed", "run_failed", "run_suspended"):
        return ""
    if event.kind in ("step_started", "step_completed", "step_failed"):
        return _short_id(event.step_id)
    return event.title


def _preview(content: str) -> str:
    first_line = content.partition("\n")[0]
    if len(first_line) <= 80:
        return first_line
    return f"{first_line[:77]}..."


def _preview_block(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    preview = _preview(lines[0])
    hidden_lines = len(lines) - 1
    if hidden_lines > 0:
        return f"{preview} ... (+{hidden_lines} lines)"
    return preview


def _short_id(value: str | None) -> str:
    if value is None:
        return ""
    if len(value) <= 12:
        return value
    return f"{value[:8]}...{value[-4:]}"


def _indented(value: str) -> str:
    return "\n".join(f"  {line}" for line in value.splitlines())
