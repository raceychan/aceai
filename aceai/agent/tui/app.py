"""Read-only Textual application for AceAI event streams."""

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.events import Key
from textual.timer import Timer
from textual.widgets import Footer, Header

from aceai.core.events import AgentEvent

from aceai.agent.session import SessionRecorder, SessionStore

from aceai.agent.cost import format_usd
from .events import TUIEvent
from .metadata import MetadataScreen, MetadataSection
from .session_adapter import tui_event_to_session_event
from .session_display import session_display_title
from .session_replay import event_log_to_tui_events
from .setup import SessionSelectScreen
from .state import TUIRunState, apply_tui_event, initial_state, reduce_events, select_event
from .trajectory import TrajectoryScreen
from .widgets import (
    ApprovalWidget,
    CommandInput,
    DetailWidget,
    StatusBarWidget,
    StreamWidget,
)

STREAM_DELTA_REFRESH_CHARS = 32
STREAM_DELTA_REFRESH_SECONDS = 0.02


class AceAITUI(App[None]):
    """Read-only TUI prototype backed by normalized TUI events."""

    CSS = """
    Screen {
        layout: vertical;
        background: #2e3440;
        color: #e5e9f0;
    }

    Header {
        background: #3b4252;
        color: #eceff4;
        text-style: bold;
    }

    #main {
        height: 1fr;
    }

    #stream {
        min-width: 40;
    }

    DetailWidget.collapsed {
        display: none;
    }

    Footer {
        background: #3b4252;
        color: #eceff4;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("d", "toggle_debug_mode", "Debug"),
        ("c", "config", "Config"),
        ("t", "trajectory", "Trajectory"),
        ("i", "metadata", "Info"),
        ("s", "session_switcher", "Sessions"),
    ]

    def __init__(
        self,
        events: list[TUIEvent] | None = None,
        *,
        model: str | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        record_events: bool = True,
    ) -> None:
        super().__init__()
        self._events = list(events or [])
        self._state: TUIRunState = initial_state()
        self._status_model = model
        self._session_recorder = session_recorder
        self._session_id = session_id
        self._record_events = record_events
        self._pending_stream_delta_chars = 0
        self._pending_stream_delta: TUIEvent | None = None
        self._pending_stream_delta_timer: Timer | None = None
        self.title = "AceAI" if session_id is None else f"AceAI {session_id}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield StreamWidget(id="stream")
            yield DetailWidget(id="detail", classes="collapsed")
        yield ApprovalWidget(id="approval", classes="collapsed")
        yield StatusBarWidget(id="status")
        yield CommandInput(id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.load_events(self._events)
        self.query_one(StreamWidget).focus()

    def on_key(self, event: Key) -> None:
        if event.key != "enter":
            return
        command_input = self.query_one(CommandInput)
        if command_input.has_focus:
            return
        command_input.focus()
        event.stop()

    def load_events(self, events: list[TUIEvent]) -> None:
        self._clear_pending_stream_delta()
        if self.is_mounted:
            self.clear_approval_request()
        self._state = reduce_events(events)
        self._refresh_widgets()

    def show_sessions(self) -> None:
        store = self._session_store()
        sessions = store.list_sessions()
        if not sessions:
            self.append_event(TUIEvent.session_notice("No sessions found."))
            return
        total_cost = store.total_cost_usd()
        lines = [f"Total cost: {format_usd(total_cost)}", "", "Sessions:"]
        for session in sessions:
            marker = "*" if session.session_id == self._session_id else "-"
            lines.append(
                f"{marker} {session.session_id}  "
                f"{session_display_title(session.title)}  {session.updated_at}"
            )
        lines.append("")
        lines.append("Use /resume <session_id> to switch sessions.")
        self.append_event(TUIEvent.session_notice("\n".join(lines)))

    def open_session_selector(self) -> None:
        store = self._session_store()
        sessions = store.list_sessions()
        if not sessions:
            self.append_event(TUIEvent.session_notice("No sessions found."))
            return
        self.push_screen(
            SessionSelectScreen(
                store=store,
                sessions=sessions,
                current_session_id=self._session_id,
            ),
            self._handle_session_selection,
        )

    def _handle_session_selection(self, session_id: str | None) -> None:
        if session_id is None:
            return
        self.switch_session(session_id)

    def switch_session(self, session_id: str) -> None:
        if session_id == self._session_id:
            return
        store = self._session_store()
        try:
            metadata = store.get_session(session_id)
        except KeyError:
            self.append_event(TUIEvent.session_notice(f"Session not found: {session_id}"))
            return
        if self._session_recorder is not None:
            self._session_recorder.finalize()
        self._session_recorder = SessionRecorder(store, metadata.session_id)
        self._session_id = metadata.session_id
        self.title = f"AceAI {metadata.session_id}"
        event_log = store.load_event_log(metadata.session_id)
        self.load_events(event_log_to_tui_events(event_log))
        self.append_event(TUIEvent.session_notice(f"Resumed session {metadata.session_id}"))

    def ensure_session(self) -> None:
        if self._session_recorder is not None:
            return
        store = SessionStore()
        metadata = store.create_session()
        self._session_recorder = SessionRecorder(store, metadata.session_id)
        self._session_id = metadata.session_id
        self.title = f"AceAI {metadata.session_id}"

    def _session_store(self) -> SessionStore:
        if self._session_recorder is not None:
            return self._session_recorder.store
        return SessionStore()

    def append_event(self, event: TUIEvent) -> None:
        if self._should_buffer_stream_delta(event):
            return
        self._flush_pending_stream_delta()
        self._append_event_to_state(event)
        self._refresh_widgets()

    def _append_event_to_state(self, event: TUIEvent) -> None:
        self._state = apply_tui_event(self._state, event)
        if self._record_events and self._session_recorder is not None:
            self._session_recorder.record(tui_event_to_session_event(event))

    def append_agent_event(self, event: AgentEvent) -> None:
        self.append_event(TUIEvent.from_agent_event(event))

    def set_status_model(self, model: str | None) -> None:
        self._status_model = model
        if self.is_mounted:
            self.query_one(StatusBarWidget).set_status(
                model=self._status_model,
                status=self._state.status,
                usage=self._state.usage,
            )

    def show_approval_request(self, request) -> None:
        self.query_one(ApprovalWidget).show_request(request)

    def clear_approval_request(self) -> None:
        self.query_one(ApprovalWidget).clear_request()

    def exit_command_input(self, command_input: CommandInput) -> None:
        command_input.value = ""
        command_input.blur()
        self._focus_message_panel()

    def action_toggle_debug_mode(self) -> None:
        stream = self.query_one(StreamWidget)
        selected_event_id = stream.set_debug_mode(not stream.debug_mode)
        stream.focus()
        if selected_event_id is None:
            self.query_one(DetailWidget).add_class("collapsed")
            return
        self._select_debug_event(selected_event_id)

    def action_config(self) -> None:
        self.append_event(
            TUIEvent.session_notice("Configuration is only available in live TUI runs.")
        )

    def action_session_switcher(self) -> None:
        self.open_session_selector()

    def action_metadata(self) -> None:
        self.open_metadata_screen()

    def action_trajectory(self) -> None:
        self.open_trajectory_screen()

    def open_metadata_screen(self) -> None:
        self.push_screen(MetadataScreen(self._metadata_sections()))

    def open_trajectory_screen(self) -> None:
        self.push_screen(TrajectoryScreen(self._state.events))

    def _metadata_sections(self) -> list[MetadataSection]:
        usage = self._state.usage
        lines = [
            f"session: {self._session_id or '-'}",
            f"model: {self._status_model or 'unconfigured'}",
            f"status: {self._state.status}",
            f"events: {len(self._state.events)}",
        ]
        cost_lines = [
            f"context: {_format_tokens(usage.current_context_tokens)}",
            f"session tokens: {_format_tokens(usage.session_total_tokens)}",
            f"input: {_format_tokens(usage.session_input_tokens)}",
            f"cached input: {_format_tokens(usage.session_cached_input_tokens)}",
            f"output: {_format_tokens(usage.session_output_tokens)}",
            f"session cost: {format_usd(usage.session_cost_usd)}",
        ]
        return [
            MetadataSection(title="Runtime", lines=lines),
            MetadataSection(title="Usage", lines=cost_lines),
        ]

    def on_stream_widget_event_selected(
        self,
        event: StreamWidget.EventSelected,
    ) -> None:
        self._select_debug_event(event.event_id)
        event.stop()

    def _select_debug_event(self, event_id: str) -> None:
        self._state = select_event(self._state, event_id)
        self.query_one(DetailWidget).remove_class("collapsed")
        self.query_one(DetailWidget).set_state(self._state)

    def _refresh_widgets(self) -> None:
        self.query_one(StreamWidget).set_state(self._state)
        self.query_one(DetailWidget).set_state(self._state)
        self.query_one(StatusBarWidget).set_status(
            model=self._status_model,
            status=self._state.status,
            usage=self._state.usage,
        )
        command_input = self.query_one(CommandInput)
        if self._state.status == "suspended":
            command_input.placeholder = "Choose Approve or Reject"
        else:
            command_input.placeholder = "Ask AceAI or type /quit"

    def _focus_message_panel(self) -> None:
        self.query_one(StreamWidget).focus()

    def _should_buffer_stream_delta(self, event: TUIEvent) -> bool:
        if event.kind not in ("assistant_delta", "thinking_delta"):
            return False
        if self._pending_stream_delta is not None and not _same_stream_delta(
            self._pending_stream_delta,
            event,
        ):
            self._flush_pending_stream_delta()
        if self._pending_stream_delta is None and not _same_as_last_stream_delta(
            self._state.events,
            event,
        ):
            self._append_event_to_state(event)
            self._refresh_widgets()
            return True
        self._pending_stream_delta_chars += len(event.content)
        self._pending_stream_delta = _merge_pending_stream_delta(
            self._pending_stream_delta,
            event,
        )
        if self._pending_stream_delta_chars < STREAM_DELTA_REFRESH_CHARS:
            self._ensure_pending_stream_delta_timer()
            return True
        self._flush_pending_stream_delta()
        self._refresh_widgets()
        return True

    def _ensure_pending_stream_delta_timer(self) -> None:
        if not self.is_mounted:
            return
        if self._pending_stream_delta_timer is not None:
            return
        self._pending_stream_delta_timer = self.set_timer(
            STREAM_DELTA_REFRESH_SECONDS,
            self._flush_pending_stream_delta_from_timer,
            name="stream-delta-refresh",
        )

    def _flush_pending_stream_delta_from_timer(self) -> None:
        if self._flush_pending_stream_delta():
            self._refresh_widgets()

    def _flush_pending_stream_delta(self) -> bool:
        if self._pending_stream_delta is None:
            self._clear_pending_stream_delta_timer()
            return False
        pending = self._pending_stream_delta
        self._clear_pending_stream_delta()
        self._append_event_to_state(pending)
        return True

    def _clear_pending_stream_delta(self) -> None:
        self._pending_stream_delta_chars = 0
        self._pending_stream_delta = None
        self._clear_pending_stream_delta_timer()

    def _clear_pending_stream_delta_timer(self) -> None:
        timer = self._pending_stream_delta_timer
        self._pending_stream_delta_timer = None
        if timer is not None:
            timer.stop()

    def on_unmount(self) -> None:
        self._flush_pending_stream_delta()
        if self._record_events and self._session_recorder is not None:
            self._session_recorder.finalize()


def run_static_tui(events: list[TUIEvent]) -> None:
    AceAITUI(events).run()


def _merge_pending_stream_delta(previous: TUIEvent | None, event: TUIEvent) -> TUIEvent:
    if previous is None:
        return event
    if not _same_stream_delta(previous, event):
        raise ValueError("pending stream delta changed shape before flush")
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


def _same_stream_delta(previous: TUIEvent, event: TUIEvent) -> bool:
    return (
        previous.kind == event.kind
        and previous.step_id == event.step_id
        and previous.step_index == event.step_index
        and previous.tool_call_id == event.tool_call_id
    )


def _same_as_last_stream_delta(events: list[TUIEvent], event: TUIEvent) -> bool:
    if not events:
        return False
    return _same_stream_delta(events[-1], event)


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"
