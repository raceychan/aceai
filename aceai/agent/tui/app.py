"""Read-only Textual application for AceAI event streams."""

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.events import Key
from textual.widgets import Footer, Header

from aceai.core.events import AgentEvent

from aceai.agent.session import SessionRecorder

from aceai.agent.cost import format_usd
from .events import TUIEvent
from .session_adapter import tui_event_to_session_event
from .session_display import session_display_title
from .session_replay import event_log_to_tui_events
from .setup import SessionSelectScreen
from .state import TUIRunState, apply_tui_event, initial_state, reduce_events
from .widgets import (
    CommandInput,
    DetailWidget,
    StatusBarWidget,
    StreamWidget,
    TimelineWidget,
)

STREAM_DELTA_REFRESH_CHARS = 512


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

    TimelineWidget.collapsed {
        display: none;
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
        ("e", "toggle_events", "Events"),
        ("d", "toggle_detail", "Raw Log"),
        ("m", "model_switcher", "Model"),
        ("s", "session_switcher", "Sessions"),
    ]

    def __init__(
        self,
        events: list[TUIEvent] | None = None,
        *,
        model: str | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self._events = list(events or [])
        self._state: TUIRunState = initial_state()
        self._status_model = model
        self._session_recorder = session_recorder
        self._session_id = session_id
        self._pending_stream_delta_chars = 0
        self._pending_stream_delta: TUIEvent | None = None
        self.title = "AceAI" if session_id is None else f"AceAI {session_id}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield TimelineWidget(id="timeline", classes="collapsed")
            yield StreamWidget(id="stream")
            yield DetailWidget(id="detail", classes="collapsed")
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
        self._state = reduce_events(events)
        self._refresh_widgets()

    def show_sessions(self) -> None:
        if self._session_recorder is None:
            self.append_event(TUIEvent.session_notice("No session store is configured."))
            return
        sessions = self._session_recorder.store.list_sessions()
        if not sessions:
            self.append_event(TUIEvent.session_notice("No sessions found."))
            return
        total_cost = self._session_recorder.store.total_cost_usd()
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
        if self._session_recorder is None:
            self.append_event(TUIEvent.session_notice("No session store is configured."))
            return
        sessions = self._session_recorder.store.list_sessions()
        if not sessions:
            self.append_event(TUIEvent.session_notice("No sessions found."))
            return
        self.push_screen(
            SessionSelectScreen(
                store=self._session_recorder.store,
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
        if self._session_recorder is None:
            self.append_event(TUIEvent.session_notice("No session store is configured."))
            return
        self._session_recorder.finalize()
        store = self._session_recorder.store
        metadata = store.get_session(session_id)
        self._session_recorder = SessionRecorder(store, metadata.session_id)
        self._session_id = metadata.session_id
        self.title = f"AceAI {metadata.session_id}"
        event_log = store.load_event_log(metadata.session_id)
        self.load_events(event_log_to_tui_events(event_log))
        self.append_event(TUIEvent.session_notice(f"Resumed session {metadata.session_id}"))

    def append_event(self, event: TUIEvent) -> None:
        if self._should_buffer_stream_delta(event):
            return
        self._flush_pending_stream_delta()
        self._append_event_to_state(event)
        self._refresh_widgets()

    def _append_event_to_state(self, event: TUIEvent) -> None:
        self._state = apply_tui_event(self._state, event)
        if self._session_recorder is not None:
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

    def exit_command_input(self, command_input: CommandInput) -> None:
        command_input.value = ""
        command_input.blur()
        self.query_one("#stream").focus()

    def action_toggle_detail(self) -> None:
        detail = self.query_one(DetailWidget)
        if detail.has_class("collapsed"):
            detail.remove_class("collapsed")
            detail.focus()
        else:
            detail.add_class("collapsed")
            self.query_one(StreamWidget).focus()

    def action_toggle_events(self) -> None:
        timeline = self.query_one(TimelineWidget)
        if timeline.has_class("collapsed"):
            timeline.remove_class("collapsed")
            timeline.focus()
        else:
            timeline.add_class("collapsed")
            self.query_one(StreamWidget).focus()

    def action_model_switcher(self) -> None:
        self.append_event(
            TUIEvent.session_notice("Model selection is only available in live TUI runs.")
        )

    def action_session_switcher(self) -> None:
        self.open_session_selector()

    def _refresh_widgets(self) -> None:
        self.query_one(TimelineWidget).set_state(self._state)
        self.query_one(StreamWidget).set_state(self._state)
        self.query_one(DetailWidget).set_state(self._state)
        self.query_one(StatusBarWidget).set_status(
            model=self._status_model,
            status=self._state.status,
            usage=self._state.usage,
        )

    def _should_buffer_stream_delta(self, event: TUIEvent) -> bool:
        if event.kind not in ("assistant_delta", "thinking_delta"):
            return False
        if self._pending_stream_delta is not None and not _same_stream_delta(
            self._pending_stream_delta,
            event,
        ):
            self._flush_pending_stream_delta()
        self._pending_stream_delta_chars += len(event.content)
        self._pending_stream_delta = _merge_pending_stream_delta(
            self._pending_stream_delta,
            event,
        )
        if self._pending_stream_delta_chars < STREAM_DELTA_REFRESH_CHARS:
            return True
        self._flush_pending_stream_delta()
        self._refresh_widgets()
        return True

    def _flush_pending_stream_delta(self) -> None:
        if self._pending_stream_delta is None:
            return
        pending = self._pending_stream_delta
        self._clear_pending_stream_delta()
        self._append_event_to_state(pending)

    def _clear_pending_stream_delta(self) -> None:
        self._pending_stream_delta_chars = 0
        self._pending_stream_delta = None

    def on_unmount(self) -> None:
        self._flush_pending_stream_delta()
        if self._session_recorder is not None:
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
