"""Read-only Textual application for AceAI event streams."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import Key
from textual.timer import Timer
from textual.widgets import TextArea

from aceai import __version__
from aceai.core.events import AgentEvent

from aceai.agent.ideas import IdeaStore
from aceai.agent.project import ProjectMetadata, default_project
from aceai.agent.session import SessionRecorder, SessionStore

from aceai.agent.cost import format_usd
from aceai.agent.provider_catalog import (
    context_window_for_model_any_provider,
    supports_reasoning_effort_any_provider,
)
from .events import TUIEvent
from .metadata import MetadataScreen, MetadataSection
from .session_adapter import tui_event_to_session_event
from .session_display import session_display_title
from .session_replay import event_log_to_tui_events
from .setup import SessionSelectScreen
from .state import (
    TUIRunState,
    apply_tui_event,
    initial_state,
    reduce_events,
    reset_cache_rate,
    select_event,
)
from .trajectory import TrajectoryScreen
from .widgets import (
    ApprovalWidget,
    CommandCompletionItem,
    CommandCompletionWidget,
    CommandInput,
    CitationPreviewWidget,
    DetailWidget,
    QueuedTurnsWidget,
    StatusBarWidget,
    StreamWidget,
    SubagentStatusWidget,
    TopBarWidget,
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

    #main {
        height: 1fr;
    }

    #stream {
        min-width: 40;
    }

    DetailWidget.collapsed {
        display: none;
    }

    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit"),
        Binding("d", "toggle_debug_mode", "Debug"),
        Binding("c", "config", "Config"),
        Binding("i", "ideas", "Ideas"),
        Binding("s", "session_switcher", "Sessions"),
    ]

    def __init__(
        self,
        events: list[TUIEvent] | None = None,
        *,
        model: str | None = None,
        reasoning_level: str = "auto",
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        idea_store: IdeaStore | None = None,
        project: ProjectMetadata | None = None,
        record_events: bool = True,
    ) -> None:
        super().__init__()
        self._events = list(events or [])
        self._state: TUIRunState = initial_state()
        self._status_model = model
        self._status_reasoning_level = reasoning_level
        self._session_recorder = session_recorder
        self._session_id = session_id
        self._project = project or (
            session_recorder.store.project
            if session_recorder is not None
            else default_project()
        )
        self._idea_store = idea_store or IdeaStore()
        self._record_events = record_events
        self._pending_stream_delta_chars = 0
        self._pending_stream_delta: TUIEvent | None = None
        self._pending_stream_delta_timer: Timer | None = None
        self._command_completion_selected_index = 0
        self.title = self._window_title()

    def compose(self) -> ComposeResult:
        yield TopBarWidget(id="topbar")
        with Horizontal(id="main"):
            yield StreamWidget(id="stream", project_name=self._project.name)
            yield SubagentStatusWidget(id="subagents", classes="hidden")
            yield DetailWidget(id="detail", classes="collapsed")
        yield ApprovalWidget(id="approval", classes="collapsed")
        yield StatusBarWidget(id="status")
        yield CommandCompletionWidget(id="command-completions", classes="hidden")
        yield QueuedTurnsWidget(id="queued-turns", classes="hidden")
        yield CitationPreviewWidget(id="citation-preview", classes="hidden")
        yield CommandInput(id="input")

    def on_mount(self) -> None:
        self.load_events(self._events)
        self.query_one(StreamWidget).focus()

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and self.cancel_active_run():
            event.stop()
            return
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
                f"{session.project_name}  "
                f"{session_display_title(session.title)}  {session.updated_at}"
            )
        lines.append("")
        lines.append("Use the session picker to switch sessions.")
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
            self.notify_session(f"Session not found: {session_id}")
            return
        if self._session_recorder is not None:
            self._session_recorder.finalize()
        self._session_recorder = SessionRecorder(store, metadata.session_id)
        self._session_id = metadata.session_id
        self.title = self._window_title()
        event_log = store.load_event_log(metadata.session_id)
        self.load_events(event_log_to_tui_events(event_log))
        self.notify_session(f"Resumed session {metadata.session_id}")

    def ensure_session(self) -> None:
        if self._session_recorder is not None:
            return
        store = SessionStore(project=self._project)
        metadata = store.create_session()
        self._session_recorder = SessionRecorder(store, metadata.session_id)
        self._session_id = metadata.session_id
        self.title = self._window_title()
        if self.is_mounted:
            self.query_one(TopBarWidget).set_title(self.title)

    def _session_store(self) -> SessionStore:
        if self._session_recorder is not None:
            return self._session_recorder.store
        return SessionStore(project=self._project)

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
        tui_event = TUIEvent.from_agent_event(event)
        if tui_event.kind == "llm_retrying":
            self.notify(
                tui_event.content,
                title=_retrying_title(tui_event),
                severity="warning",
                timeout=3.0,
            )
        self.append_event(tui_event)

    def notify_session(self, content: str) -> None:
        self.notify(content, title="AceAI", severity="information", timeout=3.0)

    def cancel_active_run(self) -> bool:
        return False

    def set_status_model(
        self,
        model: str | None,
        *,
        reasoning_level: str | None = None,
    ) -> None:
        self._status_model = model
        if reasoning_level is not None:
            self._status_reasoning_level = reasoning_level
        if self.is_mounted:
            self.query_one(StatusBarWidget).set_status(
                model=self._status_model,
                reasoning_level=self._status_reasoning_level,
                status=self._state.status,
                usage=self._state.usage,
            )

    def reset_status_cache_rate(self) -> None:
        self._state = reset_cache_rate(self._state)
        if self.is_mounted:
            self.query_one(StatusBarWidget).set_status(
                model=self._status_model,
                reasoning_level=self._status_reasoning_level,
                status=self._state.status,
                usage=self._state.usage,
            )

    def on_top_bar_widget_quit_requested(
        self,
        event: TopBarWidget.QuitRequested,
    ) -> None:
        event.stop()
        self.exit()

    def on_top_bar_widget_debug_requested(
        self,
        event: TopBarWidget.DebugRequested,
    ) -> None:
        event.stop()
        self.action_toggle_debug_mode()

    def on_top_bar_widget_config_requested(
        self,
        event: TopBarWidget.ConfigRequested,
    ) -> None:
        event.stop()
        self.action_config()

    def on_status_bar_widget_metadata_requested(
        self,
        event: StatusBarWidget.MetadataRequested,
    ) -> None:
        event.stop()
        self.open_metadata_screen()

    def show_approval_request(self, request) -> None:
        self.query_one(ApprovalWidget).show_request(request)

    def clear_approval_request(self) -> None:
        self.query_one(ApprovalWidget).clear_request()

    def exit_command_input(self, command_input: CommandInput) -> None:
        command_input.value = ""
        self._hide_command_completions()
        command_input.blur()
        self._focus_message_panel()

    def command_names(self) -> tuple[str, ...]:
        return ()

    def command_completion_items(self) -> tuple[CommandCompletionItem, ...]:
        return tuple(
            CommandCompletionItem(command=name, description="Run command")
            for name in self.command_names()
        )

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not isinstance(event.text_area, CommandInput):
            return
        self._command_completion_selected_index = 0
        self._refresh_command_completions(event.text_area.value)

    def on_command_input_completion_requested(
        self,
        event: CommandInput.CompletionRequested,
    ) -> None:
        matches = self._matching_command_items(event.input.value)
        if not matches:
            return
        index = min(self._command_completion_selected_index, len(matches) - 1)
        event.input.value = f"/{matches[index].command} "
        self._command_completion_selected_index = 0
        self._refresh_command_completions(event.input.value)

    def on_command_input_completion_navigation_requested(
        self,
        event: CommandInput.CompletionNavigationRequested,
    ) -> None:
        command_input = self.query_one(CommandInput)
        matches = self._matching_command_items(command_input.value)
        if not matches:
            return
        self._command_completion_selected_index = (
            self._command_completion_selected_index + event.direction
        ) % len(matches)
        self._refresh_command_completions(command_input.value)
        event.stop()

    def _refresh_command_completions(self, value: str) -> None:
        completions = self._command_completion_widget()
        if completions is None:
            return
        matches = self._matching_command_items(value)
        if not matches:
            completions.hide()
            return
        if self._command_completion_selected_index >= len(matches):
            self._command_completion_selected_index = 0
        completions.show_commands(
            list(matches),
            selected_index=self._command_completion_selected_index,
        )

    def _matching_command_items(self, value: str) -> tuple[CommandCompletionItem, ...]:
        if not value.startswith("/"):
            return ()
        body = value.removeprefix("/")
        if " " in body or "\n" in body:
            return ()
        return tuple(
            item
            for item in self.command_completion_items()
            if item.command.startswith(body)
        )

    def _hide_command_completions(self) -> None:
        completions = self._command_completion_widget()
        if completions is None:
            return
        completions.hide()

    def _command_completion_widget(self) -> CommandCompletionWidget | None:
        matches = list(self.query(CommandCompletionWidget))
        if not matches:
            return None
        return matches[0]

    def action_toggle_debug_mode(self) -> None:
        if self._state.subagents:
            self.notify_session("Debug mode is unavailable while subagents are visible.")
            return
        if (
            not self.query_one(StreamWidget).debug_mode
            and self._state.status != "completed"
        ):
            self.notify_session("Debug mode is available after the run completes.")
            return
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

    def action_ideas(self) -> None:
        self.append_event(
            TUIEvent.session_notice("Ideas are only available in live TUI runs.")
        )

    def action_trajectory(self) -> None:
        self.open_trajectory_screen()

    def open_metadata_screen(self) -> None:
        self.push_screen(MetadataScreen(self._metadata_sections()))

    def open_trajectory_screen(self) -> None:
        self.push_screen(TrajectoryScreen(self._state.events))

    def _metadata_sections(self) -> list[MetadataSection]:
        usage = self._state.usage
        max_ctx = (
            context_window_for_model_any_provider(self._status_model)
            if self._status_model
            else None
        )
        ctx_pct = _context_window_pct(usage.current_context_tokens, max_ctx)
        lines = [
            f"session: {self._session_id or '-'}",
            f"project: {self._project.name}",
            f"project_id: {self._project.project_id}",
            f"version: {__version__}",
            f"model: {self._status_model or 'unconfigured'}",
            f"status: {self._state.status}",
            f"events: {len(self._state.events)}",
        ]
        if self._status_model is not None and supports_reasoning_effort_any_provider(
            self._status_model
        ):
            lines.insert(5, f"reasoning: {self._status_reasoning_level}")
        cost_lines = [
            f"context: {_format_tokens(usage.current_context_tokens)}{ctx_pct}",
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
        if self._state.subagents or self._state.status != "completed":
            event.stop()
            return
        self._select_debug_event(event.event_id)
        event.stop()

    def _select_debug_event(self, event_id: str) -> None:
        self._state = select_event(self._state, event_id)
        self.query_one(DetailWidget).remove_class("collapsed")
        self.query_one(DetailWidget).set_state(self._state)

    def _refresh_widgets(self) -> None:
        self.query_one(TopBarWidget).set_title(self.title)
        stream = self.query_one(StreamWidget)
        stream.set_project_name(self._project.name)
        stream.set_state(self._state)
        self.query_one(DetailWidget).set_state(self._state)
        self.query_one(SubagentStatusWidget).set_subagents(self._state.subagents)
        self._sync_side_layout()
        self.query_one(StatusBarWidget).set_status(
            model=self._status_model,
            reasoning_level=self._status_reasoning_level,
            status=self._state.status,
            usage=self._state.usage,
        )
        command_input = self.query_one(CommandInput)
        if self._state.status == "suspended":
            command_input.placeholder = "Choose Approve or Reject"
        else:
            command_input.placeholder = "Ask AceAI or type /quit"

    def _sync_side_layout(self) -> None:
        stream = self.query_one(StreamWidget)
        detail = self.query_one(DetailWidget)
        has_subagents = len(self._state.subagents) > 0
        if has_subagents:
            if stream.debug_mode:
                stream.set_debug_mode(False)
            detail.add_class("collapsed")
            return

    def _window_title(self) -> str:
        if self._session_id is None:
            return f"AceAI {self._project.name}"
        return f"AceAI {self._project.name} {self._session_id}"

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
        run_id=previous.run_id,
        retry_count=previous.retry_count,
        retry_max=previous.retry_max,
        retry_delay_seconds=previous.retry_delay_seconds,
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


def _retrying_title(event: TUIEvent) -> str:
    title = f"Retrying message {event.retry_count}/{event.retry_max}"
    if event.run_id == "":
        return title
    return f"{title} · {_short_id(event.run_id)}"


def _short_id(value: str) -> str:
    if len(value) <= 8:
        return value
    return value[:8]


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _context_window_pct(
    current_tokens: int | None,
    max_window: int | None,
) -> str:
    if current_tokens is None or max_window is None or max_window == 0:
        return ""
    pct = current_tokens / max_window * 100
    return f" ({pct:.0f}%)"
