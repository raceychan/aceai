"""Read-only Textual application for AceAI event streams."""

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from aceai.core.events import AgentEvent

from .events import TUIEvent
from .events import adapt_agent_event
from .state import TUIRunState, apply_tui_event, initial_state, reduce_events
from .widgets import CommandInput, DetailWidget, StreamWidget, TimelineWidget


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
    ]

    def __init__(self, events: list[TUIEvent] | None = None) -> None:
        super().__init__()
        self._events = list(events or [])
        self._state: TUIRunState = initial_state()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield TimelineWidget(id="timeline", classes="collapsed")
            yield StreamWidget(id="stream")
            yield DetailWidget(id="detail", classes="collapsed")
        yield CommandInput(id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.load_events(self._events)

    def load_events(self, events: list[TUIEvent]) -> None:
        self._state = reduce_events(events)
        self._refresh_widgets()

    def append_event(self, event: TUIEvent) -> None:
        self._state = apply_tui_event(self._state, event)
        self._refresh_widgets()

    def append_agent_event(self, event: AgentEvent) -> None:
        self.append_event(adapt_agent_event(event))

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

    def _refresh_widgets(self) -> None:
        self.query_one(TimelineWidget).set_state(self._state)
        self.query_one(StreamWidget).set_state(self._state)
        self.query_one(DetailWidget).set_state(self._state)


def run_static_tui(events: list[TUIEvent]) -> None:
    AceAITUI(events).run()
