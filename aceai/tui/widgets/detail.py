"""Detail pane for the read-only TUI prototype."""

from rich.pretty import Pretty
from rich.table import Table
from textual.containers import ScrollableContainer

from aceai.tui.events import TUIEvent
from aceai.tui.state import TUIRunState


class DetailWidget(ScrollableContainer):
    """Render raw details for the selected event."""

    DEFAULT_CSS = """
    DetailWidget {
        border: solid #8fbcbb;
        background: #2e3440;
        color: #e5e9f0;
        padding: 0 1;
        width: 38;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(
        self,
        state: TUIRunState | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes, can_focus=True)
        self._state = state or TUIRunState()

    def set_state(self, state: TUIRunState) -> None:
        self._state = state
        self.refresh()
        self.call_after_refresh(self.scroll_end, animate=False)

    def render(self):
        event = _selected_event(self._state)
        if event is None:
            return "No event selected"

        table = Table.grid(expand=True)
        table.add_column(justify="left", style="bold")
        table.add_column(ratio=1)
        table.add_row("kind", event.kind)
        table.add_row("step", f"{event.step_index + 1}")
        table.add_row("step_id", event.step_id)
        if event.tool_name is not None:
            table.add_row("tool", event.tool_name)
        if event.tool_call_id is not None:
            table.add_row("call_id", event.tool_call_id)
        if event.error is not None:
            table.add_row("error", event.error)
        raw = None if event.raw_event is None else event.raw_event.asdict()
        table.add_row("raw", Pretty(raw))
        return table


def _selected_event(state: TUIRunState) -> TUIEvent | None:
    if state.selected_event_id is None:
        return None
    for event in state.events:
        if event.event_id == state.selected_event_id:
            return event
    return None
