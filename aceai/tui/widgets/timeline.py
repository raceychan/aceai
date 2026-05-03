"""Timeline pane for the read-only TUI prototype."""

from rich.table import Table
from rich.text import Text
from textual.containers import ScrollableContainer

from aceai.tui.state import TUIRunState, TUIStepState, TUIToolState


class TimelineWidget(ScrollableContainer):
    """Render a compact step and tool timeline."""

    DEFAULT_CSS = """
    TimelineWidget {
        border: solid #5e81ac;
        background: #2e3440;
        color: #e5e9f0;
        padding: 0 1;
        width: 28;
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

    def render(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_row(Text("timeline", style="bold #eceff4"))
        for step in self._state.steps:
            table.add_row(_step_text(step))
            for tool_state in step.tools:
                table.add_row(_tool_text(tool_state))
        if not self._state.steps:
            table.add_row(Text("No events yet", style="#d8dee9"))
        return table


def _step_text(step: TUIStepState) -> Text:
    marker = "x" if step.status == "failed" else "*" if step.status == "running" else "+"
    style = "#bf616a" if step.status == "failed" else "#88c0d0" if step.status == "running" else "#a3be8c"
    text = Text()
    text.append(f"{marker} Step {step.step_index + 1}", style=style)
    text.append(f"  {step.status}", style="#d8dee9")
    return text


def _tool_text(tool_state: TUIToolState) -> Text:
    marker = "x" if tool_state.status == "failed" else "*" if tool_state.status == "running" else "+"
    style = "#bf616a" if tool_state.status == "failed" else "#ebcb8b" if tool_state.status == "running" else "#a3be8c"
    name = tool_state.name or tool_state.call_id
    text = Text("  ")
    text.append(f"{marker} {name}", style=style)
    text.append(f"  {tool_state.status}", style="#d8dee9")
    return text
