"""Timeline pane for the read-only TUI prototype."""

from rich.text import Text
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from aceai.agent.tui.state import TUIRunState, TUIStepState, TUIToolState


class TimelineWidget(OptionList):
    """Render a compact step and tool timeline."""

    class EventSelected(Message):
        def __init__(self, event_id: str) -> None:
            self.event_id = event_id
            super().__init__()

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
        super().__init__(id=id, classes=classes)
        self._state = state or TUIRunState()
        self._option_event_ids: dict[str, str] = {}

    def set_state(self, state: TUIRunState) -> None:
        self._state = state
        options = _timeline_options(state)
        self._option_event_ids = options.event_ids
        self.set_options(options.items)
        self.call_after_refresh(self.scroll_end, animate=False)

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        if event.option_id is None:
            raise ValueError("timeline option must include an event id")
        self.post_message(self.EventSelected(self._option_event_ids[event.option_id]))


class _TimelineOptions:
    def __init__(self, items: list[Option], event_ids: dict[str, str]) -> None:
        self.items = items
        self.event_ids = event_ids


def _timeline_options(state: TUIRunState) -> _TimelineOptions:
    options: list[Option] = [Option(Text("timeline", style="bold #eceff4"), disabled=True)]
    event_ids: dict[str, str] = {}
    for step in state.steps:
        step_option_id = _row_option_id("step", len(event_ids))
        options.append(Option(_step_text(step), id=step_option_id))
        event_ids[step_option_id] = _step_event_id(step)
        for tool_state in step.tools:
            tool_option_id = _row_option_id("tool", len(event_ids))
            options.append(Option(_tool_text(tool_state), id=tool_option_id))
            event_ids[tool_option_id] = _tool_event_id(tool_state)
    if not state.steps:
        options.append(Option(Text("No events yet", style="#d8dee9"), disabled=True))
    return _TimelineOptions(options, event_ids)


def _row_option_id(kind: str, index: int) -> str:
    return f"{kind}-{index}"


def _step_event_id(step: TUIStepState) -> str:
    if not step.events:
        raise ValueError("timeline step must include events")
    return step.events[-1].event_id


def _tool_event_id(tool_state: TUIToolState) -> str:
    if not tool_state.events:
        raise ValueError("timeline tool must include events")
    return tool_state.events[-1].event_id


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
