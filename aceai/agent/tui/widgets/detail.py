"""Detail pane for the read-only TUI prototype."""

import json

from rich.console import Group, RenderableType
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.events import Key, Resize
from textual.widgets import RichLog

from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.state import TUIRunState
from aceai.agent.tui.theme import EVENT_LABELS, EVENT_STYLES


class DetailWidget(RichLog):
    """Render readable details for the selected event."""

    can_focus = True

    DEFAULT_CSS = """
    DetailWidget {
        border: round #8fbcbb;
        background: #2e3440;
        color: #e5e9f0;
        padding: 1 2;
        width: 38;
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
    }
    """

    def __init__(
        self,
        state: TUIRunState | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes, wrap=True, auto_scroll=False)
        self._state = state or TUIRunState()

    def set_state(self, state: TUIRunState) -> None:
        self._state = state
        self.clear()
        self.call_after_refresh(self._render_state)

    def on_resize(self, event: Resize) -> None:
        self._render_state()

    def on_key(self, event: Key) -> None:
        if event.key in ("up", "k"):
            self.scroll_up(animate=False, force=True, immediate=True)
            event.stop()
            return
        if event.key in ("down", "j"):
            self.scroll_down(animate=False, force=True, immediate=True)
            event.stop()
            return
        if event.key == "pageup":
            self.scroll_page_up(animate=False, force=True)
            event.stop()
            return
        if event.key == "pagedown":
            self.scroll_page_down(animate=False, force=True)
            event.stop()
            return
        if event.key == "home":
            self.scroll_home(animate=False, force=True)
            event.stop()
            return
        if event.key == "end":
            self.scroll_end(animate=False, force=True)
            event.stop()

    def _render_state(self) -> None:
        self.clear()
        width = max(1, self.scrollable_content_region.width)
        self.write(self._render_detail(), width=width)
        self.call_after_refresh(self.scroll_home, animate=False)

    def _render_detail(self) -> RenderableType:
        event = _selected_event(self._state)
        if event is None:
            return "No event selected"

        renderables: list[RenderableType] = [_event_header(event)]
        _append_section(renderables, "content", _event_content(event))
        _append_section(renderables, "tool call", _tool_call_detail(event))
        _append_section(renderables, "result", _tool_result_detail(event))
        _append_section(renderables, "error", _event_error(event))
        _append_section(renderables, "agent event", _raw_event_detail(event))
        return Group(*renderables)


def _event_header(event: TUIEvent) -> RenderableType:
    table = Table.grid(expand=True)
    table.add_column(justify="left", style="bold #d8dee9", no_wrap=True, width=9)
    table.add_column(ratio=1, overflow="fold")
    table.add_row("event", Text(EVENT_LABELS[event.kind], style=EVENT_STYLES[event.kind]))
    table.add_row("kind", Text(event.kind, style="dim"))
    if event.step_index >= 0:
        table.add_row("step", f"{event.step_index + 1}")
    else:
        table.add_row("step", "session")
    table.add_row("step id", Text(_short_id(event.step_id), style="dim"))
    if event.tool_name is not None:
        table.add_row("tool", Text(event.tool_name, style="bold #ebcb8b"))
    if event.tool_call_id is not None:
        table.add_row("call id", Text(_short_id(event.tool_call_id), style="dim"))
    return table


def _append_section(
    renderables: list[RenderableType],
    title: str,
    body: RenderableType | None,
) -> None:
    if body is None:
        return
    renderables.append(Text("\n" + title.upper(), style="bold #88c0d0"))
    renderables.append(body)


def _event_content(event: TUIEvent) -> RenderableType | None:
    if event.tool_result is not None:
        return None
    if event.content == "":
        return None
    return Text(event.content)


def _tool_call_detail(event: TUIEvent) -> RenderableType | None:
    if event.tool_call is None:
        return None

    renderables: list[RenderableType] = []
    table = Table.grid(expand=True)
    table.add_column(justify="left", style="bold #d8dee9", no_wrap=True, width=9)
    table.add_column(ratio=1, overflow="fold")
    table.add_row("name", event.tool_call.name)
    table.add_row("type", Text(event.tool_call.type, style="dim"))
    table.add_row("call id", Text(_short_id(event.tool_call.call_id), style="dim"))
    renderables.append(table)
    if event.tool_call.arguments != "":
        renderables.append(Text("\narguments\n", style="bold #d8dee9"))
        renderables.append(_json_block(event.tool_call.arguments))
    return Group(*renderables)


def _tool_result_detail(event: TUIEvent) -> RenderableType | None:
    if event.tool_result is None:
        return None

    renderables: list[RenderableType] = []
    if event.tool_result.output != "":
        renderables.append(Text("output\n", style="bold #d8dee9"))
        renderables.append(Text(event.tool_result.output, style="#eceff4"))
    if event.tool_result.error is not None:
        renderables.append(Text("\nerror\n", style="bold #bf616a"))
        renderables.append(Text(event.tool_result.error, style="bold #bf616a"))
    if len(renderables) == 0:
        return None
    return Group(*renderables)


def _event_error(event: TUIEvent) -> RenderableType | None:
    if event.error is None:
        return None
    if event.tool_result is not None and event.tool_result.error == event.error:
        return None
    return Text(event.error, style="bold #bf616a")


def _raw_event_detail(event: TUIEvent) -> RenderableType | None:
    if event.raw_event is None:
        return None
    return Pretty(event.raw_event.asdict(), expand_all=False)


def _short_id(value: str) -> str:
    if len(value) <= 14:
        return value
    return f"{value[:8]}...{value[-4:]}"


def _json_block(value: str) -> RenderableType:
    payload = json.loads(value)
    formatted = json.dumps(payload, indent=2, ensure_ascii=False)
    return Syntax(
        formatted,
        "json",
        background_color="#2e3440",
        word_wrap=True,
    )


def _selected_event(state: TUIRunState) -> TUIEvent | None:
    if state.selected_event_id is None:
        return None
    for event in state.events:
        if event.event_id == state.selected_event_id:
            return event
    return None
