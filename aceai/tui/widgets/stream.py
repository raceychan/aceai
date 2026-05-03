"""Main event stream pane for the read-only TUI prototype."""

from rich.console import RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.widgets import RichLog

from aceai.tui.events import TUIEvent, TUIEventKind
from aceai.tui.state import TUIRunState
from aceai.tui.theme import EVENT_LABELS, EVENT_STYLES


class StreamWidget(RichLog):
    """Render the readable event transcript."""

    DEFAULT_CSS = """
    StreamWidget {
        border: solid #81a1c1;
        background: #2e3440;
        color: #e5e9f0;
        padding: 0 1;
        width: 1fr;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(
        self,
        state: TUIRunState | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, wrap=True, auto_scroll=True)
        self._state = state or TUIRunState()

    def set_state(self, state: TUIRunState) -> None:
        self._state = state
        self.clear()
        if not self._state.events:
            self.write(Text("No events yet", style="#d8dee9"))
        else:
            for renderable in _render_events(self._state.events):
                self.write(renderable)
        self.call_after_refresh(self.scroll_end, animate=False)

def _render_events(events: list[TUIEvent]) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    assistant_buffer = ""
    assistant_buffer_step_id = ""
    thinking_buffer = ""
    assistant_step_ids: set[str] = set()

    for event in events:
        if event.kind == "assistant_delta":
            assistant_buffer_step_id = event.step_id
            assistant_buffer += event.content
            continue
        if event.kind == "thinking_delta":
            thinking_buffer += event.content
            continue

        if assistant_buffer:
            renderables.append(
                _render_text_block(
                    "assistant",
                    assistant_buffer,
                    event_kind="assistant_delta",
                )
            )
            assistant_step_ids.add(assistant_buffer_step_id)
            assistant_buffer = ""
            assistant_buffer_step_id = ""
        if thinking_buffer:
            renderables.append(
                _render_text_block(
                    "thinking",
                    thinking_buffer,
                    event_kind="thinking_delta",
                )
            )
            thinking_buffer = ""

        if event.kind == "llm_completed" and event.step_id in assistant_step_ids:
            continue

        rendered = _render_event(event)
        if rendered is not None:
            if event.kind == "llm_completed":
                assistant_step_ids.add(event.step_id)
            renderables.append(rendered)

    if assistant_buffer:
        renderables.append(
            _render_text_block(
                "assistant",
                assistant_buffer,
                event_kind="assistant_delta",
            )
        )
        assistant_step_ids.add(assistant_buffer_step_id)
    if thinking_buffer:
        renderables.append(
            _render_text_block(
                "thinking",
                thinking_buffer,
                event_kind="thinking_delta",
            )
        )
    return renderables


def _render_text_block(
    label: str,
    content: str,
    *,
    event_kind: TUIEventKind,
) -> Panel:
    style = EVENT_STYLES[event_kind]
    return Panel(
        Text(content, style=style),
        title=label,
        border_style=style,
    )


def _render_event(event: TUIEvent) -> RenderableType | None:
    label = EVENT_LABELS[event.kind]
    style = EVENT_STYLES[event.kind]
    if event.kind == "user_message":
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind == "tool_call_delta":
        return Panel(
            Syntax(event.content, "json", word_wrap=True),
            title=_tool_title("tool call", event),
            border_style=style,
        )
    if event.kind == "assistant_delta":
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind in ("thinking_delta", "reasoning_summary"):
        return Panel(
            Text(event.content, style=style),
            title=label,
            border_style=style,
        )
    if event.kind in ("tool_started", "tool_completed", "tool_failed", "tool_output"):
        if event.kind == "tool_started":
            return None
        title = _tool_title(label, event)
        return Panel(
            Text(event.content or event.tool_call_id or "", style=style),
            title=title,
            border_style=style,
        )
    if event.kind == "media":
        return Panel(
            Text("media segment", style=style),
            title=label,
            border_style=style,
        )
    if event.kind == "llm_completed":
        if event.content == "":
            return None
        return _render_text_block("assistant", event.content, event_kind="assistant_delta")
    if event.kind in ("step_started", "step_completed", "run_completed"):
        return None
    return Text(f"{label}: {event.content}", style=style)


def _tool_title(label: str, event: TUIEvent) -> str:
    if event.tool_name is not None:
        return f"{label}: {event.tool_name}"
    if event.tool_call_id is not None:
        return f"{label}: {event.tool_call_id}"
    return label
