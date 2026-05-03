"""Main event stream pane for the read-only TUI prototype."""

from msgspec import Struct
from rich.cells import cell_len
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.events import Resize
from textual.widgets import RichLog

from aceai.agent.tui.events import TUIEvent, TUIEventKind
from aceai.agent.tui.state import TUIRunState
from aceai.agent.tui.theme import EVENT_LABELS, EVENT_STYLES

DEFAULT_CHAT_PANEL_WIDTH = 100
CHAT_PANEL_WIDTH_RATIO = 0.86


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
        overflow-x: hidden;
    }
    """

    def __init__(
        self,
        state: TUIRunState | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, wrap=True, auto_scroll=True, min_width=0)
        self._state = state or TUIRunState()

    def set_state(self, state: TUIRunState) -> None:
        self._state = state
        self.clear()
        if not self._state.events:
            self.write(Text("No events yet", style="#d8dee9"))
        else:
            for renderable in _render_events(self._state.events):
                self._write_stream_renderable(renderable)
        self.call_after_refresh(self.scroll_end, animate=False)

    def on_resize(self, event: Resize) -> None:
        if self._state.events:
            self.set_state(self._state)

    def _write_stream_renderable(self, renderable: RenderableType) -> None:
        if isinstance(renderable, Table):
            self.write(renderable, expand=True)
            return
        if _is_chat_panel(renderable):
            self.write(
                renderable,
                width=_chat_panel_width(
                    renderable,
                    _available_stream_width(self),
                ),
            )
            return
        self.write(renderable)


class _ToolBlockState(Struct, kw_only=True):
    call_id: str
    name: str | None = None
    arguments: str = ""
    output: str = ""
    status: str = "running"


def _render_events(events: list[TUIEvent]) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    assistant_buffer = ""
    assistant_buffer_step_id = ""
    thinking_buffer = ""
    assistant_step_ids: set[str] = set()
    tool_blocks: dict[str, _ToolBlockState] = {}
    rendered_tool_call_ids: set[str] = set()

    for event in events:
        if event.kind == "assistant_delta":
            assistant_buffer_step_id = event.step_id
            assistant_buffer += event.content
            continue
        if event.kind == "thinking_delta":
            thinking_buffer += event.content
            continue

        if event.tool_call_id is not None and event.kind in (
            "tool_call_delta",
            "tool_started",
            "tool_output",
            "tool_completed",
            "tool_failed",
        ):
            assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
                renderables,
                assistant_buffer,
                assistant_buffer_step_id,
                assistant_step_ids,
            )
            thinking_buffer = _flush_thinking_buffer(renderables, thinking_buffer)
            _update_tool_block(tool_blocks, event)
            if event.kind in ("tool_completed", "tool_failed"):
                renderables.append(_render_tool_block(tool_blocks[event.tool_call_id]))
                rendered_tool_call_ids.add(event.tool_call_id)
            continue

        assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
            renderables,
            assistant_buffer,
            assistant_buffer_step_id,
            assistant_step_ids,
        )
        thinking_buffer = _flush_thinking_buffer(renderables, thinking_buffer)

        if event.kind == "llm_completed" and event.step_id in assistant_step_ids:
            continue

        rendered = _render_event(event)
        if rendered is not None:
            if event.kind == "llm_completed":
                assistant_step_ids.add(event.step_id)
            renderables.append(rendered)

    assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
        renderables,
        assistant_buffer,
        assistant_buffer_step_id,
        assistant_step_ids,
    )
    thinking_buffer = _flush_thinking_buffer(renderables, thinking_buffer)
    for call_id, tool_block in tool_blocks.items():
        if call_id in rendered_tool_call_ids:
            continue
        if tool_block.name is None:
            continue
        renderables.append(_render_tool_block(tool_block))
    return renderables


def _flush_assistant_buffer(
    renderables: list[RenderableType],
    assistant_buffer: str,
    assistant_buffer_step_id: str,
    assistant_step_ids: set[str],
) -> tuple[str, str]:
    if assistant_buffer == "":
        return "", assistant_buffer_step_id
    renderables.append(
        _render_text_block(
            "assistant",
            assistant_buffer,
            event_kind="assistant_delta",
            markdown=True,
        )
    )
    assistant_step_ids.add(assistant_buffer_step_id)
    return "", ""


def _flush_thinking_buffer(
    renderables: list[RenderableType],
    thinking_buffer: str,
) -> str:
    if thinking_buffer:
        renderables.append(
            _render_text_block(
                "thinking",
                thinking_buffer,
                event_kind="thinking_delta",
            )
        )
    return ""


def _update_tool_block(
    tool_blocks: dict[str, _ToolBlockState],
    event: TUIEvent,
) -> None:
    if event.tool_call_id is None:
        raise ValueError("tool event must include tool_call_id")
    tool_block = tool_blocks.get(event.tool_call_id)
    if tool_block is None:
        tool_block = _ToolBlockState(call_id=event.tool_call_id)
        tool_blocks[event.tool_call_id] = tool_block
    if event.tool_name is not None:
        tool_block.name = event.tool_name
    if event.tool_call is not None and event.tool_call.arguments != "":
        tool_block.arguments = event.tool_call.arguments
    elif event.kind == "tool_call_delta":
        tool_block.arguments += event.content
    if event.kind == "tool_output":
        tool_block.output += event.content
    elif event.kind in ("tool_completed", "tool_failed"):
        tool_block.output = event.content
    if event.kind == "tool_failed":
        tool_block.status = "failed"
    elif event.kind == "tool_completed":
        tool_block.status = "completed"


def _render_tool_block(tool_block: _ToolBlockState) -> Panel:
    event_kind: TUIEventKind = (
        "tool_failed"
        if tool_block.status == "failed"
        else "tool_completed"
        if tool_block.status == "completed"
        else "tool_started"
    )
    style = EVENT_STYLES[event_kind]
    if tool_block.name is None:
        raise ValueError("tool block must include a tool name before rendering")
    title = f"tool: {tool_block.name}"
    return Panel(
        Text(_tool_summary(tool_block), style=style),
        title=title,
        border_style=style,
    )


def _tool_summary(tool_block: _ToolBlockState) -> str:
    if tool_block.status == "failed":
        return "failed"
    if tool_block.status == "completed":
        summary = _tool_output_summary(tool_block.output)
        if summary != "":
            return f"completed - {summary}"
        return "completed"
    return "running"


def _tool_output_summary(output: str) -> str:
    if output == "":
        return ""
    if '"entries":[' in output:
        entry_count = output.count('"name"')
        return f"{entry_count} entries"
    if '"bytes_written":' in output:
        return "file written"
    if '"exit_code":0' in output:
        return "command exited 0"
    if '"exit_code":' in output:
        return "command finished"
    if '"matches":' in output:
        return "search finished"
    return "result ready"


def _render_text_block(
    label: str,
    content: str,
    *,
    event_kind: TUIEventKind,
    markdown: bool = False,
) -> Panel:
    style = EVENT_STYLES[event_kind]
    body: RenderableType
    if markdown:
        body = Markdown(content, style=style)
    else:
        body = Text(content, style=style)
    return Panel(
        body,
        border_style=style,
        expand=False,
    )


def _render_event(event: TUIEvent) -> RenderableType | None:
    label = EVENT_LABELS[event.kind]
    style = EVENT_STYLES[event.kind]
    if event.kind == "user_message":
        return _render_user_message(event.content, label=label, event_kind=event.kind)
    if event.kind == "session_notice":
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind == "tool_call_delta":
        return None
    if event.kind == "assistant_delta":
        return _render_text_block(
            label,
            event.content,
            event_kind=event.kind,
            markdown=True,
        )
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
        return _render_text_block(
            "assistant",
            event.content,
            event_kind="assistant_delta",
            markdown=True,
        )
    if event.kind in ("step_started", "step_completed", "run_completed"):
        return None
    return Text(f"{label}: {event.content}", style=style)


def _tool_title(label: str, event: TUIEvent) -> str:
    if event.tool_name is not None:
        return f"{label}: {event.tool_name}"
    if event.tool_call_id is not None:
        return f"{label}: {event.tool_call_id}"
    return label


def _render_user_message(
    content: str,
    *,
    label: str,
    event_kind: TUIEventKind,
) -> Table:
    style = EVENT_STYLES[event_kind]
    row = Table.grid(expand=True)
    row.add_column(ratio=1)
    row.add_column()
    row.add_row(
        "",
        Panel(
            Text(content, style=style),
            border_style=style,
            expand=False,
        ),
    )
    return row


def _is_chat_panel(renderable: RenderableType) -> bool:
    return isinstance(renderable, Panel) and renderable.title is None


def _chat_panel_width(panel: Panel, available_width: int) -> int:
    if available_width <= 0:
        return min(_natural_panel_width(panel), DEFAULT_CHAT_PANEL_WIDTH)
    max_width = max(24, min(available_width, int(available_width * CHAT_PANEL_WIDTH_RATIO)))
    return min(_natural_panel_width(panel), max_width)


def _available_stream_width(stream: StreamWidget) -> int:
    widths = (
        stream.scrollable_content_region.width,
        stream.content_size.width,
        stream.size.width,
    )
    for width in widths:
        if width > 0:
            return width
    return 0


def _natural_panel_width(panel: Panel) -> int:
    body = panel.renderable
    if isinstance(body, Text):
        content = body.plain
    elif isinstance(body, Markdown):
        content = body.markup
    else:
        return 80
    widest_line = 1
    for line in content.splitlines():
        widest_line = max(widest_line, cell_len(line))
    return widest_line + 4
