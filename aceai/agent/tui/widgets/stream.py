"""Main event stream pane for the read-only TUI prototype."""

from msgspec import Struct
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.events import Resize
from textual.widgets import RichLog

from aceai.agent.tui.events import TUIEvent, TUIEventKind
from aceai.agent.tui.state import TUIRunState
from aceai.agent.tui.theme import EVENT_LABELS, EVENT_STYLES

PROMPT_BAR_STYLE = "bold #eceff4 on #3b4252"
PROMPT_MARK_STYLE = "bold #88c0d0 on #3b4252"
SUBTLE_BULLET_STYLE = "bold #9aa3b2"
REASONING_MARK_STYLE = "bold #d08770"
TRANSCRIPT_GUTTER = "  "


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
        self.write(renderable, width=_available_stream_width(self))


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
    pending_reasoning: dict[str, list[TUIEvent]] = {}
    rendered_reasoning_step_ids: set[str] = set()
    tool_blocks: dict[str, _ToolBlockState] = {}
    rendered_tool_call_ids: set[str] = set()

    for event in events:
        if event.kind == "reasoning_summary":
            pending_reasoning.setdefault(event.step_id, []).append(event)
            continue
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
            thinking_buffer = _flush_thinking_buffer(renderables, thinking_buffer)
            _flush_pending_reasoning(
                renderables,
                assistant_buffer_step_id,
                pending_reasoning,
                rendered_reasoning_step_ids,
            )
            assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
                renderables,
                assistant_buffer,
                assistant_buffer_step_id,
                assistant_step_ids,
            )
            _update_tool_block(tool_blocks, event)
            if event.kind in ("tool_completed", "tool_failed"):
                renderables.append(_render_tool_block(tool_blocks[event.tool_call_id]))
                rendered_tool_call_ids.add(event.tool_call_id)
            continue

        if event.kind == "llm_completed":
            _flush_pending_reasoning(
                renderables,
                event.step_id,
                pending_reasoning,
                rendered_reasoning_step_ids,
            )
        thinking_buffer = _flush_thinking_buffer(renderables, thinking_buffer)
        assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
            renderables,
            assistant_buffer,
            assistant_buffer_step_id,
            assistant_step_ids,
        )

        if event.kind == "llm_completed" and event.step_id in assistant_step_ids:
            continue

        rendered = _render_event(event)
        if rendered is not None:
            _flush_pending_reasoning(
                renderables,
                event.step_id,
                pending_reasoning,
                rendered_reasoning_step_ids,
            )
            if event.kind == "user_message" and renderables:
                renderables.append(Text(""))
            if event.kind == "llm_completed":
                assistant_step_ids.add(event.step_id)
            renderables.append(rendered)

    thinking_buffer = _flush_thinking_buffer(renderables, thinking_buffer)
    assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
        renderables,
        assistant_buffer,
        assistant_buffer_step_id,
        assistant_step_ids,
    )
    for step_id in pending_reasoning:
        _flush_pending_reasoning(
            renderables,
            step_id,
            pending_reasoning,
            rendered_reasoning_step_ids,
        )
    for call_id, tool_block in tool_blocks.items():
        if call_id in rendered_tool_call_ids:
            continue
        if tool_block.name is None:
            continue
        renderables.append(_render_tool_block(tool_block))
    return renderables


def _flush_pending_reasoning(
    renderables: list[RenderableType],
    step_id: str,
    pending_reasoning: dict[str, list[TUIEvent]],
    rendered_reasoning_step_ids: set[str],
) -> None:
    if step_id in rendered_reasoning_step_ids:
        return
    events = pending_reasoning.get(step_id)
    if events is None:
        return
    for event in events:
        rendered = _render_event(event)
        if rendered is not None:
            renderables.append(rendered)
    rendered_reasoning_step_ids.add(step_id)


def _flush_assistant_buffer(
    renderables: list[RenderableType],
    assistant_buffer: str,
    assistant_buffer_step_id: str,
    assistant_step_ids: set[str],
) -> tuple[str, str]:
    if assistant_buffer == "":
        return "", assistant_buffer_step_id
    renderables.append(
        _render_assistant_block(assistant_buffer)
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
                "reasoning",
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


def _render_tool_block(tool_block: _ToolBlockState) -> Text:
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
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    text.append("●", style=SUBTLE_BULLET_STYLE)
    text.append(" ")
    text.append(tool_block.name, style=style)
    text.append(f"  {_tool_summary(tool_block)}", style=style)
    return text


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


def _render_assistant_block(content: str) -> RenderableType:
    style = EVENT_STYLES["assistant_delta"]
    if _looks_like_markdown(content):
        return Group(
            Text(""),
            Markdown(content, style=style),
        )
    lines = content.splitlines()
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    if not lines:
        return text
    text.append(lines[0], style=style)
    for line in lines[1:]:
        text.append("\n  ")
        text.append(line, style=style)
    return text


def _render_text_block(
    label: str,
    content: str,
    *,
    event_kind: TUIEventKind,
    markdown: bool = False,
) -> RenderableType:
    style = EVENT_STYLES[event_kind]
    if event_kind in ("thinking_delta", "reasoning_summary"):
        return _render_reasoning_line(label, content, style=style)
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    text.append("●", style=SUBTLE_BULLET_STYLE)
    text.append(" ")
    text.append(label, style=f"bold {style}")
    if content != "":
        text.append("  ")
    text.append(content, style=style)
    return text


def _render_reasoning_line(label: str, content: str, *, style: str) -> Text:
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    text.append("*", style=REASONING_MARK_STYLE)
    text.append(" ")
    text.append(label, style=f"bold {style}")
    if content != "":
        text.append("  ")
        text.append(content, style=style)
    return text


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
        return _render_assistant_block(event.content)
    if event.kind in ("thinking_delta", "reasoning_summary"):
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind in ("tool_started", "tool_completed", "tool_failed", "tool_output"):
        if event.kind == "tool_started":
            return None
        title = _tool_title(label, event)
        return _render_text_block(
            title,
            event.content or event.tool_call_id or "",
            event_kind=event.kind,
        )
    if event.kind == "media":
        return _render_text_block(label, "media segment", event_kind=event.kind)
    if event.kind == "llm_completed":
        if event.content == "":
            return None
        return _render_assistant_block(event.content)
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
    row = Table.grid(expand=True)
    row.add_column(ratio=1, style=PROMPT_BAR_STYLE)
    text = Text()
    text.append("▌ ", style=PROMPT_MARK_STYLE)
    text.append(content, style=PROMPT_BAR_STYLE)
    row.add_row(text, style=PROMPT_BAR_STYLE)
    return row


def _looks_like_markdown(content: str) -> bool:
    for line in content.splitlines():
        if line.startswith(("#", "> ", "- ", "* ", "```")):
            return True
        if line[:2].isdigit() and line[2:4] == ". ":
            return True
    return "`" in content or "**" in content or "__" in content


def _available_stream_width(stream: StreamWidget) -> int:
    widths = (
        stream.scrollable_content_region.width,
        stream.content_size.width,
        stream.size.width,
    )
    for width in widths:
        if width > 4:
            return width - 4
    return 1
