"""Main event stream pane for the read-only TUI prototype."""

import json
import shutil
import subprocess
from hashlib import sha1

from rich import box
from rich.align import Align
from msgspec import Struct
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.events import Click, Key, Resize
from textual.message import Message
from textual.timer import Timer
from textual.widgets import RichLog

from aceai import __version__
from aceai.agent.tui.events import TUIEvent, TUIEventKind, TUIIdeaItem
from aceai.agent.citations import TurnCitation, citation_origin_name
from aceai.agent.tui.state import TUIRunState
from aceai.agent.tui.theme import EVENT_LABELS, EVENT_STYLES

PROMPT_BAR_STYLE = "bold #eceff4 on #3b4252"
PROMPT_MARK_STYLE = "bold #88c0d0 on #3b4252"
SUBTLE_BULLET_STYLE = "bold #9aa3b2"
REASONING_MARK_STYLE = "bold #d08770"
EXPAND_MARK_STYLE = "bold #88c0d0"
TRANSCRIPT_GUTTER = "  "
EMPTY_STATE_FRAME_SECONDS = 0.45
EMPTY_STATE_DOG_PIXELS: tuple[tuple[str, ...], ...] = (
    (
        "  oo    oo  ",
        "  oooooooo  ",
        " oodoooodoo ",
        " oooonoooo  ",
        "  oooooooo  ",
        "    oooo    ",
        "   oooooo   ",
        "  oo    oo  ",
    ),
    (
        "  oo    oo  ",
        "  oooooooo  ",
        " oodoooodoo ",
        " oooonoooo  ",
        "  oooooooo  ",
        "    oooo  oo",
        "   oooooooo ",
        "  oo    oo  ",
    ),
    (
        "  oo    oo  ",
        "  oooooooo  ",
        " oodoooodoo ",
        " oooonoooo  ",
        "  oooooooo  ",
        "oo  oooo    ",
        " oooooooo   ",
        "  oo    oo  ",
    ),
)
EMPTY_STATE_DOG_HEIGHT = len(EMPTY_STATE_DOG_PIXELS[0])
EMPTY_STATE_DOG_WIDTH = max(len(row) for row in EMPTY_STATE_DOG_PIXELS[0]) * 2
EMPTY_STATE_MIN_WIDTH = EMPTY_STATE_DOG_WIDTH + 4
EMPTY_STATE_CONTENT_HEIGHT = EMPTY_STATE_DOG_HEIGHT + 2
TOOL_ARGUMENT_PREVIEW_MAX_CHARS = 96
TOOL_ARGUMENT_VALUE_PREVIEW_MAX_CHARS = 48
EMPTY_STATE_PIXEL_STYLES: dict[str, str] = {
    "o": "#e5b86f",
    "n": "#111827",
    "d": "#2e3440",
}


class StreamWidget(RichLog):
    """Render the readable event transcript."""

    class EventSelected(Message):
        def __init__(self, event_id: str) -> None:
            self.event_id = event_id
            super().__init__()

    DEFAULT_CSS = """
    StreamWidget {
        border: round #5e81ac;
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
        project_name: str = "",
        project_root_path: str = "",
    ) -> None:
        super().__init__(id=id, wrap=True, auto_scroll=True, min_width=0)
        self._state = state or TUIRunState()
        self._project_name = project_name
        self._project_root_path = project_root_path
        self._debug_mode = False
        self._selected_debug_index = 0
        self._empty_state_frame_index = 0
        self._empty_state_timer: Timer | None = None
        self._debug_line_spans: list[_DebugLineSpan] = []
        self._expanded_tool_activity_ids: set[str] = set()
        self._tool_activity_line_spans: list[_ToolActivityLineSpan] = []

    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    def set_project_name(self, project_name: str) -> None:
        self._project_name = project_name

    def set_project_root_path(self, project_root_path: str) -> None:
        self._project_root_path = project_root_path

    def set_debug_mode(self, enabled: bool) -> str | None:
        self._debug_mode = enabled
        selectable_events = _selectable_debug_events(self._state.events)
        if enabled and selectable_events:
            self._selected_debug_index = min(
                self._selected_debug_index,
                len(selectable_events) - 1,
            )
            selected_event_id = selectable_events[self._selected_debug_index].event_id
        else:
            selected_event_id = None
        self.set_state(self._state)
        return selected_event_id

    def set_state(self, state: TUIRunState) -> None:
        self._state = state
        self.clear()
        if not self._state.events:
            self._ensure_empty_state_timer()
            self._write_empty_state()
        elif self._debug_mode:
            self._stop_empty_state_timer()
            self._write_debug_stream()
        else:
            self._stop_empty_state_timer()
            self._tool_activity_line_spans = []
            for entry in _render_event_entries(
                self._state.events,
                collapse_tool_activity=self._state.status == "completed",
                expanded_tool_activity_ids=self._expanded_tool_activity_ids,
            ):
                start_line = len(self.lines)
                self._write_stream_renderable(entry.renderable)
                if entry.tool_activity_id != "":
                    self._tool_activity_line_spans.append(
                        _ToolActivityLineSpan(
                            activity_id=entry.tool_activity_id,
                            start_line=start_line,
                            end_line=len(self.lines),
                        )
                    )
        self.call_after_refresh(self.scroll_end, animate=False)

    def on_key(self, event: Key) -> None:
        if not self._debug_mode:
            return
        if event.key == "pageup":
            detail = self.app.query_one("#detail")
            detail.scroll_page_up(animate=False, force=True)
            event.stop()
            return
        if event.key == "pagedown":
            detail = self.app.query_one("#detail")
            detail.scroll_page_down(animate=False, force=True)
            event.stop()
            return
        if event.key in ("up", "k"):
            self._select_debug_event(self._selected_debug_index - 1)
            event.stop()
            return
        if event.key in ("down", "j"):
            self._select_debug_event(self._selected_debug_index + 1)
            event.stop()
            return
        if event.key == "home":
            self._select_debug_event(0)
            event.stop()
            return
        if event.key == "end":
            self._select_debug_event(len(_selectable_debug_events(self._state.events)) - 1)
            event.stop()

    def on_click(self, event: Click) -> None:
        line_index = self.scroll_y + event.y
        if not self._debug_mode:
            activity_id = _tool_activity_id_from_click(event)
            if activity_id != "":
                self._toggle_tool_activity(activity_id)
                event.stop()
                return
            for span in self._tool_activity_line_spans:
                if span.start_line <= line_index < span.end_line:
                    self._toggle_tool_activity(span.activity_id)
                    event.stop()
                    return
            return
        for span in self._debug_line_spans:
            if span.start_line <= line_index < span.end_line:
                self._select_debug_event(span.index)
                event.stop()
                return

    def _toggle_tool_activity(self, activity_id: str) -> None:
        if activity_id in self._expanded_tool_activity_ids:
            self._expanded_tool_activity_ids.remove(activity_id)
        else:
            self._expanded_tool_activity_ids.add(activity_id)
        self.set_state(self._state)

    def _select_debug_event(self, index: int) -> None:
        selectable_events = _selectable_debug_events(self._state.events)
        if not selectable_events:
            return
        self._selected_debug_index = max(0, min(index, len(selectable_events) - 1))
        self.set_state(self._state)
        self.post_message(
            self.EventSelected(selectable_events[self._selected_debug_index].event_id)
        )

    def _write_debug_stream(self) -> None:
        self._debug_line_spans = []
        selectable_events = _selectable_debug_events(self._state.events)
        if not selectable_events:
            self.write(Text("No inspectable messages yet", style="#d8dee9"))
            return
        self._selected_debug_index = min(
            self._selected_debug_index,
            len(selectable_events) - 1,
        )
        selected_event_id = selectable_events[self._selected_debug_index].event_id
        for index, entry in enumerate(_render_debug_events(selectable_events)):
            renderable = entry.renderable
            if entry.event_id == selected_event_id:
                renderable = _selected_debug_panel(renderable)
            elif index > 0:
                self.write(Text(""))
            start_line = len(self.lines)
            self._write_stream_renderable(renderable)
            self._debug_line_spans.append(
                _DebugLineSpan(
                    index=index,
                    event_id=entry.event_id,
                    start_line=start_line,
                    end_line=len(self.lines),
                )
            )

    def on_resize(self, event: Resize) -> None:
        self.set_state(self._state)

    def _write_stream_renderable(self, renderable: RenderableType) -> None:
        if isinstance(renderable, Table):
            self.write(renderable, expand=True)
            return
        self.write(renderable, width=_available_stream_width(self))

    def _ensure_empty_state_timer(self) -> None:
        if self._empty_state_timer is not None:
            return
        self._empty_state_timer = self.set_interval(
            EMPTY_STATE_FRAME_SECONDS,
            self._advance_empty_state,
            name="stream-empty-labrador",
        )

    def _stop_empty_state_timer(self) -> None:
        timer = self._empty_state_timer
        self._empty_state_timer = None
        if timer is not None:
            timer.stop()

    def _advance_empty_state(self) -> None:
        if self._state.events:
            self._stop_empty_state_timer()
            return
        self._empty_state_frame_index += 1
        self.clear()
        self._write_empty_state()

    def _write_empty_state(self) -> None:
        if _available_stream_width(self) < EMPTY_STATE_MIN_WIDTH:
            return
        self._write_stream_renderable(self._render_empty_state())

    def _render_empty_state(self) -> RenderableType:
        dog = _render_empty_state_dog(
            EMPTY_STATE_DOG_PIXELS[
                self._empty_state_frame_index % len(EMPTY_STATE_DOG_PIXELS)
            ]
        )
        title = _center_empty_state_text(
            f"AceAI v{__version__}",
            style="bold #8fbcbb",
        )
        project = _center_empty_state_text(
            f"Project: {self._project_name}",
            style="bold #d8dee9",
        )
        branch_name = _git_branch_name(self._project_root_path)
        branch = (
            _center_empty_state_text(
                f"Git: {branch_name}",
                style="#a7b1c2",
            )
            if branch_name is not None
            else Text("")
        )
        content = Align.center(
            Group(dog, Text(""), title, project, branch),
            vertical="middle",
        )
        top_padding = _empty_state_top_padding(self)
        if top_padding == 0:
            return content
        return Group(*[Text("") for _ in range(top_padding)], content)

    def on_unmount(self) -> None:
        self._stop_empty_state_timer()


class _ToolBlockState(Struct, kw_only=True):
    call_id: str
    name: str | None = None
    arguments: str = ""
    output: str = ""
    status: str = "running"


class _WorkingHistoryItem(Struct, kw_only=True):
    renderable: RenderableType | None = None
    tool_block: _ToolBlockState | None = None


class _ToolActivityState(Struct, kw_only=True):
    items: list[_WorkingHistoryItem]


class _StreamRenderable(Struct, kw_only=True):
    renderable: RenderableType
    tool_activity_id: str = ""


class _DebugRenderable(Struct, kw_only=True):
    event_id: str
    renderable: RenderableType


class _ToolActivityLineSpan(Struct, kw_only=True):
    activity_id: str
    start_line: int
    end_line: int


class _DebugLineSpan(Struct, kw_only=True):
    index: int
    event_id: str
    start_line: int
    end_line: int


def _render_events(
    events: list[TUIEvent],
    *,
    collapse_tool_activity: bool = False,
) -> list[RenderableType]:
    return [
        entry.renderable
        for entry in _render_event_entries(
            events,
            collapse_tool_activity=collapse_tool_activity,
            expanded_tool_activity_ids=set(),
        )
    ]


def _render_event_entries(
    events: list[TUIEvent],
    *,
    collapse_tool_activity: bool,
    expanded_tool_activity_ids: set[str],
) -> list[_StreamRenderable]:
    if collapse_tool_activity:
        return _render_completed_event_entries(
            events,
            expanded_tool_activity_ids,
        )

    entries: list[_StreamRenderable] = []
    assistant_buffer = ""
    assistant_buffer_step_id = ""
    thinking_buffer = ""
    assistant_step_ids: set[str] = set()
    pending_reasoning: dict[str, list[TUIEvent]] = {}
    rendered_reasoning_step_ids: set[str] = set()
    tool_blocks: dict[str, _ToolBlockState] = {}
    rendered_tool_call_ids: set[str] = set()
    pending_working_history: _ToolActivityState | None = None

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
            "tool_approval_requested",
            "tool_approval_resolved",
            "tool_completed",
            "tool_failed",
            "run_suspended",
        ):
            thinking_buffer = _flush_thinking_buffer(entries, thinking_buffer)
            _flush_pending_reasoning(
                entries,
                event.step_id,
                pending_reasoning,
                rendered_reasoning_step_ids,
            )
            assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
                entries,
                assistant_buffer,
                assistant_buffer_step_id,
                assistant_step_ids,
            )
            _update_tool_block(tool_blocks, event)
            if event.kind in ("tool_completed", "tool_failed", "tool_approval_requested"):
                entries.append(
                    _StreamRenderable(
                        renderable=_render_tool_block(tool_blocks[event.tool_call_id])
                    )
                )
                rendered_tool_call_ids.add(event.tool_call_id)
            continue

        if event.kind == "llm_completed":
            _flush_pending_reasoning(
                entries,
                event.step_id,
                pending_reasoning,
                rendered_reasoning_step_ids,
            )
            if event.tool_calls:
                if assistant_buffer_step_id == event.step_id:
                    assistant_buffer = ""
                    assistant_buffer_step_id = ""
                continue
        if _is_invisible_control_event(event):
            continue
        thinking_buffer = _flush_thinking_buffer(entries, thinking_buffer)
        assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
            entries,
            assistant_buffer,
            assistant_buffer_step_id,
            assistant_step_ids,
        )
        pending_working_history = _flush_tool_activity(
            entries,
            pending_working_history,
            expanded_tool_activity_ids,
        )

        if event.kind == "llm_completed" and event.step_id in assistant_step_ids:
            continue

        rendered = _render_event(event)
        if rendered is not None:
            _flush_pending_reasoning(
                entries,
                event.step_id,
                pending_reasoning,
                rendered_reasoning_step_ids,
            )
            if event.kind == "user_message" and entries:
                entries.append(_StreamRenderable(renderable=Text("")))
            if event.kind == "llm_completed":
                assistant_step_ids.add(event.step_id)
            entries.append(_StreamRenderable(renderable=rendered))

    thinking_buffer = _flush_thinking_buffer(entries, thinking_buffer)
    for step_id in pending_reasoning:
        _flush_pending_reasoning(
            entries,
            step_id,
            pending_reasoning,
            rendered_reasoning_step_ids,
        )
    for call_id, tool_block in tool_blocks.items():
        if call_id in rendered_tool_call_ids:
            continue
        if tool_block.name is None:
            continue
        entries.append(_StreamRenderable(renderable=_render_tool_block(tool_block)))
    _flush_tool_activity(
        entries,
        pending_working_history,
        expanded_tool_activity_ids,
    )
    assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
        entries,
        assistant_buffer,
        assistant_buffer_step_id,
        assistant_step_ids,
    )
    return entries


def _render_completed_event_entries(
    events: list[TUIEvent],
    expanded_tool_activity_ids: set[str],
) -> list[_StreamRenderable]:
    entries: list[_StreamRenderable] = []
    assistant_buffer = ""
    assistant_buffer_step_id = ""
    assistant_step_ids: set[str] = set()
    thinking_buffer = ""
    pending_working_history: _ToolActivityState | None = None
    tool_blocks: dict[str, _ToolBlockState] = {}
    rendered_tool_call_ids: set[str] = set()

    for event in events:
        if event.kind == "user_message":
            pending_working_history = _flush_thinking_buffer_to_working_history(
                pending_working_history,
                thinking_buffer,
            )
            thinking_buffer = ""
            pending_working_history, assistant_buffer, assistant_buffer_step_id = (
                _flush_completed_turn(
                    entries,
                    pending_working_history,
                    assistant_buffer,
                    assistant_buffer_step_id,
                    assistant_step_ids,
                    expanded_tool_activity_ids,
                )
            )
            if entries:
                entries.append(_StreamRenderable(renderable=Text("")))
            rendered = _render_event(event)
            if rendered is not None:
                entries.append(_StreamRenderable(renderable=rendered))
            continue

        if event.kind == "thinking_delta":
            thinking_buffer += event.content
            continue

        if event.kind == "reasoning_summary":
            rendered = _render_event(event)
            if rendered is not None:
                pending_working_history = _append_working_history_renderable(
                    pending_working_history,
                    rendered,
                )
            continue

        if event.kind == "assistant_delta":
            assistant_buffer_step_id = event.step_id
            assistant_buffer += event.content
            continue

        if event.kind == "llm_completed":
            if event.tool_calls:
                if assistant_buffer_step_id == event.step_id:
                    assistant_buffer = ""
                    assistant_buffer_step_id = ""
                continue
            if assistant_buffer == "" and event.content != "":
                assistant_buffer_step_id = event.step_id
                assistant_buffer = event.content
            continue

        if event.kind == "run_completed":
            if assistant_buffer == "" and event.content != "":
                assistant_buffer_step_id = event.step_id
                assistant_buffer = event.content
            continue

        if event.tool_call_id is not None and event.kind in (
            "tool_call_delta",
            "tool_started",
            "tool_output",
            "tool_approval_requested",
            "tool_approval_resolved",
            "tool_completed",
            "tool_failed",
            "run_suspended",
        ):
            pending_working_history = _flush_thinking_buffer_to_working_history(
                pending_working_history,
                thinking_buffer,
            )
            thinking_buffer = ""
            _update_tool_block(tool_blocks, event)
            if event.kind in ("tool_completed", "tool_failed", "tool_approval_requested"):
                pending_working_history = _append_tool_block(
                    entries,
                    pending_working_history,
                    tool_blocks[event.tool_call_id],
                    expanded_tool_activity_ids,
                )
                rendered_tool_call_ids.add(event.tool_call_id)
            continue

        if event.kind in ("session_notice", "idea_list"):
            pending_working_history = _flush_thinking_buffer_to_working_history(
                pending_working_history,
                thinking_buffer,
            )
            thinking_buffer = ""
            pending_working_history, assistant_buffer, assistant_buffer_step_id = (
                _flush_completed_turn(
                    entries,
                    pending_working_history,
                    assistant_buffer,
                    assistant_buffer_step_id,
                    assistant_step_ids,
                    expanded_tool_activity_ids,
                )
            )
            rendered = _render_event(event)
            if rendered is not None:
                entries.append(_StreamRenderable(renderable=rendered))

    pending_working_history = _flush_thinking_buffer_to_working_history(
        pending_working_history,
        thinking_buffer,
    )
    thinking_buffer = ""
    for call_id, tool_block in tool_blocks.items():
        if call_id in rendered_tool_call_ids:
            continue
        if tool_block.name is None:
            continue
        pending_working_history = _append_tool_block(
            entries,
            pending_working_history,
            tool_block,
            expanded_tool_activity_ids,
        )
    _flush_completed_turn(
        entries,
        pending_working_history,
        assistant_buffer,
        assistant_buffer_step_id,
        assistant_step_ids,
        expanded_tool_activity_ids,
    )
    return entries


def _flush_completed_turn(
    entries: list[_StreamRenderable],
    pending_working_history: _ToolActivityState | None,
    assistant_buffer: str,
    assistant_buffer_step_id: str,
    assistant_step_ids: set[str],
    expanded_tool_activity_ids: set[str],
) -> tuple[_ToolActivityState | None, str, str]:
    pending_working_history = _flush_tool_activity(
        entries,
        pending_working_history,
        expanded_tool_activity_ids,
    )
    assistant_buffer, assistant_buffer_step_id = _flush_assistant_buffer(
        entries,
        assistant_buffer,
        assistant_buffer_step_id,
        assistant_step_ids,
    )
    return pending_working_history, assistant_buffer, assistant_buffer_step_id


def _render_debug_events(events: list[TUIEvent]) -> list[_DebugRenderable]:
    renderables: list[_DebugRenderable] = []
    for event in events:
        rendered = _render_debug_event(event)
        if rendered is None:
            continue
        renderables.append(
            _DebugRenderable(event_id=event.event_id, renderable=rendered)
        )
    return renderables


def _render_debug_event(event: TUIEvent) -> RenderableType | None:
    if event.kind == "tool_started":
        return _render_text_block(
            event.tool_name or EVENT_LABELS[event.kind],
            "running",
            event_kind=event.kind,
        )
    return _render_event(event)


def _selectable_debug_events(events: list[TUIEvent]) -> list[TUIEvent]:
    selectable: list[TUIEvent] = []
    for event in events:
        if _render_debug_event(event) is None:
            continue
        selectable.append(event)
    return selectable


def _selected_debug_panel(renderable: RenderableType) -> RenderableType:
    return Panel(
        renderable,
        box=box.ROUNDED,
        border_style="#88c0d0",
        style="on #3b4252",
        padding=(0, 0),
    )


def _flush_pending_reasoning(
    renderables: list[_StreamRenderable],
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
            renderables.append(_StreamRenderable(renderable=rendered))
    rendered_reasoning_step_ids.add(step_id)


def _flush_pending_reasoning_to_working_history(
    pending_working_history: _ToolActivityState | None,
    step_id: str,
    pending_reasoning: dict[str, list[TUIEvent]],
    rendered_reasoning_step_ids: set[str],
) -> _ToolActivityState | None:
    if step_id in rendered_reasoning_step_ids:
        return pending_working_history
    events = pending_reasoning.get(step_id)
    if events is None:
        return pending_working_history
    for event in events:
        rendered = _render_event(event)
        if rendered is not None:
            pending_working_history = _append_working_history_renderable(
                pending_working_history,
                rendered,
            )
    rendered_reasoning_step_ids.add(step_id)
    return pending_working_history


def _flush_assistant_buffer(
    renderables: list[_StreamRenderable],
    assistant_buffer: str,
    assistant_buffer_step_id: str,
    assistant_step_ids: set[str],
) -> tuple[str, str]:
    if assistant_buffer == "":
        return "", assistant_buffer_step_id
    renderables.append(
        _StreamRenderable(renderable=_render_assistant_block(assistant_buffer))
    )
    assistant_step_ids.add(assistant_buffer_step_id)
    return "", ""


def _flush_thinking_buffer(
    renderables: list[_StreamRenderable],
    thinking_buffer: str,
) -> str:
    if thinking_buffer:
        renderables.append(
            _StreamRenderable(
                renderable=_render_text_block(
                    "reasoning",
                    thinking_buffer,
                    event_kind="thinking_delta",
                )
            )
        )
    return ""


def _flush_thinking_buffer_to_working_history(
    pending_working_history: _ToolActivityState | None,
    thinking_buffer: str,
) -> _ToolActivityState | None:
    if thinking_buffer == "":
        return pending_working_history
    return _append_working_history_renderable(
        pending_working_history,
        _render_text_block(
            "reasoning",
            thinking_buffer,
            event_kind="thinking_delta",
        ),
    )


def _append_working_history_renderable(
    pending_working_history: _ToolActivityState | None,
    renderable: RenderableType,
) -> _ToolActivityState:
    if pending_working_history is None:
        pending_working_history = _ToolActivityState(items=[])
    pending_working_history.items.append(
        _WorkingHistoryItem(renderable=renderable)
    )
    return pending_working_history


def _append_tool_block(
    renderables: list[_StreamRenderable],
    pending_tool_activity: _ToolActivityState | None,
    tool_block: _ToolBlockState,
    expanded_tool_activity_ids: set[str],
) -> _ToolActivityState | None:
    if not _tool_block_belongs_to_activity(tool_block):
        pending_tool_activity = _flush_tool_activity(
            renderables,
            pending_tool_activity,
            expanded_tool_activity_ids,
        )
        renderables.append(_StreamRenderable(renderable=_render_tool_block(tool_block)))
        return pending_tool_activity
    if pending_tool_activity is None:
        pending_tool_activity = _ToolActivityState(items=[])
    cloned = _clone_tool_block(tool_block)
    for index, item in enumerate(pending_tool_activity.items):
        if item.tool_block is not None and item.tool_block.call_id == cloned.call_id:
            pending_tool_activity.items[index] = _WorkingHistoryItem(tool_block=cloned)
            return pending_tool_activity
    pending_tool_activity.items.append(_WorkingHistoryItem(tool_block=cloned))
    return pending_tool_activity


def _flush_tool_activity(
    renderables: list[_StreamRenderable],
    pending_tool_activity: _ToolActivityState | None,
    expanded_tool_activity_ids: set[str],
) -> None:
    if (
        pending_tool_activity is not None
        and _working_history_tool_blocks(pending_tool_activity)
    ):
        renderables.extend(
            _render_tool_activity_entries(
                pending_tool_activity,
                expanded_tool_activity_ids,
            )
        )
    return None


def _tool_block_belongs_to_activity(tool_block: _ToolBlockState) -> bool:
    if tool_block.name is None:
        raise ValueError("tool block must include a tool name before rendering")
    return tool_block.status in ("awaiting_approval", "completed", "failed")


def _clone_tool_block(tool_block: _ToolBlockState) -> _ToolBlockState:
    return _ToolBlockState(
        call_id=tool_block.call_id,
        name=tool_block.name,
        status=tool_block.status,
        arguments=tool_block.arguments,
        output=tool_block.output,
    )


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
    if event.kind in ("tool_approval_requested", "run_suspended"):
        tool_block.status = "awaiting_approval"
        tool_block.output = event.content
    elif event.kind == "tool_approval_resolved":
        tool_block.status = "running"
        tool_block.output = event.content
    elif event.kind == "tool_failed":
        tool_block.status = "failed"
    elif event.kind == "tool_completed":
        tool_block.status = "completed"


def _render_tool_block(tool_block: _ToolBlockState) -> Text:
    event_kind: TUIEventKind = (
        "tool_failed"
        if tool_block.status == "failed"
        else "tool_completed"
        if tool_block.status == "completed"
        else "tool_approval_requested"
        if tool_block.status == "awaiting_approval"
        else "tool_started"
    )
    style = EVENT_STYLES[event_kind]
    if tool_block.name is None:
        raise ValueError("tool block must include a tool name before rendering")
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    text.append("●", style=SUBTLE_BULLET_STYLE)
    text.append(" ")
    text.append(_tool_call_preview(tool_block), style=style)
    summary = _tool_summary(tool_block)
    if summary != "":
        text.append(f"  {summary}", style=style)
    return text


def _render_tool_activity_entries(
    tool_activity: _ToolActivityState,
    expanded_tool_activity_ids: set[str],
) -> list[_StreamRenderable]:
    activity_id = _tool_activity_id(tool_activity)
    expanded = activity_id in expanded_tool_activity_ids
    entries = [
        _StreamRenderable(
            renderable=_render_tool_activity_header(tool_activity, expanded=expanded),
            tool_activity_id=activity_id,
        )
    ]
    if expanded:
        for item in tool_activity.items:
            entries.append(
                _StreamRenderable(renderable=_render_expanded_working_history_item(item))
            )
    return entries


def _render_tool_activity_header(
    tool_activity: _ToolActivityState,
    *,
    expanded: bool,
) -> Text:
    blocks = _working_history_tool_blocks(tool_activity)
    activity_id = _tool_activity_id(tool_activity)
    event_kind = _tool_activity_event_kind(blocks)
    style = EVENT_STYLES[event_kind]
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    text.append("─", style=EXPAND_MARK_STYLE)
    text.append(" ")
    text.append("[-]" if expanded else "[+]", style=EXPAND_MARK_STYLE)
    text.append(" ")
    text.append("work history", style=f"bold {style}")
    text.append(" · ", style=EXPAND_MARK_STYLE)
    text.append(
        f"{len(blocks)} tool {_pluralize('call', len(blocks))}",
        style=style,
    )
    text.stylize(_tool_activity_click_style(activity_id), 0, len(text))
    return text


def _render_expanded_working_history_item(item: _WorkingHistoryItem) -> RenderableType:
    if item.renderable is not None:
        return _indent_renderable(item.renderable)
    if item.tool_block is None:
        raise ValueError("working history item must include content")
    return _render_expanded_tool_block(item.tool_block)


def _render_expanded_tool_block(tool_block: _ToolBlockState) -> Text:
    base = _render_tool_block(tool_block)
    text = Text()
    text.append("  ")
    text.append(base)
    return text


def _pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    return f"{word}s"


def _indent_renderable(renderable: RenderableType) -> RenderableType:
    if isinstance(renderable, Text):
        text = Text("  ")
        text.append(renderable)
        return text
    return Group(Text("  "), renderable)


def _tool_activity_id(tool_activity: _ToolActivityState) -> str:
    blocks = _working_history_tool_blocks(tool_activity)
    if blocks:
        return "|".join(block.call_id for block in blocks)
    digest = sha1()
    for item in tool_activity.items:
        if item.renderable is not None and isinstance(item.renderable, Text):
            digest.update(item.renderable.plain.encode("utf-8"))
    return f"working-history:{digest.hexdigest()}"


def _tool_activity_click_style(activity_id: str) -> Style:
    return Style(meta={"tool_activity_id": activity_id})


def _tool_activity_id_from_click(event: Click) -> str:
    if event.style is None:
        return ""
    activity_id = event.style.meta.get("tool_activity_id")
    if type(activity_id) is not str:
        return ""
    return activity_id


def _tool_activity_event_kind(blocks: list[_ToolBlockState]) -> TUIEventKind:
    for block in blocks:
        if block.status == "awaiting_approval":
            return "tool_approval_requested"
    for block in blocks:
        if block.status == "failed":
            return "tool_failed"
    return "tool_completed"


def _working_history_tool_blocks(
    tool_activity: _ToolActivityState,
) -> list[_ToolBlockState]:
    blocks: list[_ToolBlockState] = []
    for item in tool_activity.items:
        if item.tool_block is not None:
            blocks.append(item.tool_block)
    return blocks


def _tool_activity_names_summary(blocks: list[_ToolBlockState]) -> str:
    counts: dict[str, int] = {}
    for block in blocks:
        if block.name is None:
            raise ValueError("tool block must include a tool name before rendering")
        counts[block.name] = counts.get(block.name, 0) + 1
    parts: list[str] = []
    for name, count in counts.items():
        if count == 1:
            parts.append(name)
        else:
            parts.append(f"{name} x{count}")
    return ", ".join(parts)


def _tool_summary(tool_block: _ToolBlockState) -> str:
    if tool_block.status == "failed":
        return "failed"
    if tool_block.status == "completed":
        return _tool_output_summary(tool_block.output)
    if tool_block.status == "awaiting_approval":
        return "waiting for approval"
    return "running"


def _tool_call_preview(tool_block: _ToolBlockState) -> str:
    if tool_block.name is None:
        raise ValueError("tool block must include a tool name before rendering")
    arguments = _tool_argument_preview(tool_block.arguments)
    if arguments == "":
        return tool_block.name
    return f"{tool_block.name}({arguments})"


def _tool_argument_preview(arguments: str) -> str:
    if arguments == "":
        return ""
    payload = json.loads(arguments)
    if not isinstance(payload, dict):
        return "..."
    items = list(payload.items())
    if len(items) == 1:
        return _tool_argument_value_preview(items[0][1])
    parts: list[str] = []
    for key, value in items:
        if type(key) is not str:
            return "..."
        parts.append(f"{key}: {_tool_argument_value_preview(value)}")
    preview = ", ".join(parts)
    if len(preview) > TOOL_ARGUMENT_PREVIEW_MAX_CHARS:
        return preview[: TOOL_ARGUMENT_PREVIEW_MAX_CHARS - 3] + "..."
    return preview


def _tool_argument_value_preview(value: object) -> str:
    if isinstance(value, str):
        if len(value) > TOOL_ARGUMENT_VALUE_PREVIEW_MAX_CHARS:
            value = value[: TOOL_ARGUMENT_VALUE_PREVIEW_MAX_CHARS - 3] + "..."
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int | float | bool) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return "..."


def _tool_output_summary(output: str) -> str:
    if output == "":
        return ""
    if '"entries":[' in output:
        entry_count = output.count('"name"')
        return f"{entry_count} entries"
    if '"bytes_written":' in output:
        return "file written"
    if '"matches":' in output:
        return "search finished"
    if '"exit_code":' in output:
        return _shell_output_summary(output)
    return "result ready"


def _shell_output_summary(output: str) -> str:
    payload = json.loads(output)
    exit_code = payload["exit_code"]
    stdout = payload["stdout"]
    stderr = payload["stderr"]
    if type(exit_code) is not int:
        raise TypeError("shell tool exit_code must be int")
    if type(stdout) is not str:
        raise TypeError("shell tool stdout must be str")
    if type(stderr) is not str:
        raise TypeError("shell tool stderr must be str")
    if exit_code == 0:
        return "succeeded"
    return f"exit {exit_code}"


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


def _render_idea_list(items: list[TUIIdeaItem]) -> RenderableType:
    if not items:
        return _render_text_block(
            "ideas",
            "No saved ideas yet.",
            event_kind="idea_list",
        )
    renderables: list[RenderableType] = [_render_idea_header(len(items))]
    for item in items:
        renderables.append(_render_idea_item(item))
    return Group(*renderables)


def _render_idea_header(count: int) -> Text:
    text = Text()
    text.append(TRANSCRIPT_GUTTER)
    text.append("Ideas", style=f"bold {EVENT_STYLES['idea_list']}")
    text.append(f"  {count}", style=SUBTLE_BULLET_STYLE)
    return text


def _render_idea_item(item: TUIIdeaItem) -> Panel:
    title = Text()
    title.append(f"{item.index}. ", style=SUBTLE_BULLET_STYLE)
    title.append(item.title, style="bold #eceff4")
    title.append(f"  {item.created_at}", style="#9aa3b2")
    title.append(f"  {item.project_name}", style="#8fbcbb")
    body = Text()
    body.append(item.body if item.body != "" else " ", style="#d8dee9")
    return Panel(
        body,
        box=box.ROUNDED,
        title=title,
        title_align="left",
        border_style="#4c566a",
        padding=(0, 1),
    )


def _render_event(event: TUIEvent) -> RenderableType | None:
    label = EVENT_LABELS[event.kind]
    if event.kind == "user_message":
        return _render_user_message(
            event.content,
            label=label,
            event_kind=event.kind,
            citations=event.citations,
        )
    if event.kind == "session_notice":
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind == "idea_list":
        return _render_idea_list(event.idea_items)
    if event.kind == "tool_call_delta":
        return None
    if event.kind == "assistant_delta":
        return _render_assistant_block(event.content)
    if event.kind in ("thinking_delta", "reasoning_summary"):
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind == "llm_retrying":
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind in (
        "context_compaction_started",
        "context_compaction_failed",
        "context_compressed",
    ):
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind in (
        "tool_started",
        "tool_approval_requested",
        "tool_approval_resolved",
        "tool_completed",
        "tool_failed",
        "tool_output",
    ):
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
    if event.kind in ("step_failed", "run_failed"):
        return _render_text_block(label, event.content, event_kind=event.kind)
    if event.kind == "llm_completed":
        if event.content == "":
            return None
        return _render_assistant_block(event.content)
    if event.kind in (
        "step_started",
        "step_completed",
        "run_completed",
        "run_suspended",
    ):
        return None
    return None


def _is_invisible_control_event(event: TUIEvent) -> bool:
    return event.kind in (
        "step_started",
        "step_completed",
        "run_completed",
        "run_suspended",
    )


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
    citations: tuple[TurnCitation, ...],
) -> RenderableType:
    row = Table.grid(expand=True)
    row.add_column(ratio=1, style=PROMPT_BAR_STYLE)
    text = Text()
    text.append("▌ ", style=PROMPT_MARK_STYLE)
    text.append(content, style=PROMPT_BAR_STYLE)
    row.add_row(text, style=PROMPT_BAR_STYLE)
    if not citations:
        return row
    return Group(_render_cited_sources(citations), row)


def _render_cited_sources(citations: tuple[TurnCitation, ...]) -> Panel:
    lines: list[Text] = []
    for index, citation in enumerate(citations, start=1):
        title = Text()
        title.append(f"[{index}] ", style=SUBTLE_BULLET_STYLE)
        title.append(citation_origin_name(citation.origin), style="bold #d8dee9")
        body = Text(citation.content, style="#d8dee9")
        lines.append(title)
        lines.append(body)
    return Panel(
        Group(*lines),
        box=box.ROUNDED,
        title="cited source",
        title_align="left",
        border_style="#4c566a",
        padding=(0, 1),
    )


def _looks_like_markdown(content: str) -> bool:
    for line in content.splitlines():
        if line.startswith(("#", "> ", "- ", "* ", "```")):
            return True
        if line[:2].isdigit() and line[2:4] == ". ":
            return True
    return "`" in content or "**" in content or "__" in content


def _render_empty_state_dog(frame: tuple[str, ...]) -> RenderableType:
    lines: list[Text] = []
    for row in frame:
        line = Text()
        for pixel in row:
            style = EMPTY_STATE_PIXEL_STYLES.get(pixel)
            if style is None:
                line.append("  ")
            else:
                line.append("██", style=style)
        lines.append(line)
    return Group(*lines)


def _center_empty_state_text(value: str, *, style: str) -> Text:
    left_padding = max(0, (EMPTY_STATE_DOG_WIDTH - len(value)) // 2)
    return Text(f"{' ' * left_padding}{value}", style=style)


def _git_branch_name(project_root_path: str) -> str | None:
    if project_root_path == "" or shutil.which("git") is None:
        return None
    completed = subprocess.run(
        ["git", "-C", project_root_path, "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    branch = completed.stdout.strip()
    if branch == "":
        return None
    return branch


def _empty_state_top_padding(stream: StreamWidget) -> int:
    heights = (
        stream.scrollable_content_region.height,
        stream.content_size.height,
        stream.size.height,
    )
    for height in heights:
        if height > EMPTY_STATE_CONTENT_HEIGHT:
            return (height - EMPTY_STATE_CONTENT_HEIGHT) // 2
    return 0


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
