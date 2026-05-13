"""Subagent activity tracker for delegated child-agent runs."""

from dataclasses import dataclass

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key, MouseScrollDown, MouseScrollUp, Resize
from textual.message import Message
from textual.widgets import Button, RichLog

from agent_core.session import MAIN_THREAD_ID
from agent_core.tui.state import TUISubagentState, TUISubagentToolResult


@dataclass(frozen=True)
class SubagentThreadOption:
    thread_id: str
    label: str
    status: str
    role: str
    agent_id: str = ""
    run_id: str = ""
    parent_run_id: str = ""
    instructions: str = ""
    context_brief: str = ""
    allowed_tools: tuple[str, ...] = ()
    summary: str = ""
    final_answer: str = ""
    step_count: int = 0
    tool_result_count: int = 0
    inbox_pending_count: int = 0
    inbox_latest: str = ""


class SubagentStatusWidget(Vertical):
    """Paged detail area for delegate_to_subagent tool calls."""

    DEFAULT_CSS = """
    SubagentStatusWidget {
        width: 48;
        height: 1fr;
        margin-left: 1;
        padding: 1;
        background: #2e3440;
        color: #d8dee9;
        border: round #8fbcbb;
    }

    SubagentStatusWidget.hidden {
        display: none;
    }

    SubagentStatusWidget #subagent-detail {
        height: 1fr;
        background: #2e3440;
        color: #d8dee9;
        border: none;
        padding: 0;
        overflow-y: auto;
        overflow-x: hidden;
    }

    SubagentStatusWidget #subagent-activate {
        height: 3;
        margin-top: 1;
        width: 100%;
        background: #3b4252;
        color: #d8dee9;
        border: tall #4c566a;
    }

    SubagentStatusWidget #subagent-activate.enabled {
        background: #a3be8c;
        color: #2e3440;
        text-style: bold;
        border: tall #a3be8c;
    }
    """

    class ThreadActivated(Message):
        def __init__(self, thread_id: str) -> None:
            super().__init__()
            self.thread_id = thread_id

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display_text = ""
        self.display = False
        self.can_focus = True
        self._subagents: list[TUISubagentState] = []
        self._thread_options: list[SubagentThreadOption] = []
        self._active_thread_id = ""
        self._page_index = 0

    def compose(self) -> ComposeResult:
        yield RichLog(
            id="subagent-detail",
            wrap=True,
            auto_scroll=False,
        )
        yield Button("activate", id="subagent-activate", disabled=True)

    def set_state(
        self,
        *,
        subagents: list[TUISubagentState],
        thread_options: list[SubagentThreadOption],
        active_thread_id: str,
    ) -> None:
        if not subagents and len(thread_options) <= 1:
            self.display = False
            self.add_class("hidden")
            self.display_text = ""
            self._subagents = []
            self._thread_options = []
            self._active_thread_id = active_thread_id
            self._page_index = 0
            if self.is_mounted:
                self._detail_log().clear()
            return
        self.display = True
        self.remove_class("hidden")
        self._subagents = subagents
        self._thread_options = thread_options
        self._active_thread_id = active_thread_id
        if self._page_index >= len(subagents):
            self._page_index = len(subagents) - 1
        if self._page_index < 0:
            self._page_index = 0
        self._render_current_page()
        self._sync_activate_button()

    def set_subagents(self, subagents: list[TUISubagentState]) -> None:
        self.set_state(
            subagents=subagents,
            thread_options=[],
            active_thread_id="",
        )

    def on_key(self, event: Key) -> None:
        if event.key in ("right", "l", "]"):
            self.next_page()
            event.stop()
            return
        if event.key in ("left", "h", "["):
            self.previous_page()
            event.stop()
            return
        if event.key in ("up", "k"):
            self._detail_log().scroll_up(animate=False, force=True, immediate=True)
            event.stop()
            return
        if event.key in ("down", "j"):
            self._detail_log().scroll_down(animate=False, force=True, immediate=True)
            event.stop()
            return
        if event.key == "pageup":
            self._detail_log().scroll_page_up(animate=False, force=True)
            event.stop()
            return
        if event.key == "pagedown":
            self._detail_log().scroll_page_down(animate=False, force=True)
            event.stop()
            return
        if event.key == "home":
            self._detail_log().scroll_home(animate=False, force=True)
            event.stop()
            return
        if event.key == "end":
            self._detail_log().scroll_end(animate=False, force=True)
            event.stop()

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._detail_log().scroll_relative(y=3, animate=False, force=True, immediate=True)
        event.stop()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._detail_log().scroll_relative(y=-3, animate=False, force=True, immediate=True)
        event.stop()

    def on_resize(self, event: Resize) -> None:
        self._render_current_page()

    def next_page(self) -> None:
        if not self._subagents:
            return
        self._page_index = min(self._page_index + 1, len(self._subagents) - 1)
        self._render_current_page()

    def previous_page(self) -> None:
        if not self._subagents:
            return
        self._page_index = max(self._page_index - 1, 0)
        self._render_current_page()

    @property
    def renderable(self) -> str:
        return self.display_text

    def _render_current_page(self) -> None:
        if not self.is_mounted:
            if not self._subagents:
                self.display_text = _empty_subagent_status_text(self._thread_options)
                return
            self.display_text = _subagent_status_text(
                self._subagents,
                self._page_index,
                self._thread_options,
                self._active_thread_id,
            )
            return
        detail_log = self._detail_log()
        detail_log.clear()
        if not self._subagents:
            self.display_text = _empty_subagent_status_text(self._thread_options)
            detail_log.write(
                _empty_subagent_status_renderable(self._thread_options),
                width=max(1, detail_log.scrollable_content_region.width),
            )
            self._sync_activate_button()
            return
        self.display_text = _subagent_status_text(
            self._subagents,
            self._page_index,
            self._thread_options,
            self._active_thread_id,
        )
        width = max(1, detail_log.scrollable_content_region.width)
        detail_log.write(
            _subagent_status_renderable(
                self._subagents,
                self._page_index,
                self._thread_options,
                self._active_thread_id,
            ),
            width=width,
        )
        detail_log.scroll_home(animate=False, force=True, immediate=True)
        self._sync_activate_button()

    def _sync_activate_button(self) -> None:
        if not self.is_mounted:
            return
        button = self.query_one("#subagent-activate", Button)
        button.remove_class("enabled")
        if not self._subagents:
            button.label = "activate"
            button.disabled = True
            return
        thread_id = self._subagents[self._page_index].thread_id
        if thread_id == "":
            button.label = "no transcript"
            button.disabled = True
            return
        if thread_id == self._active_thread_id:
            button.label = "activated"
            button.disabled = True
            return
        button.label = "activate"
        button.disabled = False
        button.add_class("enabled")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "subagent-activate":
            return
        if not self._subagents:
            return
        thread_id = self._subagents[self._page_index].thread_id
        if thread_id == "":
            return
        self.post_message(self.ThreadActivated(thread_id))

    def _detail_log(self) -> RichLog:
        return self.query_one("#subagent-detail", RichLog)


def _subagent_status_text(
    subagents: list[TUISubagentState],
    page_index: int,
    thread_options: list[SubagentThreadOption],
    active_thread_id: str,
) -> str:
    subagent = subagents[page_index]
    lines = [
        _subagent_header(subagents, page_index),
    ]
    lines.extend(_subagent_text_lines(page_index, subagent))
    return "\n".join(lines)


def _subagent_status_renderable(
    subagents: list[TUISubagentState],
    page_index: int,
    thread_options: list[SubagentThreadOption],
    active_thread_id: str,
) -> RenderableType:
    subagent = subagents[page_index]
    if _is_main_agent_panel(subagent):
        return Group(
            _subagent_overview(subagents),
            Text(""),
            _page_carousel_renderable(len(subagents), page_index, 42),
            Text(""),
            _main_agent_title(index=page_index),
            Text(""),
            _main_agent_panel(subagent),
        )
    return Group(
        _subagent_overview(subagents),
        Text(""),
        _page_carousel_renderable(len(subagents), page_index, 42),
        Text(""),
        _subagent_title(index=page_index, subagent=subagent),
        Text(""),
        _subagent_status_panel(subagent),
        Text(""),
        _subagent_task_panel(subagent),
        _subagent_access_panel(subagent),
        _subagent_identity_panel(subagent),
        _subagent_evidence_panel(subagent),
        _subagent_tool_results_panel(subagent.tool_results),
        _subagent_error_panel(subagent.error),
    )


def _empty_subagent_status_text(thread_options: list[SubagentThreadOption]) -> str:
    return "subagents  no detail for this thread"


def _empty_subagent_status_renderable(
    thread_options: list[SubagentThreadOption],
) -> Text:
    text = Text()
    text.append("subagents", style="bold #8fbcbb")
    text.append("  no detail for this thread", style="#9aa3b2")
    return text


def _subagent_text_lines(index: int, subagent: TUISubagentState) -> list[str]:
    if _is_main_agent_panel(subagent):
        return _main_agent_text_lines(index, subagent)
    lines = [
        "",
        f"#{index + 1} [{subagent.status}] {_subagent_label(subagent)}",
    ]
    lines.extend(_subagent_summary_lines(subagent))
    lines.extend(_collection_lines("evidence", subagent.important_evidence))
    lines.extend(_tool_result_lines(subagent.tool_results))
    if subagent.error is not None:
        lines.extend(_detail_lines("error", subagent.error))
    return lines


def _subagent_header(subagents: list[TUISubagentState], page_index: int) -> str:
    return (
        f"subagents  {_subagent_counts(subagents)}\n"
        f"{_centered_page_carousel_label(len(subagents), page_index, 42)}"
    )


def _page_carousel_label(total_pages: int, page_index: int) -> str:
    current_page = page_index + 1
    parts = ["<"]
    if current_page > 1:
        parts.append(str(current_page - 1))
    parts.append(f"[{current_page}]")
    if current_page < total_pages:
        parts.append(str(current_page + 1))
    parts.append(">")
    return " ".join(parts)


def _centered_page_carousel_label(
    total_pages: int,
    page_index: int,
    width: int,
) -> str:
    label = _page_carousel_label(total_pages, page_index)
    padding = max(0, (width - len(label)) // 2)
    return (" " * padding) + label


def _append_centered_page_carousel(
    text: Text,
    total_pages: int,
    page_index: int,
    width: int,
) -> None:
    label = _page_carousel_label(total_pages, page_index)
    padding = max(0, (width - len(label)) // 2)
    text.append(" " * padding)
    _append_page_carousel(text, total_pages, page_index)


def _append_page_carousel(text: Text, total_pages: int, page_index: int) -> None:
    current_page = page_index + 1
    text.append("<", style=_page_arrow_style(current_page > 1))
    if current_page > 1:
        text.append(" ")
        text.append(
            f" {current_page - 1} ",
            style="#8fbcbb on #3b4252",
        )
    text.append(" ")
    text.append(
        f" {current_page} ",
        style="bold #2e3440 on #a3be8c",
    )
    if current_page < total_pages:
        text.append(" ")
        text.append(
            f" {current_page + 1}",
            style="#a3be8c on #3b4252",
        )
    text.append(" ")
    text.append(">", style=_page_arrow_style(current_page < total_pages))


def _page_carousel_renderable(
    total_pages: int,
    page_index: int,
    width: int,
) -> Text:
    text = Text()
    _append_centered_page_carousel(text, total_pages, page_index, width)
    return text


def _page_arrow_style(enabled: bool) -> str:
    if enabled:
        return "bold #8fbcbb"
    return "#4c566a"


def _subagent_counts(subagents: list[TUISubagentState]) -> str:
    child_subagents = _child_subagent_panels(subagents)
    running = _subagent_count(child_subagents, "running")
    completed = _subagent_count(child_subagents, "completed")
    failed = _subagent_count(child_subagents, "failed")
    return (
        f"{len(child_subagents)} total | {running} running | "
        f"{completed} done | {failed} failed"
    )


def _subagent_count(subagents: list[TUISubagentState], status: str) -> int:
    count = 0
    for subagent in subagents:
        if subagent.status == status:
            count += 1
    return count


def _is_main_agent_panel(subagent: TUISubagentState) -> bool:
    return subagent.thread_id == MAIN_THREAD_ID


def _has_main_agent_panel(subagents: list[TUISubagentState]) -> bool:
    for subagent in subagents:
        if _is_main_agent_panel(subagent):
            return True
    return False


def _child_subagent_panels(
    subagents: list[TUISubagentState],
) -> list[TUISubagentState]:
    return [
        subagent
        for subagent in subagents
        if not _is_main_agent_panel(subagent)
    ]


def _append_subagent_card(text: Text, index: int, subagent: TUISubagentState) -> None:
    text.append("\n\n")
    text.append(f"#{index + 1} ", style="#81a1c1")
    text.append(
        _subagent_status_badge(subagent.status),
        style=_subagent_status_style(subagent.status),
    )
    text.append(" ")
    text.append(_subagent_label(subagent), style="bold #eceff4")
    _append_agent_metadata(text, subagent)
    _append_collection(text, "evidence", subagent.important_evidence)
    _append_tool_results(text, subagent.tool_results)
    if subagent.error is not None:
        _append_detail(text, "error", subagent.error, style="#bf616a")


def _subagent_overview(subagents: list[TUISubagentState]) -> RenderableType:
    overview = Table.grid(expand=True)
    overview.add_column()
    title = Text("subagents", style="bold #8fbcbb")
    child_subagents = _child_subagent_panels(subagents)
    if _has_main_agent_panel(subagents):
        title.append("  main", style="bold #ebcb8b")
        if child_subagents:
            title.append(" +", style="#4c566a")
    title.append("  ")
    for subagent in child_subagents:
        title.append(_subagent_status_mark(subagent.status), style=_subagent_dot_style(subagent.status))
        title.append(" ")
    overview.add_row(title)
    return _section_panel("", overview, padding=(0, 0))


def _subagent_title(index: int, subagent: TUISubagentState) -> Text:
    if _is_main_agent_panel(subagent):
        return _main_agent_title(index=index)
    title = Text()
    title.append(f"#{index + 1} ", style="#81a1c1")
    title.append(_subagent_status_badge(subagent.status), style=_subagent_status_style(subagent.status))
    title.append(" ")
    title.append(_subagent_label(subagent), style="bold #eceff4")
    return title


def _main_agent_title(index: int) -> Text:
    title = Text()
    title.append(f"#{index + 1} ", style="#81a1c1")
    title.append("< ", style="bold #ebcb8b")
    title.append("main agent", style="bold #ebcb8b")
    title.append("  parent thread", style="#9aa3b2")
    return title


def _main_agent_panel(subagent: TUISubagentState) -> RenderableType:
    rows: list[RenderableType] = [
        _field_block("role", "parent conversation", value_style="#ebcb8b"),
        _field_block("action", "activate returns to main", value_style="#d8dee9"),
    ]
    if subagent.thread_id != "":
        rows.append(_id_row("thread", subagent.thread_id))
    if subagent.agent_id != "":
        rows.append(_id_row("agent", subagent.agent_id))
    return _section_panel("main", Group(*rows), border_style="#ebcb8b")


def _main_agent_text_lines(index: int, subagent: TUISubagentState) -> list[str]:
    lines = [
        "",
        f"#{index + 1} < main agent",
        "   role: parent conversation",
        "   action: activate returns to main",
    ]
    if subagent.thread_id != "":
        lines.append(f"   thread {subagent.thread_id}")
    if subagent.agent_id != "":
        lines.append(f"   agent {subagent.agent_id}")
    return lines


def _subagent_status_panel(subagent: TUISubagentState) -> RenderableType:
    metrics = Table.grid(expand=True)
    metrics.add_column(ratio=1)
    metrics.add_column(ratio=1)
    metrics.add_column(ratio=1)
    metrics.add_column(ratio=1)
    metrics.add_row(
        _metric_block("state", _subagent_status_label(subagent.status), _subagent_status_style(subagent.status)),
        _metric_block("steps", f"{subagent.step_count}", "#d8dee9"),
        _metric_block("results", f"{_subagent_tool_result_count(subagent)}", "#d8dee9"),
        _metric_block("inbox", f"{subagent.inbox_pending_count}", "#ebcb8b"),
    )
    return _section_panel("run", metrics)


def _metric_block(label: str, value: str, value_style: str) -> RenderableType:
    block = Table.grid()
    block.add_column()
    block.add_row(Text(label, style="#88c0d0"))
    block.add_row(Text(value, style=value_style))
    return block


def _subagent_task_panel(subagent: TUISubagentState) -> RenderableType:
    rows: list[RenderableType] = []
    if subagent.context_brief != "":
        rows.append(_field_block("context", subagent.context_brief))
    if subagent.instructions != "":
        rows.append(_field_block("ask", subagent.instructions))
    if subagent.summary != "":
        rows.append(_field_block("summary", subagent.summary, value_style="#a3be8c"))
    if subagent.inbox_latest != "":
        rows.append(_field_block("inbox", subagent.inbox_latest, value_style="#ebcb8b"))
    if not rows:
        return Text("")
    return _section_panel("task", Group(*rows))


def _subagent_access_panel(subagent: TUISubagentState) -> RenderableType:
    tools = Table.grid(expand=True)
    tools.add_column()
    for tool_name in _allowed_tool_lines(subagent):
        line = Text()
        line.append("• ", style="#4c566a")
        line.append(tool_name, style="#d8dee9")
        tools.add_row(line)
    return _section_panel("tools", tools)


def _subagent_identity_panel(subagent: TUISubagentState) -> RenderableType:
    rows: list[RenderableType] = []
    if subagent.agent_id != "":
        rows.append(_id_row("agent", subagent.agent_id))
    if subagent.thread_id != "":
        rows.append(_id_row("thread", subagent.thread_id))
    if subagent.run_id != "":
        rows.append(_id_row("run", subagent.run_id))
    if not rows:
        return Text("")
    return _section_panel("ids", Group(*rows))


def _subagent_evidence_panel(subagent: TUISubagentState) -> RenderableType:
    if not subagent.important_evidence:
        return Text("")
    rows = []
    for value in subagent.important_evidence:
        rows.append(_bullet_line(value))
    return _section_panel("evidence", Group(*rows))


def _subagent_tool_results_panel(
    tool_results: list[TUISubagentToolResult],
) -> RenderableType:
    if not tool_results:
        return Text("")
    rows: list[RenderableType] = []
    for index, result in enumerate(tool_results):
        heading = Text()
        heading.append(f"#{index + 1} ", style="#81a1c1")
        heading.append(result.tool_name, style="bold #d8dee9")
        heading.append(f"  {result.call_id}", style="#9aa3b2")
        rows.append(heading)
        rows.append(_field_block("output", result.output))
        if result.error is not None:
            rows.append(_field_block("error", result.error, value_style="#bf616a"))
    return _section_panel("tool results", Group(*rows))


def _subagent_error_panel(error: str | None) -> RenderableType:
    if error is None:
        return Text("")
    return _section_panel("error", Text(error, style="#bf616a"), border_style="#bf616a")


def _field_block(
    label: str,
    value: str,
    *,
    value_style: str = "#d8dee9",
) -> RenderableType:
    block = Table.grid(expand=True)
    block.add_column()
    block.add_row(Text(label, style="bold #88c0d0"))
    block.add_row(Text(value, style=value_style))
    return block


def _id_row(label: str, value: str) -> RenderableType:
    row = Table.grid(expand=True)
    row.add_column(width=8)
    row.add_column(ratio=1)
    row.add_row(Text(label, style="#88c0d0"), Text(value, style="#9aa3b2"))
    return row


def _bullet_line(value: str) -> Text:
    line = Text()
    line.append("• ", style="#4c566a")
    line.append(value, style="#d8dee9")
    return line


def _section_panel(
    title: str,
    renderable: RenderableType,
    *,
    border_style: str = "#4c566a",
    padding: tuple[int, int] = (0, 1),
) -> RenderableType:
    return Panel(
        renderable,
        box=box.ROUNDED,
        title=Text(title, style="bold #8fbcbb") if title != "" else None,
        title_align="left",
        border_style=border_style,
        padding=padding,
    )


def _subagent_status_badge(status: str) -> str:
    if status == "running":
        return "↻"
    if status == "completed":
        return "✓"
    if status == "failed":
        return "✕"
    if status == "blocked":
        return "!"
    if status == "cancelled":
        return "-"
    if status == "pending":
        return "○"
    return "•"


def _subagent_status_mark(status: str) -> str:
    if status == "completed":
        return "✓"
    if status == "failed":
        return "✕"
    if status == "blocked":
        return "!"
    if status == "cancelled":
        return "-"
    if status == "pending":
        return "○"
    return "↻"


def _subagent_status_label(status: str) -> str:
    if status == "completed":
        return "✓ done"
    if status == "failed":
        return "✕ failed"
    if status == "pending":
        return "○ pending"
    if status == "running":
        return "↻ running"
    if status == "blocked":
        return "! blocked"
    if status == "cancelled":
        return "- cancelled"
    return status


def _subagent_dot_style(status: str) -> str:
    if status == "running":
        return "bold #ebcb8b"
    if status == "completed":
        return "bold #a3be8c"
    if status == "failed":
        return "bold #bf616a"
    if status == "blocked":
        return "bold #d08770"
    if status == "cancelled":
        return "bold #9aa3b2"
    return "bold #88c0d0"


def _subagent_summary_lines(subagent: TUISubagentState) -> list[str]:
    lines = [
        "   run",
        f"   status: {subagent.status}",
        f"   steps: {subagent.step_count}",
        f"   results: {_subagent_tool_result_count(subagent)}",
        f"   inbox: {subagent.inbox_pending_count}",
    ]
    lines.append("   tools")
    for tool_name in _allowed_tool_lines(subagent):
        lines.append(f"     - {tool_name}")
    if subagent.agent_id != "":
        lines.append("   ids")
        lines.append(f"   agent {subagent.agent_id}")
    if subagent.thread_id != "":
        lines.append(f"   thread {subagent.thread_id}")
    if subagent.run_id != "":
        lines.append(f"   run {subagent.run_id}")
    if subagent.context_brief != "":
        lines.append("   task / context")
        lines.append(f"     {subagent.context_brief}")
    if subagent.instructions != "":
        lines.append("   task / ask")
        lines.append(f"     {subagent.instructions}")
    if subagent.summary != "":
        lines.append("   task / summary")
        lines.append(f"     {subagent.summary}")
    if subagent.inbox_latest != "":
        lines.append("   task / inbox")
        lines.append(f"     {subagent.inbox_latest}")
    return lines


def _append_agent_metadata(text: Text, subagent: TUISubagentState) -> None:
    text.append("\n")
    text.append("   ")
    text.append("╭", style="#4c566a")
    text.append(" metadata ", style="bold #8fbcbb")
    text.append("─" * 21, style="#4c566a")
    text.append("╮", style="#4c566a")
    _append_metadata_pair(
        text,
        "status",
        subagent.status,
        "steps",
        str(subagent.step_count),
    )
    if _subagent_tool_result_count(subagent) > 0:
        _append_metadata_pair(
            text,
            "results",
            str(_subagent_tool_result_count(subagent)),
            "",
            "",
        )
    if subagent.inbox_pending_count > 0:
        _append_metadata_pair(
            text,
            "inbox",
            str(subagent.inbox_pending_count),
            "",
            "",
        )
    _append_metadata_gap(text)
    _append_metadata_heading(text, "tools")
    for tool_name in _allowed_tool_lines(subagent):
        _append_metadata_bullet(text, tool_name)
    if subagent.agent_id != "":
        _append_metadata_gap(text)
        _append_metadata_heading(text, "ids")
        _append_metadata_line(text, "agent", subagent.agent_id)
    if subagent.thread_id != "":
        _append_metadata_line(text, "thread", subagent.thread_id)
    if subagent.run_id != "":
        _append_metadata_line(text, "run", subagent.run_id)
    if subagent.context_brief != "":
        _append_metadata_gap(text)
        _append_metadata_heading(text, "context")
        _append_metadata_text(text, subagent.context_brief, "#d8dee9")
    if subagent.instructions != "":
        _append_metadata_gap(text)
        _append_metadata_heading(text, "ask")
        _append_metadata_text(text, subagent.instructions, "#d8dee9")
    if subagent.summary != "":
        _append_metadata_gap(text)
        _append_metadata_heading(text, "summary")
        _append_metadata_text(text, subagent.summary, "#a3be8c")
    if subagent.inbox_latest != "":
        _append_metadata_gap(text)
        _append_metadata_heading(text, "inbox")
        _append_metadata_text(text, subagent.inbox_latest, "#ebcb8b")
    text.append("\n")
    text.append("   ")
    text.append("╰", style="#4c566a")
    text.append("─" * 28, style="#4c566a")
    text.append("╯", style="#4c566a")


def _append_metadata_pair(
    text: Text,
    left_label: str,
    left_value: str,
    right_label: str,
    right_value: str,
) -> None:
    text.append("\n")
    text.append("   ")
    text.append("│ ", style="#4c566a")
    text.append(left_label + " ", style="#88c0d0")
    text.append(left_value, style="#d8dee9")
    if right_label != "":
        text.append("   ")
        text.append(right_label + " ", style="#88c0d0")
        text.append(right_value, style="#d8dee9")


def _append_metadata_line(text: Text, label: str, value: str) -> None:
    text.append("\n")
    text.append("   ")
    text.append("│ ", style="#4c566a")
    text.append(label + " ", style="#88c0d0")
    text.append(value, style="#9aa3b2")


def _append_metadata_gap(text: Text) -> None:
    text.append("\n")
    text.append("   ")
    text.append("│", style="#4c566a")


def _append_metadata_heading(text: Text, label: str) -> None:
    text.append("\n")
    text.append("   ")
    text.append("│ ", style="#4c566a")
    text.append(label, style="bold #88c0d0")


def _append_metadata_bullet(text: Text, value: str) -> None:
    text.append("\n")
    text.append("   ")
    text.append("│   - ", style="#4c566a")
    text.append(value, style="#d8dee9")


def _append_metadata_text(text: Text, value: str, style: str) -> None:
    text.append("\n")
    text.append("   ")
    text.append("│   ", style="#4c566a")
    text.append(value, style=style)


def _subagent_label(subagent: TUISubagentState) -> str:
    label = subagent.task
    if label == "":
        label = subagent.call_id
    return label


def _allowed_tools_label(subagent: TUISubagentState) -> str:
    if not subagent.allowed_tools:
        return "none"
    return ", ".join(subagent.allowed_tools)


def _allowed_tool_lines(subagent: TUISubagentState) -> list[str]:
    if not subagent.allowed_tools:
        return ["none"]
    return subagent.allowed_tools


def _subagent_tool_result_count(subagent: TUISubagentState) -> int:
    if subagent.tool_result_count > 0:
        return subagent.tool_result_count
    return len(subagent.tool_results)


def _append_detail(text: Text, label: str, value: str, *, style: str) -> None:
    if value == "":
        return
    text.append("\n   ")
    text.append(label + ": ", style="#88c0d0")
    text.append(value, style=style)


def _append_collection(text: Text, label: str, values: list[str]) -> None:
    if not values:
        return
    text.append("\n   ")
    text.append(label + ":", style="#88c0d0")
    for value in values:
        text.append("\n   - ", style="#4c566a")
        text.append(value, style="#d8dee9")


def _append_tool_results(text: Text, tool_results: list[TUISubagentToolResult]) -> None:
    if not tool_results:
        return
    text.append("\n   ")
    text.append("tool results:", style="#88c0d0")
    for index, result in enumerate(tool_results):
        text.append("\n   - ", style="#4c566a")
        text.append(
            f"#{index + 1} {result.tool_name} {result.call_id}",
            style="#d8dee9",
        )
        text.append("\n     output: ", style="#88c0d0")
        text.append(result.output, style="#d8dee9")
        if result.error is not None:
            text.append("\n     error: ", style="#bf616a")
            text.append(result.error, style="#bf616a")


def _detail_lines(label: str, value: str) -> list[str]:
    if value == "":
        return []
    return [f"   {label}: {value}"]


def _collection_lines(label: str, values: list[str]) -> list[str]:
    if not values:
        return []
    lines = [f"   {label}:"]
    for value in values:
        lines.append(f"   - {value}")
    return lines


def _tool_result_lines(tool_results: list[TUISubagentToolResult]) -> list[str]:
    if not tool_results:
        return []
    lines = ["   tool results:"]
    for index, result in enumerate(tool_results):
        lines.append(f"   - #{index + 1} {result.tool_name} {result.call_id}")
        lines.append(f"     output: {result.output}")
        if result.error is not None:
            lines.append(f"     error: {result.error}")
    return lines


def _subagent_status_style(status: str) -> str:
    if status == "running":
        return "bold #ebcb8b"
    if status == "completed":
        return "bold #a3be8c"
    if status == "failed":
        return "bold #bf616a"
    if status == "blocked":
        return "bold #d08770"
    if status == "cancelled":
        return "bold #9aa3b2"
    return "bold #88c0d0"
