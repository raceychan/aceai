"""Subagent activity tracker for delegated child-agent runs."""

from rich.text import Text
from textual.events import Key, MouseScrollDown, MouseScrollUp, Resize
from textual.widgets import RichLog

from aceai.agent.tui.state import TUISubagentState, TUISubagentToolResult


class SubagentStatusWidget(RichLog):
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
        overflow-y: auto;
        overflow-x: hidden;
    }

    SubagentStatusWidget.hidden {
        display: none;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, wrap=True, auto_scroll=False, **kwargs)
        self.display_text = ""
        self.display = False
        self.can_focus = True
        self._subagents: list[TUISubagentState] = []
        self._page_index = 0

    def set_subagents(self, subagents: list[TUISubagentState]) -> None:
        if not subagents:
            self.display = False
            self.add_class("hidden")
            self.display_text = ""
            self._subagents = []
            self._page_index = 0
            self.clear()
            return
        self.display = True
        self.remove_class("hidden")
        self._subagents = subagents
        if self._page_index >= len(subagents):
            self._page_index = len(subagents) - 1
        self._render_current_page()

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

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self.scroll_relative(y=3, animate=False, force=True, immediate=True)
        event.stop()

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self.scroll_relative(y=-3, animate=False, force=True, immediate=True)
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
        self.clear()
        if not self._subagents:
            return
        self.display_text = _subagent_status_text(self._subagents, self._page_index)
        width = max(1, self.scrollable_content_region.width)
        self.write(
            _subagent_status_renderable(self._subagents, self._page_index),
            width=width,
        )
        self.scroll_home(animate=False, force=True, immediate=True)


def _subagent_status_text(subagents: list[TUISubagentState], page_index: int) -> str:
    subagent = subagents[page_index]
    lines = [
        _subagent_header(subagents, page_index),
    ]
    lines.extend(_subagent_text_lines(page_index, subagent))
    return "\n".join(lines)


def _subagent_status_renderable(
    subagents: list[TUISubagentState],
    page_index: int,
) -> Text:
    subagent = subagents[page_index]
    text = Text()
    text.append("subagents", style="bold #8fbcbb")
    text.append("  ")
    text.append(_subagent_counts(subagents), style="#9aa3b2")
    text.append("\n")
    text.append(_page_label(subagents, page_index), style="#81a1c1")
    text.append("  ")
    text.append("use ←/→ or h/l", style="#4c566a")
    _append_subagent_card(text, page_index, subagent)
    return text


def _subagent_text_lines(index: int, subagent: TUISubagentState) -> list[str]:
    lines = [
        "",
        f"#{index + 1} [{subagent.status}] {_subagent_label(subagent)}",
        "   " + _subagent_meta(subagent),
    ]
    lines.extend(_detail_lines("brief", subagent.context_brief))
    lines.extend(_detail_lines("ask", subagent.instructions))
    lines.extend(_detail_lines("summary", subagent.summary))
    lines.extend(_collection_lines("evidence", subagent.important_evidence))
    lines.extend(_tool_result_lines(subagent.tool_results))
    if subagent.error is not None:
        lines.extend(_detail_lines("error", subagent.error))
    return lines


def _subagent_header(subagents: list[TUISubagentState], page_index: int) -> str:
    return f"subagents  {_subagent_counts(subagents)}  {_page_label(subagents, page_index)}"


def _page_label(subagents: list[TUISubagentState], page_index: int) -> str:
    return f"page {page_index + 1}/{len(subagents)}"


def _subagent_counts(subagents: list[TUISubagentState]) -> str:
    running = _subagent_count(subagents, "running")
    completed = _subagent_count(subagents, "completed")
    failed = _subagent_count(subagents, "failed")
    return (
        f"{len(subagents)} total | {running} running | "
        f"{completed} done | {failed} failed"
    )


def _subagent_count(subagents: list[TUISubagentState], status: str) -> int:
    count = 0
    for subagent in subagents:
        if subagent.status == status:
            count += 1
    return count


def _append_subagent_card(text: Text, index: int, subagent: TUISubagentState) -> None:
    text.append("\n\n")
    text.append(f"#{index + 1} ", style="#81a1c1")
    text.append(
        _subagent_status_badge(subagent.status),
        style=_subagent_status_style(subagent.status),
    )
    text.append(" ")
    text.append(_subagent_label(subagent), style="bold #eceff4")
    text.append("\n   ")
    text.append(_subagent_meta(subagent), style="#9aa3b2")
    _append_detail(text, "brief", subagent.context_brief, style="#d8dee9")
    _append_detail(text, "ask", subagent.instructions, style="#d8dee9")
    _append_detail(text, "summary", subagent.summary, style="#a3be8c")
    _append_collection(text, "evidence", subagent.important_evidence)
    _append_tool_results(text, subagent.tool_results)
    if subagent.error is not None:
        _append_detail(text, "error", subagent.error, style="#bf616a")


def _subagent_status_badge(status: str) -> str:
    if status == "running":
        return "[running]"
    if status == "completed":
        return "[done]"
    if status == "failed":
        return "[failed]"
    return f"[{status}]"


def _subagent_meta(subagent: TUISubagentState) -> str:
    parts = [
        f"tools {_allowed_tools_label(subagent)}",
        f"steps {subagent.step_count}",
    ]
    if subagent.tool_results:
        parts.append(f"tool results {len(subagent.tool_results)}")
    if subagent.run_id != "":
        parts.append(f"agent {subagent.agent_id}")
        parts.append(f"run {subagent.run_id}")
    return " | ".join(parts)


def _subagent_label(subagent: TUISubagentState) -> str:
    label = subagent.task
    if label == "":
        label = subagent.call_id
    return label


def _allowed_tools_label(subagent: TUISubagentState) -> str:
    if not subagent.allowed_tools:
        return "none"
    return ", ".join(subagent.allowed_tools)


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
    return "bold #88c0d0"
