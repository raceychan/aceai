"""Command input widget for the read-only TUI prototype."""

from dataclasses import dataclass
from typing import Any, cast

from rich.cells import cell_len, set_cell_size
from rich.style import Style
from rich.text import Text
from textual.events import Click, Key
from textual.message import Message
from textual.widgets import Static, TextArea

from aceai.agent.citations import TurnCitation

CITATION_PREVIEW_LINES = 3
CITATION_PREVIEW_WIDTH = 88


@dataclass(frozen=True)
class CommandCompletionItem:
    command: str
    description: str


class CommandInput(TextArea):
    """Bottom input bar for interactive TUI runs."""

    class Submitted(Message):
        def __init__(self, input: "CommandInput", value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()

    class CompletionRequested(Message):
        def __init__(self, input: "CommandInput") -> None:
            self.input = input
            super().__init__()

    class CompletionNavigationRequested(Message):
        def __init__(self, *, direction: int) -> None:
            self.direction = direction
            super().__init__()

    BINDINGS = [
        ("escape", "exit_input_mode", "Exit input"),
        ("enter", "submit_or_complete", "Submit"),
        ("ctrl+m", "submit_or_complete", "Submit"),
    ]

    DEFAULT_CSS = """
    CommandInput {
        height: 4;
        background: #3b4252;
        color: #eceff4;
        border: round #8fbcbb;
    }

    CommandInput:focus {
        background: #3b4252;
        border: round #8fbcbb;
    }

    CommandInput .text-area--cursor-line {
        background: transparent;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(
            placeholder="Ask AceAI or type /quit",
            id=id,
            show_line_numbers=False,
            soft_wrap=True,
        )

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.load_text(text)
        lines = text.splitlines()
        if not lines:
            self.cursor_location = (0, 0)
            return
        self.cursor_location = (len(lines) - 1, len(lines[-1]))

    async def _on_key(self, event: Key) -> None:
        if event.key == "shift+enter":
            self.insert("\n")
            event.stop()
            event.prevent_default()
            return
        if event.key in ("enter", "ctrl+m"):
            self._submit_or_complete()
            event.stop()
            event.prevent_default()
            return
        if _is_slash_command_selection(self.text) and event.key in ("up", "down"):
            direction = -1 if event.key == "up" else 1
            self.post_message(self.CompletionNavigationRequested(direction=direction))
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    def on_key(self, event: Key) -> None:
        if event.key == "escape" and cast(Any, self.app).cancel_active_run():
            event.stop()

    def action_exit_input_mode(self) -> None:
        """Return focus to the main stream so app-level shortcuts work."""
        if cast(Any, self.app).cancel_active_run():
            return
        self.blur()
        self.app.query_one("#stream").focus()

    def action_submit_or_complete(self) -> None:
        self._submit_or_complete()

    def _submit_or_complete(self) -> None:
        if _is_slash_command_selection(self.text):
            self.post_message(self.CompletionRequested(self))
            return
        self.post_message(self.Submitted(self, self.text))


class CommandCompletionWidget(Static):
    """Inline slash-command completion hints."""

    DEFAULT_CSS = """
    CommandCompletionWidget {
        height: auto;
        max-height: 16;
        padding: 0 1;
        background: #2e3440;
        color: #d8dee9;
        border: round #5e81ac;
    }

    CommandCompletionWidget.hidden {
        display: none;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display_text = ""
        self.selected_index = 0

    def show_commands(
        self,
        items: list[CommandCompletionItem],
        *,
        selected_index: int = 0,
    ) -> None:
        if not items:
            self.add_class("hidden")
            self.display_text = ""
            self.selected_index = 0
            self.update("")
            return
        self.remove_class("hidden")
        self.selected_index = max(0, min(selected_index, len(items) - 1))
        command_width = max(len(f"/{item.command}") for item in items)
        lines: list[str] = []
        for index, item in enumerate(items):
            command = f"/{item.command}".ljust(command_width)
            marker = ">" if index == self.selected_index else " "
            lines.append(f"{marker} {command}  {item.description}")
        self.display_text = "\n".join(lines)
        self.update(self.display_text)

    def hide(self) -> None:
        self.add_class("hidden")
        self.display_text = ""
        self.selected_index = 0
        self.update("")

    @property
    def renderable(self) -> str:
        return self.display_text


class QueuedTurnsWidget(Static):
    """Clickable queued messages shown above the input bar."""

    class Selected(Message):
        def __init__(self, *, index: int) -> None:
            super().__init__()
            self.index = index

    class Cancelled(Message):
        def __init__(self, *, index: int) -> None:
            super().__init__()
            self.index = index

    DEFAULT_CSS = """
    QueuedTurnsWidget {
        height: auto;
        max-height: 5;
        padding: 0 1;
        background: #3b4252;
        color: #d8dee9;
        border: round #5e81ac;
    }

    QueuedTurnsWidget.hidden {
        display: none;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._questions: tuple[str, ...] = ()
        self.display_text = ""
        self.display = False

    def set_questions(self, questions: tuple[str, ...]) -> None:
        self._questions = questions
        if not questions:
            self.display = False
            self.add_class("hidden")
            self.display_text = ""
            if self.is_mounted:
                self.update("")
            return
        self.display = True
        self.remove_class("hidden")
        plain_lines, renderable = _queued_turns_renderable(
            questions,
            width=_queued_content_width(self),
        )
        self.display_text = "\n".join(plain_lines)
        if self.is_mounted:
            self.update(renderable)

    @property
    def renderable(self) -> str:
        return self.display_text

    def on_click(self, event: Click) -> None:
        if event.style is None:
            return
        action = event.style.meta.get("queued_action")
        index = event.style.meta.get("queued_index")
        if type(index) is not int:
            return
        event.stop()
        if action == "cancel":
            self.post_message(self.Cancelled(index=index))
            return
        if action == "steer":
            self.post_message(self.Selected(index=index))


def _queued_turns_renderable(
    questions: tuple[str, ...],
    *,
    width: int,
) -> tuple[list[str], Text]:
    lines = ["queued messages"]
    renderable = Text("queued messages")
    for index, question in enumerate(questions):
        body, spacing = _queued_row_body(index, question, width=width)
        lines.append(f"{body}{spacing}[ > ] [ x ]")
        renderable.append("\n")
        renderable.append(body)
        renderable.append(spacing)
        renderable.append("[ > ]", style=_queued_action_style(index, "steer"))
        renderable.append(" ")
        renderable.append("[ x ]", style=_queued_action_style(index, "cancel"))
    return lines, renderable


def _queued_row_body(index: int, question: str, *, width: int) -> tuple[str, str]:
    first_line = question.splitlines()[0] if question.splitlines() else ""
    turn_number = index + 1
    body = f"{turn_number}. {first_line}"
    action_width = 11
    max_body_width = max(width - action_width - 2, 16)
    if cell_len(body) > max_body_width:
        body = set_cell_size(body, max_body_width - 3) + "..."
    spacing = " " * max(width - cell_len(body) - action_width, 2)
    return body, spacing


def _queued_content_width(widget: Static) -> int:
    content_width = widget.content_size.width
    if content_width > 0:
        return content_width
    widget_width = widget.size.width
    if widget_width > 4:
        return widget_width - 4
    return 0


def _queued_action_style(index: int, action: str) -> Style:
    return Style(
        bold=True,
        color="#eceff4",
        meta={
            "queued_action": action,
            "queued_index": index,
        },
    )


class CitationPreviewWidget(Static):
    """Read-only cited source preview shown above the input box."""

    DEFAULT_CSS = """
    CitationPreviewWidget {
        height: auto;
        min-height: 6;
        max-height: 6;
        margin-bottom: 1;
        padding: 0 1;
        background: #252c37;
        color: #eceff4;
        border: round #88c0d0;
    }

    CitationPreviewWidget.hidden {
        display: none;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._citations: tuple[TurnCitation, ...] = ()
        self.display_text = ""
        self.display = False

    def set_citations(self, citations: tuple[TurnCitation, ...]) -> None:
        self._citations = citations
        if not citations:
            self.display = False
            self.add_class("hidden")
            self.display_text = ""
            self.update("")
            return
        self.display = True
        self.remove_class("hidden")
        self.display_text = _citation_preview_text(citations)
        self.update(_citation_preview_renderable(self.display_text))

    @property
    def renderable(self) -> str:
        return self.display_text


def _citation_preview_label(citation: TurnCitation) -> str:
    return "\n".join(_citation_content_preview_lines(citation.content))


def _citation_preview_text(citations: tuple[TurnCitation, ...]) -> str:
    content = "\n".join(citation.content for citation in citations)
    return "\n".join(["cited source"] + _citation_content_preview_lines(content))


def _citation_preview_renderable(display_text: str) -> Text:
    lines = display_text.splitlines()
    text = Text()
    for index, line in enumerate(lines):
        if index > 0:
            text.append("\n")
        if index == 0:
            text.append(line, style="bold #8fbcbb")
            continue
        _append_citation_preview_line(text, line)
    return text


def _append_citation_preview_line(text: Text, line: str) -> None:
    marker = "...more"
    if not line.endswith(marker):
        text.append(line, style="#eceff4")
        return
    body = line[: -len(marker)]
    text.append(body, style="#eceff4")
    text.append(marker, style="bold #ebcb8b")


def _citation_content_preview_lines(content: str) -> list[str]:
    chunks = _wrap_citation_preview_text(content)
    truncated = len(chunks) > CITATION_PREVIEW_LINES
    preview = chunks[:CITATION_PREVIEW_LINES]
    while len(preview) < CITATION_PREVIEW_LINES:
        preview.append(" ")
    if truncated:
        preview[-1] = _citation_more_line(preview[-1])
    return preview


def _wrap_citation_preview_text(content: str) -> list[str]:
    if content == "":
        return [" "]
    wrapped: list[str] = []
    for source_line in content.splitlines():
        if source_line == "":
            continue
        line = source_line
        while line != "":
            if cell_len(line) <= CITATION_PREVIEW_WIDTH:
                wrapped.append(line)
                break
            wrapped.append(set_cell_size(line, CITATION_PREVIEW_WIDTH).rstrip())
            line = line[CITATION_PREVIEW_WIDTH:]
    return wrapped


def _citation_more_line(value: str) -> str:
    suffix = " ...more"
    return f"{set_cell_size(value, CITATION_PREVIEW_WIDTH - len(suffix)).rstrip()}{suffix}"


def _is_slash_command_selection(value: str) -> bool:
    if not value.startswith("/"):
        return False
    body = value.removeprefix("/")
    return " " not in body and "\n" not in body
