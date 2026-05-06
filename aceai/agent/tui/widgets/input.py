"""Command input widget for the read-only TUI prototype."""

from dataclasses import dataclass
from typing import Any, cast

from textual.events import Click, Key
from textual.message import Message
from textual.widgets import Static, TextArea


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
        border: solid #88c0d0;
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
        border: tall #4c566a;
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

    DEFAULT_CSS = """
    QueuedTurnsWidget {
        height: auto;
        max-height: 5;
        padding: 0 1;
        background: #3b4252;
        color: #d8dee9;
        border-top: solid #5e81ac;
        border-bottom: none;
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
            self.update("")
            return
        self.display = True
        self.remove_class("hidden")
        self.display_text = "\n".join(
            ["queued - click a line to steer"]
            + [
                _queued_button_label(index, question)
                for index, question in enumerate(questions)
            ]
        )
        self.update(self.display_text)

    @property
    def renderable(self) -> str:
        return self.display_text

    def on_click(self, event: Click) -> None:
        index = event.y - 1
        if index < 0:
            return
        if index >= len(self._questions):
            return
        event.stop()
        self.post_message(self.Selected(index=index))


def _queued_button_label(index: int, question: str) -> str:
    first_line = question.splitlines()[0] if question.splitlines() else ""
    if len(first_line) > 96:
        first_line = f"{first_line[:93]}..."
    return f"{index + 1}. {first_line}"


def _is_slash_command_selection(value: str) -> bool:
    if not value.startswith("/"):
        return False
    body = value.removeprefix("/")
    return " " not in body and "\n" not in body
