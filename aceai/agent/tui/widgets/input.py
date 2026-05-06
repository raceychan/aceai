"""Command input widget for the read-only TUI prototype."""

from textual.events import Click, Key
from textual.message import Message
from textual.widgets import Static, TextArea


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

    BINDINGS = [
        ("escape", "exit_input_mode", "Exit input"),
    ]

    DEFAULT_CSS = """
    CommandInput {
        height: 3;
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

    def on_key(self, event: Key) -> None:
        if event.key == "shift+enter":
            self.insert("\n")
            event.stop()
            return
        if event.key == "enter":
            self.post_message(self.Submitted(self, self.text))
            event.stop()
            return
        if event.key == "tab" and self.text.startswith("/"):
            self.post_message(self.CompletionRequested(self))
            event.stop()
            return
        if event.key == "escape" and self.app.cancel_active_run():
            event.stop()

    def action_exit_input_mode(self) -> None:
        """Return focus to the main stream so app-level shortcuts work."""
        if self.app.cancel_active_run():
            return
        self.blur()
        self.app.query_one("#stream").focus()


class CommandCompletionWidget(Static):
    """Inline slash-command completion hints."""

    DEFAULT_CSS = """
    CommandCompletionWidget {
        height: auto;
        max-height: 4;
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

    def show_commands(self, commands: list[str]) -> None:
        if not commands:
            self.add_class("hidden")
            self.display_text = ""
            self.update("")
            return
        self.remove_class("hidden")
        self.display_text = "  ".join(f"/{command}" for command in commands)
        self.update(self.display_text)

    def hide(self) -> None:
        self.add_class("hidden")
        self.display_text = ""
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
