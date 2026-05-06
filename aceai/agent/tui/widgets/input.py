"""Command input widget for the read-only TUI prototype."""

from textual.events import Key
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

    def action_exit_input_mode(self) -> None:
        """Return focus to the main stream so app-level shortcuts work."""
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
