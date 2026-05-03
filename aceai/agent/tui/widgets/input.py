"""Command input widget for the read-only TUI prototype."""

from textual.widgets import Input


class CommandInput(Input):
    """Bottom input bar for interactive TUI runs."""

    BINDINGS = [
        ("escape", "exit_input_mode", "Exit input"),
    ]

    DEFAULT_CSS = """
    CommandInput {
        dock: bottom;
        height: 3;
        background: #3b4252;
        color: #eceff4;
        border: solid #88c0d0;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(placeholder="Ask AceAI or type /quit", id=id)

    def action_exit_input_mode(self) -> None:
        """Return focus to the main stream so app-level shortcuts work."""
        self.blur()
        self.app.query_one("#stream").focus()
