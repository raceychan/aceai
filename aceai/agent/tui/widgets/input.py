"""Command input widget for the read-only TUI prototype."""

from textual.widgets import Input


class CommandInput(Input):
    """Bottom input bar for interactive TUI runs."""

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
