"""Status bar for the AceAI TUI."""

from textual.widgets import Static

from aceai.agent.tui.state import TUIRunStatus


class StatusBarWidget(Static):
    """Show persistent runtime state such as the selected model."""

    DEFAULT_CSS = """
    StatusBarWidget {
        height: 1;
        background: #3b4252;
        color: #eceff4;
        padding: 0 1;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.current_text = ""

    def set_status(self, *, model: str | None, status: TUIRunStatus) -> None:
        model_text = model if model is not None else "unconfigured"
        self.current_text = f"model: {model_text}   status: {status}"
        self.update(self.current_text)
