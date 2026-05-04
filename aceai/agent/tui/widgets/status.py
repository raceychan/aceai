"""Status bar for the AceAI TUI."""

from rich.align import Align
from rich.text import Text
from textual.widgets import Static

from aceai.agent.cost import format_usd
from aceai.agent.tui.state import TUIRunStatus, TUIUsageState


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

    def set_status(
        self,
        *,
        model: str | None,
        status: TUIRunStatus,
        usage: TUIUsageState | None = None,
    ) -> None:
        model_text = model if model is not None else "unconfigured"
        usage_state = usage or TUIUsageState()
        context = _format_tokens(usage_state.current_context_tokens)
        cost = format_usd(usage_state.session_cost_usd)
        self.current_text = (
            f"status: {status}   model: {model_text}   ctx: {context}   cost: {cost}"
        )
        if status == "suspended":
            self.current_text = f"{self.current_text}   action: choose Approve or Reject"
        self.update(Align.right(Text(self.current_text)))


def _usage_text(usage: TUIUsageState) -> str:
    context = _format_tokens(usage.current_context_tokens)
    cached = _format_tokens(usage.session_cached_input_tokens)
    session = _format_tokens(usage.session_total_tokens)
    input_tokens = _format_tokens(usage.session_input_tokens)
    output_tokens = _format_tokens(usage.session_output_tokens)
    cost = format_usd(usage.session_cost_usd)
    return (
        f"ctx: {context}   session: {session} "
        f"({input_tokens} in / {cached} cached / {output_tokens} out)   cost: {cost}"
    )


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"
