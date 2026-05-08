"""Subagent activity tracker for delegated child-agent runs."""

from rich.text import Text
from textual.widgets import Static

from aceai.agent.tui.state import TUISubagentState

SUBAGENT_TASK_WIDTH = 88


class SubagentStatusWidget(Static):
    """Compact status area for delegate_to_subagent tool calls."""

    DEFAULT_CSS = """
    SubagentStatusWidget {
        height: auto;
        max-height: 5;
        padding: 0 1;
        background: #252c37;
        color: #d8dee9;
        border: round #5e81ac;
    }

    SubagentStatusWidget.hidden {
        display: none;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display_text = ""
        self.display = False

    def set_subagents(self, subagents: list[TUISubagentState]) -> None:
        if not subagents:
            self.display = False
            self.add_class("hidden")
            self.display_text = ""
            self.update("")
            return
        self.display = True
        self.remove_class("hidden")
        self.display_text = _subagent_status_text(subagents)
        self.update(_subagent_status_renderable(subagents))

    @property
    def renderable(self) -> str:
        return self.display_text


def _subagent_status_text(subagents: list[TUISubagentState]) -> str:
    return "\n".join(
        ["subagents"]
        + [
            f"{index + 1}. {subagent.status} - {_subagent_label(subagent)}"
            for index, subagent in enumerate(subagents)
        ]
    )


def _subagent_status_renderable(subagents: list[TUISubagentState]) -> Text:
    text = Text("subagents", style="bold #8fbcbb")
    for index, subagent in enumerate(subagents):
        text.append("\n")
        text.append(f"{index + 1}. ", style="#81a1c1")
        text.append(subagent.status, style=_subagent_status_style(subagent.status))
        text.append(" - ", style="#4c566a")
        text.append(_subagent_label(subagent), style="#eceff4")
    return text


def _subagent_label(subagent: TUISubagentState) -> str:
    label = subagent.task
    if label == "":
        label = subagent.call_id
    if len(label) <= SUBAGENT_TASK_WIDTH:
        return label
    return f"{label[: SUBAGENT_TASK_WIDTH - 3]}..."


def _subagent_status_style(status: str) -> str:
    if status == "running":
        return "bold #ebcb8b"
    if status == "completed":
        return "bold #a3be8c"
    if status == "failed":
        return "bold #bf616a"
    return "bold #88c0d0"
