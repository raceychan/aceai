"""Inline approval controls for suspended tool calls."""

import json

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.events import Key
from textual.message import Message
from textual.widgets import Button, Static

from aceai.core.models import ToolApprovalRequest


class ApprovalWidget(Container):
    """Compact inline approval panel shown inside the chat surface."""

    class Selected(Message):
        def __init__(self, *, approved: bool) -> None:
            super().__init__()
            self.approved = approved

    DEFAULT_CSS = """
    ApprovalWidget {
        height: 3;
        background: #2e3440;
        border: round #4c566a;
        padding: 0 1;
    }

    ApprovalWidget.collapsed {
        display: none;
    }

    #approval-row {
        height: 1;
    }

    #approval-summary {
        width: 1fr;
        height: 1;
        color: #d8dee9;
        margin-left: 1;
    }

    #approval-actions {
        width: auto;
        height: 1;
    }

    #approval-approve,
    #approval-reject {
        width: auto;
        min-width: 10;
        height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
        text-style: bold;
    }

    #approval-approve {
        color: #a3be8c;
    }

    #approval-reject {
        color: #bf616a;
    }

    #approval-approve:focus,
    #approval-approve:hover {
        background: #3b4252;
        color: #a3be8c;
    }

    #approval-reject:focus,
    #approval-reject:hover {
        background: #3b4252;
        color: #bf616a;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="approval-row"):
            with Horizontal(id="approval-actions"):
                yield Button("A Approve", id="approval-approve")
                yield Button("R Reject", id="approval-reject")
            yield Static("", id="approval-summary")

    def show_request(self, request: ToolApprovalRequest) -> None:
        self.query_one("#approval-summary", Static).update(_approval_summary(request))
        self.remove_class("collapsed")
        self.query_one("#approval-approve", Button).focus()

    def clear_request(self) -> None:
        self.add_class("collapsed")
        self.query_one("#approval-summary", Static).update("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-approve":
            event.stop()
            self.post_message(self.Selected(approved=True))
            return
        if event.button.id == "approval-reject":
            event.stop()
            self.post_message(self.Selected(approved=False))

    def on_key(self, event: Key) -> None:
        if event.key == "a":
            event.stop()
            self.post_message(self.Selected(approved=True))
            return
        if event.key == "r":
            event.stop()
            self.post_message(self.Selected(approved=False))
            return
        if event.key == "right":
            event.stop()
            self.query_one("#approval-reject", Button).focus()
            return
        if event.key == "left":
            event.stop()
            self.query_one("#approval-approve", Button).focus()


def _approval_summary(request: ToolApprovalRequest) -> str:
    policy = request.policy or "approval required"
    arguments = _arguments_summary(request.call.arguments)
    return f"approval required  {request.tool_name}  {policy}  {arguments}"


def _arguments_summary(arguments: str) -> str:
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return _preview_line("arguments", arguments)
    if not isinstance(payload, dict):
        return _preview_line("arguments", arguments)
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str):
            lines.append(_preview_line(key, value))
        elif isinstance(value, int | float | bool):
            lines.append(f"{key}: {value!r}")
        elif value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value.__class__.__name__}")
    return "  ".join(lines)


def _preview_line(label: str, value: str) -> str:
    first_line = value.splitlines()[0] if value.splitlines() else ""
    if len(value) <= 96:
        return f"{label}: {value}"
    return f"{label}: {len(value)} chars, starts with {first_line[:72]}"
