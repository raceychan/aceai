from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aceai import AgentBase
from aceai.events import (
    AgentEvent,
    LLMCompletedEvent,
    LLMOutputDeltaEvent,
    LLMStartedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ToolOutputEvent,
    ToolStartedEvent,
)


@dataclass
class ToolView:
    """In-memory representation of a tool invocation for terminal rendering."""

    call_id: str
    name: str
    status: str = "running"
    buffer: list[str] = field(default_factory=list)
    error: str = ""

    def status_text(self) -> str:
        if self.status == "completed":
            return "[green]completed[/]"
        if self.status == "failed":
            return "[red]failed[/]"
        return "[yellow]running[/]"

    def output_text(self) -> str:
        if self.error:
            return f"[red]{self.error}[/red]"
        if not self.buffer:
            return "[dim]waiting for output[/dim]"
        return "".join(self.buffer)


class TerminalRunState:
    """Tracks event-driven state that fuels the terminal UI render loop."""

    def __init__(self, question: str):
        self.question = question
        self.step_index = -1
        self.reasoning_log: list[str] = []
        self.tools: dict[str, ToolView] = {}
        self.tool_order: list[str] = []
        self.final_answer = ""
        self.error_message = ""
        self.last_event_type = ""

    def apply(self, event: AgentEvent) -> None:
        self.last_event_type = event.event_type
        if isinstance(event, LLMStartedEvent):
            self.step_index = event.step_index
            self.reasoning_log = []
            self.tools = {}
            self.tool_order = []
            return

        if isinstance(event, LLMOutputDeltaEvent):
            self.reasoning_log.append(event.text_delta)
            return

        if isinstance(event, LLMCompletedEvent):
            return

        if isinstance(event, ToolStartedEvent):
            view = ToolView(
                call_id=event.tool_call.call_id,
                name=event.tool_name,
            )
            self.tools[view.call_id] = view
            self.tool_order.append(view.call_id)
            return

        if isinstance(event, ToolOutputEvent):
            view = self.tools.get(event.tool_call.call_id)
            if view is not None:
                view.buffer.append(event.text_delta)
            return

        if isinstance(event, ToolCompletedEvent):
            view = self.tools.get(event.tool_call.call_id)
            if view is not None:
                view.status = "completed"
                view.buffer = [event.tool_result.output]
                view.error = ""
            return

        if isinstance(event, ToolFailedEvent):
            view = self.tools.get(event.tool_call.call_id)
            if view is None:
                view = ToolView(
                    call_id=event.tool_call.call_id,
                    name=event.tool_name,
                )
                self.tools[view.call_id] = view
                self.tool_order.append(view.call_id)
            view.status = "failed"
            view.error = event.error or event.tool_result.error or ""
            return

        if isinstance(event, StepCompletedEvent):
            self.step_index = event.step_index
            return

        if isinstance(event, StepFailedEvent):
            self.step_index = event.step_index
            self.error_message = event.error
            return

        if isinstance(event, RunCompletedEvent):
            self.final_answer = event.final_answer
            return

        if isinstance(event, RunFailedEvent):
            self.error_message = event.error

    def render(self) -> Group:
        question_panel = Panel(
            self.question or "[dim]no question provided[/dim]",
            title="Question",
            border_style="cyan",
        )
        step_label = "Reasoning"
        if self.step_index >= 0:
            step_label = f"Step {self.step_index + 1} · Reasoning"
        reasoning_text = "".join(self.reasoning_log) or "[dim]waiting on the model[/dim]"
        reasoning_panel = Panel(
            reasoning_text,
            title=step_label,
            border_style="magenta",
        )

        if not self.tool_order:
            tools_body = Text("no tool activity yet", style="dim")
        else:
            tools_table = Table(
                show_header=True,
                header_style="bold yellow",
                expand=True,
            )
            tools_table.add_column("Tool")
            tools_table.add_column("Status", style="white")
            tools_table.add_column("Details", style="white", overflow="fold")
            for call_id in self.tool_order:
                view = self.tools[call_id]
                tool_label = f"{view.name}\n[dim]{view.call_id}"
                tools_table.add_row(tool_label, view.status_text(), view.output_text())
            tools_body = tools_table

        tools_panel = Panel(tools_body, title="Tools", border_style="yellow")

        status_lines = []
        if self.last_event_type:
            status_lines.append(f"event: {self.last_event_type}")
        if self.final_answer:
            status_lines.append("run: completed")
        if self.error_message:
            status_lines.append(f"error: {self.error_message}")
        if not status_lines:
            status_lines.append("idle")
        status_panel = Panel(
            "\n".join(status_lines),
            title="Status",
            border_style="green" if self.final_answer else "red" if self.error_message else "blue",
        )

        panels: list[Panel] = [
            question_panel,
            reasoning_panel,
            tools_panel,
            status_panel,
        ]

        if self.final_answer:
            panels.append(
                Panel(
                    self.final_answer,
                    title="Final Answer",
                    border_style="green",
                )
            )
        if self.error_message and not self.final_answer:
            panels.append(
                Panel(
                    self.error_message,
                    title="Failure",
                    border_style="red",
                )
            )

        return Group(*panels)


async def run_agent_with_terminal_ui(
    agent: AgentBase,
    question: str,
) -> str:
    """Stream an agent run through a Rich-powered terminal interface."""
    active_console = Console()
    state = TerminalRunState(question)
    active_console.rule("[bold cyan]AceAI Terminal Client[/bold cyan]")
    with Live(
        state.render(),
        console=active_console,
        refresh_per_second=8,
        transient=False,
    ) as live:
        async for event in agent.run(question):
            state.apply(event)
            live.update(state.render(), refresh=True)
            if isinstance(event, RunCompletedEvent):
                return event.final_answer
    raise RuntimeError("Agent terminated without RunCompletedEvent")
