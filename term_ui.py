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


@dataclass
class StepView:
    """Captures reasoning and tool activity for a single agent step."""

    index: int
    step_id: str
    reasoning: list[str] = field(default_factory=list)
    activity_log: list[str] = field(default_factory=list)
    tools: dict[str, ToolView] = field(default_factory=dict)
    tool_order: list[str] = field(default_factory=list)
    status: str = "running"
    error: str = ""

    def append_reasoning(self, chunk: str) -> None:
        if chunk:
            self.reasoning.append(chunk)

    def reasoning_text(self) -> str:
        return "".join(self.reasoning)

    def add_activity(self, entry: str) -> None:
        if not entry:
            return
        self.activity_log.append(entry)
        if len(self.activity_log) > 20:
            self.activity_log = self.activity_log[-20:]

    def activity_text(self) -> str:
        return "\n".join(self.activity_log)

    def status_text(self) -> str:
        if self.status == "completed":
            return "[green]completed[/]"
        if self.status == "failed":
            return "[red]failed[/]"
        return "[yellow]running[/]"

    def border_style(self) -> str:
        if self.status == "completed":
            return "green"
        if self.status == "failed":
            return "red"
        return "magenta"


class TerminalRunState:
    """Tracks event-driven state that fuels the terminal UI render loop."""

    def __init__(self, question: str):
        self.question = question
        self.steps: list[StepView] = []
        self.step_lookup: dict[str, StepView] = {}
        self.step_index = -1
        self.final_answer = ""
        self.error_message = ""
        self.event_history: list[str] = []

    def _get_step(self, step_id: str) -> StepView | None:
        return self.step_lookup.get(step_id)

    def apply(self, event: AgentEvent) -> None:
        self._record_event(event)
        if isinstance(event, LLMStartedEvent):
            step = StepView(index=event.step_index, step_id=event.step_id)
            self.steps.append(step)
            self.step_lookup[event.step_id] = step
            self.step_index = event.step_index
            step.add_activity("llm started")
            return

        if isinstance(event, LLMOutputDeltaEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                step.append_reasoning(event.text_delta)
            return

        if isinstance(event, LLMCompletedEvent):
            step = self._get_step(event.step_id)
            if step is not None and not step.reasoning:
                if event.step.reasoning_log:
                    step.append_reasoning(event.step.reasoning_log)
            self._log_step_activity(event, "llm completed")
            return

        if isinstance(event, ToolStartedEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                view = ToolView(
                    call_id=event.tool_call.call_id,
                    name=event.tool_name,
                )
                step.tools[view.call_id] = view
                step.tool_order.append(view.call_id)
                step.add_activity(
                    f"tool started -> {event.tool_name} ({event.tool_call.call_id})"
                )
            return

        if isinstance(event, ToolOutputEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                view = step.tools.get(event.tool_call.call_id)
                if view is not None:
                    view.buffer.append(event.text_delta)
            return

        if isinstance(event, ToolCompletedEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                view = step.tools.get(event.tool_call.call_id)
                if view is not None:
                    view.status = "completed"
                    view.buffer = [event.tool_result.output]
                    view.error = ""
                step.add_activity(
                    f"tool completed -> {event.tool_name} ({event.tool_call.call_id})"
                )
            return

        if isinstance(event, ToolFailedEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                view = step.tools.get(event.tool_call.call_id)
                if view is None:
                    view = ToolView(
                        call_id=event.tool_call.call_id,
                        name=event.tool_name,
                    )
                    step.tools[view.call_id] = view
                    step.tool_order.append(view.call_id)
                view.status = "failed"
                view.error = event.error or event.tool_result.error or ""
                step.add_activity(
                    f"tool failed -> {event.tool_name} ({event.tool_call.call_id})"
                )
            return

        if isinstance(event, StepCompletedEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                step.status = "completed"
                step.add_activity("step completed")
            self.step_index = event.step_index
            return

        if isinstance(event, StepFailedEvent):
            step = self._get_step(event.step_id)
            if step is not None:
                step.status = "failed"
                step.error = event.error
                step.add_activity("step failed")
            self.step_index = event.step_index
            self.error_message = event.error
            return

        if isinstance(event, RunCompletedEvent):
            self.final_answer = event.final_answer
            self._log_step_activity(event, "run completed")
            return

        if isinstance(event, RunFailedEvent):
            self.error_message = event.error
            self._log_step_activity(event, "run failed")

    def _record_event(self, event: AgentEvent) -> None:
        description = self._describe_event(event)
        if not description:
            return
        self.event_history.append(description)
        if len(self.event_history) > 12:
            self.event_history = self.event_history[-12:]

    def _log_step_activity(self, event: AgentEvent, note: str) -> None:
        if not note:
            return
        step = self._get_step(event.step_id)
        if step is None:
            return
        step.add_activity(note)

    def _describe_event(self, event: AgentEvent) -> str:
        label = f"step {event.step_index + 1}"
        if isinstance(event, LLMStartedEvent):
            return f"{label}: llm started"
        if isinstance(event, LLMCompletedEvent):
            return f"{label}: llm completed"
        if isinstance(event, LLMOutputDeltaEvent):
            return ""
        if isinstance(event, ToolStartedEvent):
            return f"{label}: tool started -> {event.tool_name}"
        if isinstance(event, ToolCompletedEvent):
            return f"{label}: tool completed -> {event.tool_name}"
        if isinstance(event, ToolFailedEvent):
            return f"{label}: tool failed -> {event.tool_name}"
        if isinstance(event, ToolOutputEvent):
            return ""
        if isinstance(event, StepCompletedEvent):
            return f"{label}: step completed"
        if isinstance(event, StepFailedEvent):
            return f"{label}: step failed"
        if isinstance(event, RunCompletedEvent):
            return "run completed"
        if isinstance(event, RunFailedEvent):
            return "run failed"
        return event.event_type

    def render(self) -> Group:
        question_panel = Panel(
            self.question or "[dim]no question provided[/dim]",
            title="Question",
            border_style="cyan",
        )
        if self.steps:
            step_panels = [self._render_step(step) for step in self.steps]
        else:
            step_panels = [
                Panel(
                    "[dim]waiting for the agent to begin reasoning[/dim]",
                    title="Steps",
                    border_style="magenta",
                )
            ]

        status_lines = []
        if self.final_answer:
            status_lines.append("run: completed")
        elif self.error_message:
            status_lines.append(f"error: {self.error_message}")
        elif self.steps:
            status_lines.append("run: in progress")
        else:
            status_lines.append("run: pending")

        if self.event_history:
            status_lines.append("recent events:")
            status_lines.extend(self.event_history[-6:])
        status_panel = Panel(
            "\n".join(status_lines),
            title="Status",
            border_style="green" if self.final_answer else "red" if self.error_message else "blue",
        )

        panels: list[Panel] = [
            question_panel,
            *step_panels,
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

    def _render_step(self, step: StepView) -> Panel:
        reasoning_text = (
            step.reasoning_text() or "[dim]waiting on the model[/dim]"
        )
        body: list[object] = [Text(reasoning_text)]

        if step.activity_log:
            body.append(
                Panel(
                    step.activity_text(),
                    title="Events",
                    border_style="blue",
                )
            )

        if step.tool_order:
            tools_table = Table(
                show_header=True,
                header_style="bold yellow",
                expand=True,
            )
            tools_table.add_column("Tool")
            tools_table.add_column("Status", style="white")
            tools_table.add_column("Details", style="white", overflow="fold")
            for call_id in step.tool_order:
                view = step.tools[call_id]
                tool_label = f"{view.name}\n[dim]{view.call_id}"
                tools_table.add_row(tool_label, view.status_text(), view.output_text())
            body.append(tools_table)

        if step.error and step.status == "failed":
            body.append(Text(step.error, style="red"))

        content = Group(*body)
        title = f"Step {step.index + 1} · {step.status_text()}"
        return Panel(content, title=title, border_style=step.border_style())


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
        auto_refresh=False,
    ) as live:
        async for event in agent.run(question):
            state.apply(event)
            live.update(state.render(), refresh=True)
            if isinstance(event, RunCompletedEvent):
                return event.final_answer
    raise RuntimeError("Agent terminated without RunCompletedEvent")
