from rich.console import Console

from aceai import AgentBase
from aceai.agent.events import (
    AgentEvent,
    LLMCompletedEvent,
    LLMOutputDeltaEvent,
    RunCompletedEvent,
    RunFailedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
)
from opentelemetry.context import Context


class EventStreamPrinter:
    """Print only reasoning chunks and tool call results."""

    def __init__(self, question: str, console: Console):
        self.console = console
        self.question = question
        self.started = False
        self.current_step = -1
        self.step_has_output: dict[int, bool] = {}

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        header = self.question if self.question else "[no question]"
        self.console.print(f"[bold cyan]question[/bold cyan]: {header}")

    def _step_header(self, step_index: int) -> None:
        if step_index == self.current_step:
            return
        self.current_step = step_index
        self.console.print(f"[bold magenta]reasoning step {step_index + 1}[/bold magenta]")

    def print_reasoning_delta(self, event: LLMOutputDeltaEvent) -> None:
        self._step_header(event.step_index)
        if not event.text_delta:
            return
        self.console.print(event.text_delta, end="")
        self.step_has_output[event.step_index] = True

    def finalize_reasoning(self, event: LLMCompletedEvent) -> None:
        self._step_header(event.step_index)
        if not self.step_has_output.get(event.step_index):
            text = event.step.reasoning_log
            if text:
                self.console.print(text, end="")
                self.step_has_output[event.step_index] = True
        if self.step_has_output.get(event.step_index):
            self.console.print("")

    def print_tool_result(self, event: ToolCompletedEvent) -> None:
        output = event.tool_result.output
        label = f"[tool {event.tool_name}]"
        self.console.print(f"{label} {output}")

    def print_tool_failure(self, event: ToolFailedEvent) -> None:
        label = f"[tool {event.tool_name} failed]"
        error_message = event.error or event.tool_result.error
        self.console.print(f"{label} {error_message}")

    def print_final_answer(self, answer: str) -> None:
        self.console.print(f"[bold green]final answer[/bold green]: {answer}")

    def print_run_failure(self, error: str) -> None:
        self.console.print(f"[bold red]run failed[/bold red]: {error}")

    def handle_event(self, event: AgentEvent) -> None:
        if not self.started:
            self.start()
        if isinstance(event, LLMOutputDeltaEvent):
            self.print_reasoning_delta(event)
            return
        if isinstance(event, LLMCompletedEvent):
            self.finalize_reasoning(event)
            return
        if isinstance(event, ToolCompletedEvent):
            self.print_tool_result(event)
            return
        if isinstance(event, ToolFailedEvent):
            self.print_tool_failure(event)
            return


async def run_agent_with_terminal_ui(
    agent: AgentBase, question: str, *, trace_ctx: Context | None = None
) -> str:
    console = Console()
    printer = EventStreamPrinter(question, console)
    printer.start()
    agen = agent.run(question, trace_ctx=trace_ctx)
    try:
        async for event in agen:
            printer.handle_event(event)
            if isinstance(event, RunCompletedEvent):
                printer.print_final_answer(event.final_answer)
                return event.final_answer
            if isinstance(event, RunFailedEvent):
                error_message = event.error or "agent run failed"
                printer.print_run_failure(error_message)
                raise RuntimeError(error_message)
        raise RuntimeError("agent terminated without RunCompletedEvent")
    finally:
        await agen.aclose()
