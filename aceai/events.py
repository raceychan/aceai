"""Agent event stream data models."""

from typing import Any, Literal

from msgspec import field

from aceai.interface import Record
from aceai.llm.models import LLMToolCall

from .models import AgentStep, ToolExecutionResult


AgentStepEventType = Literal[
    "agent.llm.started",
    "agent.llm.output_text.delta",
    "agent.llm.completed",
    "agent.tool.started",
    "agent.tool.output",
    "agent.tool.completed",
    "agent.tool.failed",
    "agent.step.completed",
    "agent.step.failed",
    "agent.run.completed",
]


class AgentStepEvent(Record, kw_only=True):
    """Normalized streaming event emitted while the agent runs."""

    event_type: AgentStepEventType
    """Explicit event identifier describing the agent lifecycle stage."""

    step_index: int
    """0-based index of the reasoning step associated with this event."""

    step_id: str
    """Stable identifier that ties the event to a recorded step."""

    step: AgentStep | None = None
    """Present when an event includes the full step snapshot."""

    llm_delta: str | None = None
    """Partial text emitted by the provider during streaming."""

    tool_call: LLMToolCall | None = None
    """Tool invocation metadata announced by the provider."""

    tool_result: ToolExecutionResult | None = None
    """Completed tool execution result attached to this event."""

    error: str | None = None
    """Optional error payload for failed events."""

    annotations: dict[str, Any] = field(default_factory=dict)
    """Flexible metadata bag surfaced alongside the event."""

    @property
    def is_run_completed(self) -> bool:
        """Returns True if this event signals the end of the agent run."""
        return self.event_type == "agent.run.completed"
