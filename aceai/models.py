from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from msgspec import field

from aceai.interface import Record
from aceai.llm.models import LLMCitationRef, LLMResponse, LLMToolCall


class AgentSafetyNote(Record):
    """Lightweight safety verdict surfaced to agent consumers."""

    category: str
    """Provider-declared policy category (e.g., violence)."""

    verdict: Literal["allow", "review", "block"] = "allow"
    """Outcome category aligned with upstream provider policies."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional structured metadata for downstream renders."""


class AgentCitationRef(Record):
    """Citation metadata attached to an agent-facing segment."""

    label: str
    """Human-friendly label shown to end-users."""

    url: str | None = None
    """Optional resolved URL for the cited resource."""

    provider_ref: LLMCitationRef | None = None
    """Original citation payload for traceability."""


class ToolExecutionResult(Record, kw_only=True):
    """Structured record describing the outcome of a tool invocation."""

    call: LLMToolCall
    """Tool call metadata provided by the LLM response."""

    output: str = ""
    """Serialized tool output returned to the model or end-user."""

    error: str | None = None
    """Optional error message when execution fails."""

    annotations: dict[str, Any] = field(default_factory=dict)
    """Optional executor-provided metadata (latency, logs, etc.)."""


class AgentStepAnnotations(Record, kw_only=True):
    """Auxiliary annotations that enrich each agent reasoning step."""

    safety: list[AgentSafetyNote] = field(default_factory=list[AgentSafetyNote])
    """Safety verdicts captured for this step."""

    citations: list[AgentCitationRef] = field(default_factory=list[AgentCitationRef])
    """Citation references linked to this step."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Flexible namespace for provider or agent-specific metadata."""


def _default_step_id() -> str:
    return str(uuid4())


def _default_timestamp() -> datetime:
    return datetime.now(timezone.utc)


class AgentStep(Record, kw_only=True):
    """Single agent reasoning turn, including LLM + tool outcomes."""

    step_id: str = field(default_factory=_default_step_id)
    """Stable identifier for this reasoning turn."""

    timestamp: datetime = field(default_factory=_default_timestamp)
    """Wall-clock timestamp captured when the step was recorded."""

    llm_response: LLMResponse
    """Structured LLM payload that triggered this step."""

    tool_results: list[ToolExecutionResult] = field(
        default_factory=list[ToolExecutionResult]
    )
    """Ordered tool invocation outputs associated with this step."""

    annotations: AgentStepAnnotations = field(default_factory=AgentStepAnnotations)
    """Optional metadata bundle for safety/citation data."""


class AgentResponse(Record, kw_only=True):
    """Complete agent run outcome with per-turn traceability."""

    turns: list[AgentStep]
    """All recorded reasoning turns executed by the agent."""

    final_output: str
    """User-facing answer synthesized by the agent."""


class AgentStepEvent(Record, kw_only=True):
    """Normalized streaming event emitted while the agent runs."""

    event_type: Literal[
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
