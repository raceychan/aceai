from datetime import datetime
from typing import Literal

from msgspec import field

from aceai.core.helpers.string import uuid_str
from aceai.core.helpers.time import utc_now
from aceai.llm.interface import Record, StrDict
from aceai.llm.models import LLMCitationRef, LLMResponse, LLMToolCall


class AgentSafetyNote(Record):
    """Lightweight safety verdict surfaced to agent consumers."""

    category: str
    """Provider-declared policy category (e.g., violence)."""

    verdict: Literal["allow", "review", "block"] = "allow"
    """Outcome category aligned with upstream provider policies."""

    metadata: StrDict = field(default_factory=StrDict)
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

    annotations: StrDict = field(default_factory=StrDict)
    """Optional executor-provided metadata (latency, logs, etc.)."""


class ToolApprovalRequest(Record, kw_only=True):
    """Request for caller approval before executing a resolved tool call."""

    call: LLMToolCall
    """Tool call proposed by the model."""

    tool_name: str
    """Resolved local tool name."""

    reason: str = ""
    """Optional policy reason shown to the caller."""

    policy: str = ""
    """Policy that requires this approval."""


class ToolApprovalDecision(Record, kw_only=True):
    """Caller decision for a pending tool approval request."""

    call_id: str
    """Tool call id this decision resolves."""

    approved: bool
    """Whether the tool call may execute."""

    reason: str = ""
    """Optional caller reason, especially useful for rejection."""

    @classmethod
    def approve(cls, request: ToolApprovalRequest) -> "ToolApprovalDecision":
        return cls(call_id=request.call.call_id, approved=True)

    @classmethod
    def reject(
        cls,
        request: ToolApprovalRequest,
        *,
        reason: str,
    ) -> "ToolApprovalDecision":
        return cls(call_id=request.call.call_id, approved=False, reason=reason)


class AgentStepAnnotations(Record, kw_only=True):
    """Auxiliary annotations that enrich each agent reasoning step."""

    safety: list[AgentSafetyNote] = field(default_factory=list[AgentSafetyNote])
    """Safety verdicts captured for this step."""

    citations: list[AgentCitationRef] = field(default_factory=list[AgentCitationRef])
    """Citation references linked to this step."""

    extra: StrDict = field(default_factory=StrDict)
    """Flexible namespace for provider or agent-specific metadata."""


class AgentStep(Record, kw_only=True):
    """Single agent reasoning turn, including LLM + tool outcomes. Each LLM Response would trigger at most one step"""

    step_id: str = field(default_factory=uuid_str)
    """Stable identifier for this reasoning turn."""

    timestamp: datetime = field(default_factory=utc_now)
    """Wall-clock timestamp captured when the step was recorded."""

    llm_response: LLMResponse
    """Structured LLM payload that triggered this step."""

    tool_results: list[ToolExecutionResult] = field(
        default_factory=list[ToolExecutionResult]
    )
    """Ordered tool invocation outputs associated with this step."""

    reasoning_log: str = ""
    """Streaming log of incremental LLM output deltas for this step."""

    reasoning_log_truncated: bool = False
    """Marks whether the reasoning log was truncated to enforce buffer limits."""

    annotations: AgentStepAnnotations = field(default_factory=AgentStepAnnotations)
    """Optional metadata bundle for safety/citation data."""


# class AgentResponse(Record, kw_only=True):
#     """Complete agent run outcome with per-turn traceability."""

#     turns: list[AgentStep]
#     """All recorded reasoning turns executed by the agent."""

#     final_output: str
#     """User-facing answer synthesized by the agent."""
