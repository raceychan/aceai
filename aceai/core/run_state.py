from typing import Any, Literal

from msgspec import Struct, field

from aceai.llm.models import LLMToolCall

from .models import AgentStep, ToolApprovalRequest
from .tools import Tool

AgentRunStatus = Literal["running", "suspended", "completed", "failed"]


class ToolInvocation(Struct, kw_only=True):
    """Resolved local tool call ready for policy checks and execution."""

    call: LLMToolCall
    tool: Tool[Any, Any]

    @property
    def approval_required(self) -> bool:
        return self.tool.metadata.require_approval


class ToolRunState(Struct, kw_only=True):
    """Tool execution bookkeeping for a single agent run."""

    call_counts: dict[str, int] = field(default_factory=dict[str, int])
    approved_tool_names: set[str] = field(default_factory=set[str])


class PendingToolApproval(Struct, kw_only=True):
    """Suspended continuation waiting for a caller approval decision."""

    step: AgentStep
    invocation: ToolInvocation
    request: ToolApprovalRequest
    run_id: str
    step_index: int
    step_id: str
    tool_index: int


class AgentRunState(Struct, kw_only=True):
    """Mutable state for one Agent run."""

    status: AgentRunStatus = "running"
    tools: ToolRunState = field(default_factory=ToolRunState)
    pending_approval: PendingToolApproval | None = None

    def suspend_for_approval(
        self,
        *,
        step: AgentStep,
        invocation: ToolInvocation,
        request: ToolApprovalRequest,
        run_id: str,
        step_index: int,
        step_id: str,
        tool_index: int,
    ) -> None:
        self.status = "suspended"
        self.pending_approval = PendingToolApproval(
            step=step,
            invocation=invocation,
            request=request,
            run_id=run_id,
            step_index=step_index,
            step_id=step_id,
            tool_index=tool_index,
        )

    def resume_from_approval(self) -> PendingToolApproval:
        pending = self.pending_approval
        if pending is None:
            raise ValueError("agent run is not suspended for tool approval")
        self.pending_approval = None
        self.status = "running"
        return pending
