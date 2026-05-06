"""Agent event stream data models segregated by concern."""

from typing import ClassVar, Literal

from aceai.llm.interface import Record
from aceai.llm.models import LLMSegment, LLMToolCall, LLMToolCallDelta

from .models import (
    AgentStep,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolExecutionResult,
)

AgentEventType = Literal[
    "agent.llm.started",
    "agent.llm.output_text.delta",
    "agent.llm.tool_call.delta",
    "agent.llm.media",
    "agent.llm.reasoning",
    "agent.llm.retrying",
    "agent.llm.completed",
    "agent.tool.started",
    "agent.tool.output",
    "agent.tool.approval_requested",
    "agent.tool.approval_resolved",
    "agent.tool.completed",
    "agent.tool.failed",
    "agent.step.completed",
    "agent.step.failed",
    "agent.run.suspended",
    "agent.run.completed",
    "agent.run.failed",
]


class AgentLifecycleEvent(Record, kw_only=True):
    """Base class shared by all emitted agent events."""

    EVENT_TYPE: ClassVar[AgentEventType]

    run_id: str = ""
    """Stable identifier for one user-input-to-final-answer agent run."""

    step_index: int
    """0-based index of the reasoning step associated with this event."""

    step_id: str
    """Stable identifier that ties the event to a recorded step."""

    @property
    def event_type(self) -> AgentEventType:
        return self.EVENT_TYPE


class LLMStartedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.started"


class LLMOutputDeltaEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.output_text.delta"
    text_delta: str


class LLMToolCallDeltaEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.tool_call.delta"
    tool_call_delta: LLMToolCallDelta
    text_delta: str


class LLMMediaEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.media"
    segments: list[LLMSegment]


class LLMReasoningEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.reasoning"
    segment: LLMSegment


class LLMRetryingEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.retrying"
    retry_count: int
    retry_max: int
    retry_delay_seconds: float
    error: str


class LLMCompletedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.completed"
    step: AgentStep


class ToolLifecycleEvent(AgentLifecycleEvent):
    tool_call: LLMToolCall
    tool_name: str


class ToolStartedEvent(ToolLifecycleEvent):
    EVENT_TYPE = "agent.tool.started"


class ToolOutputEvent(ToolLifecycleEvent):
    EVENT_TYPE = "agent.tool.output"
    text_delta: str


class ToolApprovalRequestedEvent(ToolLifecycleEvent):
    EVENT_TYPE = "agent.tool.approval_requested"
    request: ToolApprovalRequest


class ToolApprovalResolvedEvent(ToolLifecycleEvent):
    EVENT_TYPE = "agent.tool.approval_resolved"
    request: ToolApprovalRequest
    decision: ToolApprovalDecision


class ToolCompletedEvent(ToolLifecycleEvent):
    EVENT_TYPE = "agent.tool.completed"
    tool_result: ToolExecutionResult


class ToolFailedEvent(ToolLifecycleEvent):
    EVENT_TYPE = "agent.tool.failed"
    tool_result: ToolExecutionResult
    error: str


class StepCompletedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.step.completed"
    step: AgentStep


class StepFailedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.step.failed"
    step: AgentStep
    error: str


class RunSuspendedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.run.suspended"
    request: ToolApprovalRequest


class RunCompletedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.run.completed"
    step: AgentStep
    final_answer: str


class RunFailedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.run.failed"
    step: AgentStep
    error: str


type AgentEvent = (
    LLMStartedEvent
    | LLMOutputDeltaEvent
    | LLMToolCallDeltaEvent
    | LLMMediaEvent
    | LLMReasoningEvent
    | LLMRetryingEvent
    | LLMCompletedEvent
    | ToolStartedEvent
    | ToolOutputEvent
    | ToolApprovalRequestedEvent
    | ToolApprovalResolvedEvent
    | ToolCompletedEvent
    | ToolFailedEvent
    | StepCompletedEvent
    | StepFailedEvent
    | RunSuspendedEvent
    | RunCompletedEvent
    | RunFailedEvent
)


class AgentEventBuilder:
    """Utility helper that stamps shared step metadata onto events."""

    __slots__ = ("run_id", "step_index", "step_id")

    def __init__(self, *, step_index: int, step_id: str, run_id: str = ""):
        self.run_id = run_id
        self.step_index = step_index
        self.step_id = step_id

    def llm_started(self) -> LLMStartedEvent:
        return LLMStartedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
        )

    def llm_text_delta(self, *, text_delta: str) -> LLMOutputDeltaEvent:
        return LLMOutputDeltaEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            text_delta=text_delta,
        )

    def llm_tool_call_delta(
        self, *, tool_call_delta: LLMToolCallDelta
    ) -> LLMToolCallDeltaEvent:
        return LLMToolCallDeltaEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call_delta=tool_call_delta,
            text_delta=tool_call_delta.arguments_delta,
        )

    def llm_media(self, *, segments: list[LLMSegment]) -> LLMMediaEvent:
        return LLMMediaEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            segments=segments,
        )

    def llm_reasoning(self, *, segment: LLMSegment) -> LLMReasoningEvent:
        return LLMReasoningEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            segment=segment,
        )

    def llm_retrying(
        self,
        *,
        retry_count: int,
        retry_max: int,
        retry_delay_seconds: float,
        error: str,
    ) -> LLMRetryingEvent:
        return LLMRetryingEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            retry_count=retry_count,
            retry_max=retry_max,
            retry_delay_seconds=retry_delay_seconds,
            error=error,
        )

    def llm_completed(self, *, step: AgentStep) -> LLMCompletedEvent:
        return LLMCompletedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
        )

    def tool_started(self, *, tool_call: LLMToolCall) -> ToolStartedEvent:
        return ToolStartedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
        )

    def tool_output(
        self, *, tool_call: LLMToolCall, text_delta: str
    ) -> ToolOutputEvent:
        return ToolOutputEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
            text_delta=text_delta,
        )

    def tool_approval_requested(
        self,
        *,
        request: ToolApprovalRequest,
    ) -> ToolApprovalRequestedEvent:
        return ToolApprovalRequestedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=request.call,
            tool_name=request.tool_name,
            request=request,
        )

    def tool_approval_resolved(
        self,
        *,
        request: ToolApprovalRequest,
        decision: ToolApprovalDecision,
    ) -> ToolApprovalResolvedEvent:
        return ToolApprovalResolvedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=request.call,
            tool_name=request.tool_name,
            request=request,
            decision=decision,
        )

    def tool_completed(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolExecutionResult,
    ) -> ToolCompletedEvent:
        return ToolCompletedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
            tool_result=tool_result,
        )

    def tool_failed(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolExecutionResult,
        error: str,
    ) -> ToolFailedEvent:
        return ToolFailedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
            tool_result=tool_result,
            error=error,
        )

    def step_completed(self, *, step: AgentStep) -> StepCompletedEvent:
        return StepCompletedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
        )

    def step_failed(self, *, step: AgentStep, error: str) -> StepFailedEvent:
        return StepFailedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
            error=error,
        )

    def run_suspended(self, *, request: ToolApprovalRequest) -> RunSuspendedEvent:
        return RunSuspendedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            request=request,
        )

    def run_completed(
        self,
        *,
        step: AgentStep,
        final_answer: str,
    ) -> RunCompletedEvent:
        return RunCompletedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
            final_answer=final_answer,
        )

    def run_failed(
        self,
        *,
        step: AgentStep,
        error: str,
    ) -> RunFailedEvent:
        return RunFailedEvent(
            run_id=self.run_id,
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
            error=error,
        )
