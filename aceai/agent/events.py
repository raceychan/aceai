"""Agent event stream data models segregated by concern."""

from typing import ClassVar, Literal

from aceai.interface import Record
from aceai.llm.models import LLMSegment, LLMToolCall

from ..models import AgentStep, ToolExecutionResult

AgentEventType = Literal[
    "agent.llm.started",
    "agent.llm.output_text.delta",
    "agent.llm.media",
    "agent.llm.completed",
    "agent.tool.started",
    "agent.tool.output",
    "agent.tool.completed",
    "agent.tool.failed",
    "agent.step.completed",
    "agent.step.failed",
    "agent.run.completed",
    "agent.run.failed",
]


class AgentLifecycleEvent(Record, kw_only=True):
    """Base class shared by all emitted agent events."""

    EVENT_TYPE: ClassVar[AgentEventType]

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


class LLMMediaEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.media"
    segments: list[LLMSegment]


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
    | LLMMediaEvent
    | LLMCompletedEvent
    | ToolStartedEvent
    | ToolOutputEvent
    | ToolCompletedEvent
    | ToolFailedEvent
    | StepCompletedEvent
    | StepFailedEvent
    | RunCompletedEvent
    | RunFailedEvent
)


class AgentEventBuilder:
    """Utility helper that stamps shared step metadata onto events."""

    __slots__ = ("step_index", "step_id")

    def __init__(self, *, step_index: int, step_id: str):
        self.step_index = step_index
        self.step_id = step_id

    def llm_started(self) -> LLMStartedEvent:
        return LLMStartedEvent(step_index=self.step_index, step_id=self.step_id)

    def llm_text_delta(self, *, text_delta: str) -> LLMOutputDeltaEvent:
        return LLMOutputDeltaEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            text_delta=text_delta,
        )

    def llm_media(self, *, segments: list[LLMSegment]) -> LLMMediaEvent:
        return LLMMediaEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            segments=segments,
        )

    def llm_completed(self, *, step: AgentStep) -> LLMCompletedEvent:
        return LLMCompletedEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
        )

    def tool_started(self, *, tool_call: LLMToolCall) -> ToolStartedEvent:
        return ToolStartedEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
        )

    def tool_output(
        self, *, tool_call: LLMToolCall, text_delta: str
    ) -> ToolOutputEvent:
        return ToolOutputEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
            text_delta=text_delta,
        )

    def tool_completed(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolExecutionResult,
    ) -> ToolCompletedEvent:
        return ToolCompletedEvent(
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
            step_index=self.step_index,
            step_id=self.step_id,
            tool_call=tool_call,
            tool_name=tool_call.name,
            tool_result=tool_result,
            error=error,
        )

    def step_completed(self, *, step: AgentStep) -> StepCompletedEvent:
        return StepCompletedEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
        )

    def step_failed(self, *, step: AgentStep, error: str) -> StepFailedEvent:
        return StepFailedEvent(
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
            error=error,
        )

    def run_completed(
        self,
        *,
        step: AgentStep,
        final_answer: str,
    ) -> RunCompletedEvent:
        return RunCompletedEvent(
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
            step_index=self.step_index,
            step_id=self.step_id,
            step=step,
            error=error,
        )
