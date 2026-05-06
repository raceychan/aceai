from typing import Any

from msgspec import Struct

from aceai.agent.cost import estimate_usage_cost
from aceai.agent.session import (
    EventLog,
    SessionEvent,
    SessionMetadata,
    SessionRecorder,
    SessionState,
    SessionStore,
)
from aceai.core.events import (
    AgentEvent,
    LLMCompletedEvent,
    LLMOutputDeltaEvent,
    LLMReasoningEvent,
    LLMRetryingEvent,
    LLMStartedEvent,
    LLMToolCallDeltaEvent,
    LLMMediaEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunSuspendedEvent,
    StepCompletedEvent,
    StepFailedEvent,
    ToolApprovalRequestedEvent,
    ToolApprovalResolvedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ToolOutputEvent,
    ToolStartedEvent,
)
from aceai.llm.interface import is_set
from aceai.llm.models import LLMMessage, LLMReasoningSegmentMeta, LLMUsage


class AgentSessionSnapshot(Struct, frozen=True, kw_only=True):
    metadata: SessionMetadata
    event_log: EventLog
    history: list[LLMMessage]
    state: SessionState


class SessionService:
    """Agent app session boundary over the low-level store and recorder."""

    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        recorder: SessionRecorder | None = None,
        session_id: str | None = None,
    ) -> None:
        if recorder is not None and session_id is not None:
            if recorder.session_id != session_id:
                raise ValueError("Session recorder and session_id disagree")
        self._store = store or (recorder.store if recorder is not None else SessionStore())
        self._recorder = recorder
        self._session_id = session_id or (recorder.session_id if recorder is not None else None)

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def recorder(self) -> SessionRecorder | None:
        return self._recorder

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def ensure_session(self) -> str:
        if self._recorder is not None and self._session_id is not None:
            return self._session_id
        metadata = self._store.create_session()
        self._recorder = SessionRecorder(self._store, metadata.session_id)
        self._session_id = metadata.session_id
        return metadata.session_id

    def attach_session(self, session_id: str) -> AgentSessionSnapshot:
        metadata = self._store.get_session(session_id)
        if self._recorder is not None:
            self._recorder.finalize()
        self._recorder = SessionRecorder(self._store, metadata.session_id)
        self._session_id = metadata.session_id
        return self.snapshot(metadata.session_id)

    def snapshot(self, session_id: str) -> AgentSessionSnapshot:
        metadata = self._store.get_session(session_id)
        event_log = self._store.load_event_log(session_id)
        return AgentSessionSnapshot(
            metadata=metadata,
            event_log=event_log,
            history=event_log.replay_llm_history(),
            state=self._store.get_session_state(session_id),
        )

    def list_sessions(self) -> list[SessionMetadata]:
        return self._store.list_sessions()

    def total_cost_usd(self) -> float:
        return self._store.total_cost_usd()

    def get_state(self, session_id: str) -> SessionState:
        return self._store.get_session_state(session_id)

    def update_state(self, session_id: str, state: SessionState) -> None:
        self._store.update_session_state(session_id, state)

    def record_user_message(self, content: str, *, run_id: str) -> None:
        self.record_session_event(
            SessionEvent(
                run_id=run_id,
                step_id=None,
                step_index=None,
                kind="user_message",
                payload={"content": content},
            )
        )

    def record_agent_event(self, event: AgentEvent) -> None:
        self.record_session_event(agent_event_to_session_event(event))

    def record_session_event(self, event: SessionEvent) -> None:
        if self._recorder is None:
            raise RuntimeError("AceAI session is not active")
        self._recorder.record(event)

    def finalize(self) -> bool:
        if self._recorder is None:
            return False
        return self._recorder.finalize()


def agent_event_to_session_event(event: AgentEvent) -> SessionEvent:
    return SessionEvent(
        run_id=event.run_id,
        step_id=event.step_id,
        step_index=event.step_index,
        kind=_kind_for_agent_event(event),
        payload=_payload_for_agent_event(event),
    )


def _kind_for_agent_event(event: AgentEvent):
    if isinstance(event, LLMStartedEvent):
        return "step_started"
    if isinstance(event, LLMOutputDeltaEvent):
        return "assistant_delta"
    if isinstance(event, LLMReasoningEvent):
        if (
            isinstance(event.segment.meta, LLMReasoningSegmentMeta)
            and event.segment.meta.is_delta
        ):
            return "thinking_delta"
        return "reasoning_summary"
    if isinstance(event, LLMRetryingEvent):
        return "llm_retrying"
    if isinstance(event, LLMToolCallDeltaEvent):
        return "tool_call_delta"
    if isinstance(event, LLMMediaEvent):
        return "media"
    if isinstance(event, LLMCompletedEvent):
        return "llm_completed"
    if isinstance(event, ToolStartedEvent):
        return "tool_started"
    if isinstance(event, ToolOutputEvent):
        return "tool_output"
    if isinstance(event, ToolApprovalRequestedEvent):
        return "tool_approval_requested"
    if isinstance(event, ToolApprovalResolvedEvent):
        return "tool_approval_resolved"
    if isinstance(event, ToolCompletedEvent):
        return "tool_completed"
    if isinstance(event, ToolFailedEvent):
        return "tool_failed"
    if isinstance(event, StepCompletedEvent):
        return "step_completed"
    if isinstance(event, StepFailedEvent):
        return "step_failed"
    if isinstance(event, RunSuspendedEvent):
        return "run_suspended"
    if isinstance(event, RunCompletedEvent):
        return "run_completed"
    if isinstance(event, RunFailedEvent):
        return "run_failed"
    raise ValueError("Unsupported agent event")


def _payload_for_agent_event(event: AgentEvent) -> dict[str, Any]:
    if isinstance(event, LLMOutputDeltaEvent):
        return {"content": event.text_delta}
    if isinstance(event, LLMReasoningEvent):
        return {"content": event.segment.content}
    if isinstance(event, LLMRetryingEvent):
        return {
            "content": _retrying_content(event),
            "error": event.error,
            "retry_count": event.retry_count,
            "retry_max": event.retry_max,
            "retry_delay_seconds": event.retry_delay_seconds,
        }
    if isinstance(event, LLMToolCallDeltaEvent):
        return {
            "content": event.text_delta,
            "tool_call_id": event.tool_call_delta.id,
        }
    if isinstance(event, LLMCompletedEvent):
        response = event.step.llm_response
        usage: LLMUsage | None = None
        if is_set(response.usage):
            usage = response.usage
        provider_name = None
        if response.provider_meta:
            provider_name = response.provider_meta[0].provider_name
        cost = estimate_usage_cost(response.model, usage, provider_name=provider_name)
        payload: dict[str, Any] = {
            "content": response.text,
            "tool_calls": [call.asdict() for call in response.tool_calls],
        }
        if usage is not None:
            payload["usage"] = {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        if cost is not None:
            payload["cost"] = cost.asdict()
        return payload
    if isinstance(event, ToolStartedEvent | ToolOutputEvent):
        payload = _tool_payload(event)
        if isinstance(event, ToolOutputEvent):
            payload["content"] = event.text_delta
        return payload
    if isinstance(event, ToolApprovalRequestedEvent):
        content = event.request.reason
        if event.request.policy != "":
            content = f"{content} ({event.request.policy})"
        return {**_tool_payload(event), "content": content}
    if isinstance(event, ToolApprovalResolvedEvent):
        decision_text = "approved" if event.decision.approved else "rejected"
        content = decision_text
        if event.decision.reason != "":
            content = f"{decision_text}: {event.decision.reason}"
        return {**_tool_payload(event), "content": content}
    if isinstance(event, ToolCompletedEvent):
        return {
            **_tool_payload(event),
            "content": event.tool_result.output,
            "tool_result": {
                "output": event.tool_result.output,
                "error": event.tool_result.error,
            },
        }
    if isinstance(event, ToolFailedEvent):
        return {
            **_tool_payload(event),
            "content": event.error,
            "error": event.error,
            "tool_result": {
                "output": event.tool_result.output,
                "error": event.tool_result.error,
            },
        }
    if isinstance(event, StepFailedEvent | RunFailedEvent):
        return {"content": event.error, "error": event.error}
    if isinstance(event, RunCompletedEvent):
        return {"content": event.final_answer}
    if isinstance(event, RunSuspendedEvent):
        return {"content": event.request.reason}
    return {"content": ""}


def _tool_payload(event) -> dict[str, Any]:
    return {
        "content": "",
        "tool_name": event.tool_name,
        "tool_call_id": event.tool_call.call_id,
        "tool_call": event.tool_call.asdict(),
    }


def _retrying_content(event: LLMRetryingEvent) -> str:
    return (
        f"Retrying LLM request {event.retry_count}/{event.retry_max} "
        f"in {event.retry_delay_seconds:.1f}s after {event.error}"
    )
