import asyncio
import json
from typing import AsyncGenerator
from urllib.error import URLError
from urllib.request import urlopen

import aceai
from msgspec import Struct
from opentelemetry.context import Context

from aceai.agent.citations import TurnCitation, message_with_citations
from aceai.agent.memory.ideas import Idea, IdeaStore
from aceai.agent.project import ProjectMetadata, default_project
from aceai.agent.session import SessionRecorder, SessionState, SessionStore
from aceai.agent.session import EventLog
from aceai.agent.session_service import AgentSessionSnapshot, SessionService
from aceai.agent.memory.subagent_artifacts import SubagentArtifactStore
from aceai.core import Agent, AgentRunContext, ToolApprovalDecision
from aceai.core.events import (
    AgentEvent,
    ContextCompactionFailedEvent,
    ContextCompactionStartedEvent,
    ContextCompressedEvent,
    LLMCompletedEvent,
    RunCompletedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
)
from aceai.core.models import ToolApprovalRequest
from aceai.llm.models import LLMMessage, LLMRequestMeta

PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/aceai/json"


class UpdateCheckResult(Struct, frozen=True, kw_only=True):
    current_version: str
    latest_version: str

    @property
    def has_update(self) -> bool:
        return _version_parts(self.latest_version) > _version_parts(
            self.current_version
        )


class AceAgentApp:
    """Reusable agent app runtime used by UI and future ports."""

    def __init__(
        self,
        agent: Agent,
        *,
        provider_name: str,
        selected_model: str,
        initial_history: list[LLMMessage] | None = None,
        session_service: SessionService | None = None,
        session_store: SessionStore | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        idea_store: IdeaStore | None = None,
        project: ProjectMetadata | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._agent = agent
        self._provider_name = provider_name
        self._selected_model = selected_model
        self._project = project or (
            session_store.project
            if session_store is not None
            else default_project()
        )
        self._request_meta = _copy_request_meta(request_meta)
        self._request_meta["model"] = selected_model
        self._session_service = session_service or SessionService(
            store=session_store,
            recorder=session_recorder,
            session_id=session_id,
            project=self._project,
        )
        self._subagent_artifacts = SubagentArtifactStore(self._session_service.store.root)
        self._trace_ctx = trace_ctx
        snapshot = None
        if initial_history is None and self._session_service.session_id is not None:
            snapshot = self._session_service.snapshot(self._session_service.session_id)
            self._llm_history = list(snapshot.history)
        else:
            self._llm_history = list(initial_history or [])
        self._active_run: AgentRunContext | None = None
        self._last_context_event_id: str | None = None
        self._queued_questions: list[str] = []
        self._idea_store = idea_store or IdeaStore()
        self._approved_tool_names: set[str] = set()
        self._update_check_completed = False
        self._update_check_lock = asyncio.Lock()
        session_id = self._session_service.session_id
        if session_id is not None:
            if snapshot is None:
                snapshot = self._session_service.snapshot(session_id)
            self._approved_tool_names.update(
                _approved_tool_names_from_event_log(snapshot.event_log)
            )
            self._last_context_event_id = _last_context_source_event_id(snapshot.event_log)

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def selected_model(self) -> str:
        return self._selected_model

    @property
    def session_service(self) -> SessionService:
        return self._session_service

    @property
    def project(self) -> ProjectMetadata:
        return self._project

    @property
    def project_name(self) -> str:
        return self._project.name

    @property
    def session_id(self) -> str | None:
        return self._session_service.session_id

    @property
    def session_recorder(self) -> SessionRecorder | None:
        return self._session_service.recorder

    @property
    def llm_history(self) -> list[LLMMessage]:
        return list(self._llm_history)

    @property
    def active_run(self) -> AgentRunContext | None:
        return self._active_run

    @property
    def queued_questions(self) -> tuple[str, ...]:
        return tuple(self._queued_questions)

    @property
    def is_running_suspended(self) -> bool:
        run = self._active_run
        return run is not None and run.status == "suspended"

    def ensure_session(self) -> str:
        return self._session_service.ensure_session()

    def switch_session(self, session_id: str) -> AgentSessionSnapshot:
        snapshot = self._session_service.attach_session(session_id)
        self._llm_history = list(snapshot.history)
        self._active_run = None
        self._queued_questions = []
        self._approved_tool_names = _approved_tool_names_from_event_log(
            snapshot.event_log
        )
        self._last_context_event_id = _last_context_source_event_id(snapshot.event_log)
        return snapshot

    def restore_history_from_active_session(self) -> None:
        session_id = self.session_id
        if session_id is None:
            self._llm_history = []
            self._active_run = None
            self._queued_questions = []
            self._last_context_event_id = None
            return
        snapshot = self._session_service.snapshot(session_id)
        self._llm_history = snapshot.history
        self._last_context_event_id = _last_context_source_event_id(snapshot.event_log)
        self._active_run = None

    def cancel_active_turn(self) -> None:
        self._active_run = None

    def enqueue_turn(self, question: str) -> int:
        self._queued_questions.append(question)
        return len(self._queued_questions)

    def pop_queued_turn(self) -> str | None:
        if not self._queued_questions:
            return None
        return self._queued_questions.pop(0)

    def take_queued_turn(self, index: int) -> str:
        return self._queued_questions.pop(index)

    def switch_model(self, model: str) -> None:
        self._selected_model = model
        self._request_meta["model"] = model
        self.persist_session_state()

    def persist_session_state(self) -> None:
        session_id = self.session_id
        if session_id is None:
            return
        self._session_service.update_state(
            session_id,
            SessionState(
                selected_provider=self._provider_name,
                selected_model=self._selected_model,
            ),
        )

    def capture_idea(self, content: str) -> Idea:
        return self._idea_store.capture(
            content,
            project=self._project,
            source_session_id=self.session_id,
        )

    def list_ideas(self) -> list[Idea]:
        return self._idea_store.list_for_display(current_project=self._project)

    def delete_idea(self, index: int) -> Idea:
        return self._idea_store.delete_displayed(
            index,
            current_project=self._project,
        )

    def update_idea(self, index: int, content: str) -> Idea:
        return self._idea_store.update_displayed(
            index,
            content,
            current_project=self._project,
        )

    def pending_approval_request(self) -> ToolApprovalRequest | None:
        run = self._active_run
        if run is None:
            return None
        pending = run.run_state.pending_approval
        if pending is None:
            return None
        return pending.request

    async def check_for_updates(self) -> UpdateCheckResult | None:
        async with self._update_check_lock:
            if self._update_check_completed:
                return None
            self._update_check_completed = True
            result = await check_for_updates()
            if result is None or not result.has_update:
                return None
            return result

    async def start_turn(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> AsyncGenerator[AgentEvent, None]:
        self.ensure_session()
        self.persist_session_state()
        llm_question = message_with_citations(question, citations)
        self._active_run = self._agent.create_resume_run(
            llm_question,
            self._llm_history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        self._active_run.run_state.tools.approved_tool_names = (
            self._approved_tool_names
        )
        self._last_context_event_id = self._session_service.record_user_message(
            question,
            citations=citations,
            run_id=self._active_run.run_id,
        )
        async for event in self._consume_run_stream(
            self._active_run,
            self._agent.execute(self._active_run),
        ):
            yield event

    async def approve_tool(self) -> AsyncGenerator[AgentEvent, None]:
        request = self.pending_approval_request()
        if request is None:
            raise RuntimeError("No pending tool approval")
        async for event in self._resume_approval(ToolApprovalDecision.approve(request)):
            yield event

    async def reject_tool(self, reason: str) -> AsyncGenerator[AgentEvent, None]:
        request = self.pending_approval_request()
        if request is None:
            raise RuntimeError("No pending tool approval")
        if reason == "":
            reason = "rejected by caller"
        async for event in self._resume_approval(
            ToolApprovalDecision.reject(request, reason=reason)
        ):
            yield event

    async def _resume_approval(
        self,
        decision: ToolApprovalDecision,
    ) -> AsyncGenerator[AgentEvent, None]:
        run = self._current_run()
        async for event in self._consume_run_stream(
            run,
            self._agent.resume_approval(run, decision),
        ):
            yield event

    async def _consume_run_stream(
        self,
        run: AgentRunContext,
        stream: AsyncGenerator[AgentEvent, None],
    ) -> AsyncGenerator[AgentEvent, None]:
        try:
            async for event in stream:
                if isinstance(event, ContextCompressedEvent):
                    self._record_context_checkpoint(event)
                event = self._archive_subagent_artifact(event)
                persisted_event_id: str | None = None
                if not isinstance(
                    event,
                    ContextCompactionFailedEvent
                    | ContextCompactionStartedEvent
                    | ContextCompressedEvent,
                ):
                    persisted_event_id = self._session_service.record_agent_event(event)
                if persisted_event_id is not None and _agent_event_updates_context(
                    event
                ):
                    self._last_context_event_id = persisted_event_id
                if isinstance(event, RunCompletedEvent):
                    self._finish_run_turn(run, event.final_answer)
                yield event
        finally:
            await stream.aclose()

    def _current_run(self) -> AgentRunContext:
        run = self._active_run
        if run is None:
            raise RuntimeError("AceAI run is not active")
        return run

    def _finish_run_turn(self, run: AgentRunContext, answer: str) -> None:
        self._llm_history = list(run.context.context[1:])
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

    def _record_context_checkpoint(self, event: ContextCompressedEvent) -> None:
        included_event_id = self._last_context_event_id
        if included_event_id is None:
            raise RuntimeError("Context checkpoint has no included transcript event")
        self._session_service.record_context_checkpoint(
            event,
            included_event_id=included_event_id,
        )

    def _archive_subagent_artifact(self, event: AgentEvent) -> AgentEvent:
        if not isinstance(event, ToolCompletedEvent):
            return event
        if event.tool_name != "delegate_to_subagent":
            return event
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("subagent artifact archive requires an active session")
        archived_result = self._subagent_artifacts.archive_tool_result(
            session_id=session_id,
            parent_run_id=event.run_id,
            tool_result=event.tool_result,
        )
        return ToolCompletedEvent(
            run_id=event.run_id,
            step_index=event.step_index,
            step_id=event.step_id,
            tool_call=event.tool_call,
            tool_name=event.tool_name,
            tool_result=archived_result,
        )

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta = _copy_request_meta(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta


def _copy_request_meta(request_meta: LLMRequestMeta | None) -> LLMRequestMeta:
    if request_meta is None:
        return {}
    return {
        **request_meta,
    }


def _agent_event_updates_context(event: AgentEvent) -> bool:
    return isinstance(event, LLMCompletedEvent | ToolCompletedEvent | ToolFailedEvent)


def _last_context_source_event_id(event_log: EventLog) -> str | None:
    for event in reversed(event_log.events):
        if event.kind in (
            "assistant_message",
            "assistant_tool_call",
            "tool_result",
            "user_message",
        ):
            return event.event_id
    return None


async def check_for_updates() -> UpdateCheckResult | None:
    current_version = aceai.__version__
    latest_version = await _fetch_latest_package_version()
    if latest_version is None:
        return None
    return UpdateCheckResult(
        current_version=current_version,
        latest_version=latest_version,
    )


async def _fetch_latest_package_version() -> str | None:
    return await asyncio.to_thread(_fetch_latest_package_version_sync)


def _fetch_latest_package_version_sync() -> str | None:
    try:
        with urlopen(PYPI_PROJECT_JSON_URL, timeout=2.0) as response:
            payload = json.loads(response.read().decode())
    except (URLError, TimeoutError):
        return None
    if type(payload) is not dict:
        raise TypeError("PyPI project payload must be a mapping")
    info = payload["info"]
    if type(info) is not dict:
        raise TypeError("PyPI project info must be a mapping")
    version = info["version"]
    if type(version) is not str:
        raise TypeError("PyPI project version must be str")
    return version


def _version_parts(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError("AceAI version must use x.y.z format")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _approved_tool_names_from_event_log(event_log: EventLog) -> set[str]:
    approved_tool_names: set[str] = set()
    for event in event_log.events:
        if event.kind != "tool_approval_resolved":
            continue
        if event.payload["content"] != "approved":
            continue
        tool_name = event.payload["tool_name"]
        if type(tool_name) is not str:
            raise TypeError("tool approval payload tool_name must be str")
        approved_tool_names.add(tool_name)
    return approved_tool_names
