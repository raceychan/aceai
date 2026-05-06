import asyncio
import json
from typing import AsyncGenerator
from urllib.error import URLError
from urllib.request import urlopen

import aceai
from msgspec import Struct
from opentelemetry.context import Context

from aceai.agent.ideas import Idea, IdeaStore
from aceai.agent.session import SessionRecorder, SessionState, SessionStore
from aceai.agent.session import EventLog
from aceai.agent.session_service import AgentSessionSnapshot, SessionService
from aceai.core import AgentBase, AgentRuntime, ToolApprovalDecision
from aceai.core.events import AgentEvent, RunCompletedEvent
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
        agent: AgentBase,
        *,
        provider_name: str,
        selected_model: str,
        initial_history: list[LLMMessage] | None = None,
        session_service: SessionService | None = None,
        session_store: SessionStore | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        idea_store: IdeaStore | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._agent = agent
        self._provider_name = provider_name
        self._selected_model = selected_model
        self._request_meta = _copy_request_meta(request_meta)
        self._request_meta["model"] = selected_model
        self._session_service = session_service or SessionService(
            store=session_store,
            recorder=session_recorder,
            session_id=session_id,
        )
        self._trace_ctx = trace_ctx
        self._llm_history = list(initial_history or [])
        self._active_runtime: AgentRuntime | None = None
        self._queued_questions: list[str] = []
        self._idea_store = idea_store or IdeaStore()
        self._approved_tool_names: set[str] = set()
        self._update_check_completed = False
        self._update_check_lock = asyncio.Lock()
        session_id = self._session_service.session_id
        if session_id is not None:
            self._approved_tool_names.update(
                _approved_tool_names_from_event_log(
                    self._session_service.snapshot(session_id).event_log
                )
            )

    @property
    def agent(self) -> AgentBase:
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
    def session_id(self) -> str | None:
        return self._session_service.session_id

    @property
    def session_recorder(self) -> SessionRecorder | None:
        return self._session_service.recorder

    @property
    def llm_history(self) -> list[LLMMessage]:
        return list(self._llm_history)

    @property
    def active_runtime(self) -> AgentRuntime | None:
        return self._active_runtime

    @property
    def queued_questions(self) -> tuple[str, ...]:
        return tuple(self._queued_questions)

    @property
    def is_running_suspended(self) -> bool:
        runtime = self._active_runtime
        return runtime is not None and runtime.status == "suspended"

    def ensure_session(self) -> str:
        return self._session_service.ensure_session()

    def switch_session(self, session_id: str) -> AgentSessionSnapshot:
        snapshot = self._session_service.attach_session(session_id)
        self._llm_history = list(snapshot.history)
        self._active_runtime = None
        self._queued_questions = []
        self._approved_tool_names = _approved_tool_names_from_event_log(
            snapshot.event_log
        )
        return snapshot

    def restore_history_from_active_session(self) -> None:
        session_id = self.session_id
        if session_id is None:
            self._llm_history = []
            self._active_runtime = None
            self._queued_questions = []
            return
        self._llm_history = self._session_service.snapshot(session_id).history
        self._active_runtime = None

    def cancel_active_turn(self) -> None:
        self._active_runtime = None

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
            source_session_id=self.session_id,
        )

    def list_ideas(self) -> list[Idea]:
        return self._idea_store.list_recent()

    def delete_idea(self, index: int) -> Idea:
        return self._idea_store.delete_recent(index)

    def pending_approval_request(self) -> ToolApprovalRequest | None:
        runtime = self._active_runtime
        if runtime is None:
            return None
        pending = runtime.run_state.pending_approval
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

    async def start_turn(self, question: str) -> AsyncGenerator[AgentEvent, None]:
        self.ensure_session()
        self.persist_session_state()
        self._active_runtime = self._agent.create_resume_run(
            question,
            self._llm_history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        self._active_runtime.run_state.tools.approved_tool_names = (
            self._approved_tool_names
        )
        self._session_service.record_user_message(
            question,
            run_id=self._active_runtime.run_id,
        )
        async for event in self._consume_runtime_stream(
            self._active_runtime,
            self._active_runtime.execute(),
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
        runtime = self._current_runtime()
        async for event in self._consume_runtime_stream(
            runtime,
            runtime.resume_approval(decision),
        ):
            yield event

    async def _consume_runtime_stream(
        self,
        runtime: AgentRuntime,
        stream: AsyncGenerator[AgentEvent, None],
    ) -> AsyncGenerator[AgentEvent, None]:
        try:
            async for event in stream:
                self._session_service.record_agent_event(event)
                if isinstance(event, RunCompletedEvent):
                    self._finish_runtime_turn(runtime, event.final_answer)
                yield event
        finally:
            await stream.aclose()

    def _current_runtime(self) -> AgentRuntime:
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("AceAI runtime is not active")
        return runtime

    def _finish_runtime_turn(self, runtime: AgentRuntime, answer: str) -> None:
        self._llm_history = list(runtime.context.context[1:])
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

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
