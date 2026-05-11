import asyncio
import os
from dataclasses import dataclass
from typing import AsyncGenerator
from msgspec import Struct
from opentelemetry.context import Context

from aceai.agent.citations import TurnCitation, message_with_citations
from aceai.agent.config import (
    AgentAppConfig,
    ReasoningLevel,
    replace_config,
    save_config,
)
from aceai.agent.features.delegation import (
    ChildAgentResult,
    build_child_agent_result,
    build_restored_delegated_child_agent,
    reset_delegated_child_run_recorder,
    set_delegated_child_run_recorder,
)
from aceai.agent.memory.ideas import Idea, IdeaStore
from aceai.agent.project import ProjectMetadata, default_project
from aceai.agent.provider_auth import default_api_key_for_provider
from aceai.agent.provider_catalog import (
    api_key_env,
    context_window_for_model_any_provider,
    model_options,
    supported_models,
    supports_reasoning_effort,
    supports_reasoning_effort_any_provider,
)
from aceai.agent.session import (
    EventLog,
    MAIN_THREAD_ID,
    SessionEvent,
    SessionRecorder,
    SessionState,
    SessionStore,
)
from aceai.agent.session_service import AgentSessionSnapshot, SessionService
from aceai.agent.memory.subagent_artifacts import SubagentArtifactStore
from aceai.core.helpers.string import uuid_str
from aceai.core import (
    Agent,
    AgentRunContext,
    Executor,
    ToolApprovalDecision,
    ToolExecutionError,
)
from aceai.core.events import (
    AgentEvent,
    ContextCompactionFailedEvent,
    ContextCompactionStartedEvent,
    ContextCompressedEvent,
    LLMCompletedEvent,
    RunFailedEvent,
    RunCompletedEvent,
    RunSuspendedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
)
from aceai.core.models import AgentStep
from aceai.core.models import ToolApprovalRequest
from aceai.llm.models import LLMMessage, LLMRequestMeta, LLMResponse

class AgentAppEvent(Struct, frozen=True, kw_only=True):
    thread_id: str
    agent_id: str
    event: AgentEvent


class AppRuntimeInfo(Struct, frozen=True, kw_only=True):
    provider_name: str
    selected_model: str
    default_model: str
    reasoning_level: ReasoningLevel
    supports_reasoning: bool
    context_window: int | None
    max_steps: str


@dataclass
class ChildThreadRuntime:
    thread_id: str
    agent_id: str
    run: AgentRunContext
    handoff: asyncio.Future[ChildAgentResult]
    task: asyncio.Task[None]
    intervention_count: int = 0


class AceAgentApp:
    """Reusable agent app runtime used by UI and future ports."""

    def __init__(
        self,
        agent: Agent,
        *,
        provider_name: str,
        selected_model: str,
        reasoning_level: ReasoningLevel = "auto",
        initial_history: list[LLMMessage] | None = None,
        session_service: SessionService | None = None,
        session_store: SessionStore | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        idea_store: IdeaStore | None = None,
        project: ProjectMetadata | None = None,
        trace_ctx: Context | None = None,
    ) -> None:
        self._agent = agent
        self._provider_name = provider_name
        self._selected_model = selected_model
        self._project = project or (
            session_store.project
            if session_store is not None
            else default_project()
        )
        self._reasoning_level: ReasoningLevel = "auto"
        self._request_meta: LLMRequestMeta = {"model": selected_model}
        self._set_reasoning_level_internal(reasoning_level)
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
        self._thread_runs: dict[str, AgentRunContext] = {}
        self._last_context_event_id: str | None = None
        self._last_context_event_ids: dict[str, str] = {}
        self._active_thread_id = MAIN_THREAD_ID
        self._thread_agents: dict[str, Agent] = {MAIN_THREAD_ID: self._agent}
        self._child_runtimes: dict[str, ChildThreadRuntime] = {}
        self._steered_child_run_ids: set[str] = set()
        self._stale_child_runtime_tasks: set[asyncio.Task[None]] = set()
        self._queued_questions: dict[str, list[str]] = {}
        self._app_event_subscribers: set[asyncio.Queue[AgentAppEvent | None]] = set()
        self._idea_store = idea_store or IdeaStore()
        self._approved_tool_names_by_thread: dict[str, set[str]] = {}
        self._child_context_event_ids: dict[str, str] = {}
        session_id = self._session_service.session_id
        if session_id is not None:
            if snapshot is None:
                snapshot = self._session_service.snapshot(session_id)
            self._approved_tool_names_by_thread[MAIN_THREAD_ID] = (
                _approved_tool_names_from_event_log(snapshot.event_log)
            )
            self._last_context_event_id = _last_context_source_event_id(snapshot.event_log)
            if self._last_context_event_id is not None:
                self._last_context_event_ids[MAIN_THREAD_ID] = self._last_context_event_id

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
    def reasoning_level(self) -> ReasoningLevel:
        return self._reasoning_level

    @property
    def model_options_text(self) -> str:
        return model_options_text_for(self._provider_name)

    def is_model_supported(self, model: str) -> bool:
        return model in supported_models(self._provider_name)

    def runtime_info(self) -> AppRuntimeInfo:
        agent = self._agent
        return AppRuntimeInfo(
            provider_name=self._provider_name,
            selected_model=self._selected_model,
            default_model=str(agent.default_model),
            reasoning_level=self._reasoning_level,
            supports_reasoning=supports_reasoning_for_model(self._selected_model),
            context_window=context_window_for_model(self._selected_model),
            max_steps=f"{agent.max_steps}",
        )

    def skill_summary_lines(self) -> list[str]:
        return [
            f"{skill.name}: {skill.description} ({skill.skill_file})"
            for skill in self._agent.skill_registry.get_skills()
        ]

    def tool_summary_lines(self) -> list[str]:
        executor = self._agent.executor
        if not isinstance(executor, Executor):
            return []
        lines: list[str] = []
        for tool in executor.tools.values():
            tags = ", ".join(tool.metadata.tags)
            tag_text = f" [{tags}]" if tags else ""
            lines.append(f"{tool.name}{tag_text}: {tool.description}")
        return lines

    def hosted_tool_summary_lines(self) -> list[str]:
        return [
            f"{tool.provider_name}:{tool.native_name}"
            for tool in self._agent.hosted_tools
        ]

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
        return tuple(self._queued_questions_for_active_thread())

    @property
    def active_thread_accepts_user_turn(self) -> bool:
        return self._active_thread_id not in self._child_runtimes

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
        self._thread_runs = {}
        self._active_thread_id = MAIN_THREAD_ID
        self._thread_agents = {MAIN_THREAD_ID: self._agent}
        self._child_runtimes = {}
        self._queued_questions = {}
        self._approved_tool_names_by_thread = {
            MAIN_THREAD_ID: _approved_tool_names_from_event_log(snapshot.event_log)
        }
        self._last_context_event_id = _last_context_source_event_id(snapshot.event_log)
        self._last_context_event_ids = {}
        if self._last_context_event_id is not None:
            self._last_context_event_ids[MAIN_THREAD_ID] = self._last_context_event_id
        return snapshot

    @property
    def active_thread_id(self) -> str:
        return self._active_thread_id

    def switch_thread(self, thread_id: str) -> AgentSessionSnapshot:
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("AceAI session is not active")
        snapshot = self._session_service.snapshot_thread(session_id, thread_id)
        self._active_thread_id = thread_id
        if thread_id == MAIN_THREAD_ID:
            self._llm_history = list(snapshot.history)
        self._approved_tool_names_by_thread[thread_id] = (
            _approved_tool_names_from_event_log(snapshot.event_log)
        )
        self._active_run = self._thread_runs.get(thread_id)
        latest_context_event_id = _last_context_source_event_id(snapshot.event_log)
        if latest_context_event_id is not None:
            self._last_context_event_ids[thread_id] = latest_context_event_id
        return snapshot

    def restore_history_from_active_session(self) -> None:
        session_id = self.session_id
        if session_id is None:
            self._llm_history = []
            self._active_run = None
            self._thread_runs = {}
            self._queued_questions = {}
            self._approved_tool_names_by_thread = {}
            self._last_context_event_id = None
            self._last_context_event_ids = {}
            return
        snapshot = self._session_service.snapshot_thread(
            session_id,
            self._active_thread_id,
        )
        if self._active_thread_id == MAIN_THREAD_ID:
            self._llm_history = snapshot.history
        latest_context_event_id = _last_context_source_event_id(snapshot.event_log)
        self._last_context_event_id = (
            latest_context_event_id if self._active_thread_id == MAIN_THREAD_ID else None
        )
        if latest_context_event_id is not None:
            self._last_context_event_ids[self._active_thread_id] = latest_context_event_id
        self._active_run = self._thread_runs.get(self._active_thread_id)

    def cancel_active_turn(self) -> None:
        self._active_run = None
        self._thread_runs.pop(self._active_thread_id, None)

    def enqueue_turn(self, question: str) -> int:
        if not self.active_thread_accepts_user_turn:
            raise RuntimeError(
                "Delegated subagent thread is still running; switch to main or wait "
                "for delegate_to_subagent to finish before sending a new message."
            )
        questions = self._queued_questions_for_active_thread()
        questions.append(question)
        return len(questions)

    def steer_active_child_thread(self, question: str) -> bool:
        thread_id = self._active_thread_id
        runtime = self._child_runtimes.get(thread_id)
        if runtime is None:
            return False
        if question == "":
            raise ValueError("child steer question cannot be empty")
        if runtime.task.done():
            self._child_runtimes.pop(thread_id, None)
            return False
        child_agent = self._agent_for_thread(thread_id)
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("AceAI session is not active")
        event_log = self._session_service.store.load_thread_event_log(
            session_id,
            thread_id,
        )
        child_run = child_agent.create_resume_run(
            question,
            event_log.replay_llm_history(),
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        child_run.run_state.tools.approved_tool_names = (
            self._approved_tool_names_for_thread(thread_id)
        )
        runtime.intervention_count += 1
        self._steered_child_run_ids.add(runtime.run.run_id)
        runtime.run = child_run
        self._thread_runs[thread_id] = child_run
        self._active_run = child_run
        event_id = self._record_child_steer_event(
            thread_id=thread_id,
            agent_id=runtime.agent_id,
            run_id=child_run.run_id,
            question=question,
        )
        self._child_context_event_ids[thread_id] = event_id
        self._last_context_event_ids[thread_id] = event_id
        old_task = runtime.task
        self._stale_child_runtime_tasks.add(old_task)
        old_task.cancel()
        runtime.task = asyncio.create_task(
            self._run_child_thread_runtime(
                thread_id=thread_id,
                child_agent=child_agent,
                child_run=child_run,
                handoff=runtime.handoff,
            )
        )
        runtime.task.add_done_callback(
            lambda completed_task: self._complete_child_runtime_task(
                thread_id=thread_id,
                task=completed_task,
                handoff=runtime.handoff,
            )
        )
        return True

    def pop_queued_turn(self) -> str | None:
        questions = self._queued_questions_for_active_thread()
        if not questions:
            return None
        return questions.pop(0)

    def take_queued_turn(self, index: int) -> str:
        return self._queued_questions_for_active_thread().pop(index)

    def cancel_queued_turn(self, index: int) -> str:
        return self._queued_questions_for_active_thread().pop(index)

    def switch_model(
        self,
        model: str,
        *,
        reasoning_level: ReasoningLevel | None = None,
    ) -> None:
        if not self.is_model_supported(model):
            raise ValueError(f"Unsupported model: {model}")
        self._selected_model = model
        self._request_meta["model"] = model
        next_level = (
            reasoning_level if reasoning_level is not None else self._reasoning_level
        )
        self._set_reasoning_level_internal(next_level)
        self.persist_session_state()

    def set_reasoning_level(self, level: ReasoningLevel) -> None:
        self._set_reasoning_level_internal(level)
        self.persist_session_state()

    def _set_reasoning_level_internal(self, level: ReasoningLevel) -> None:
        if level != "auto" and not supports_reasoning_effort(
            self._provider_name,
            self._selected_model,
        ):
            level = "auto"
        self._reasoning_level = level
        if level == "auto":
            self._request_meta.pop("reasoning", None)
        else:
            self._request_meta["reasoning"] = {"effort": level, "summary": "auto"}

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

    def pending_approval_request(
        self,
        *,
        thread_id: str | None = None,
    ) -> ToolApprovalRequest | None:
        run = self._run_for_thread(thread_id or self._active_thread_id)
        if run is None:
            return None
        pending = run.run_state.pending_approval
        if pending is None:
            return None
        return pending.request

    async def start_turn(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> AsyncGenerator[AgentEvent, None]:
        thread_id = self._active_thread_id
        async for app_event in self.start_turn_events(
            question,
            citations=citations,
        ):
            if app_event.thread_id == thread_id:
                yield app_event.event

    async def start_turn_events(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> AsyncGenerator[AgentAppEvent, None]:
        self.ensure_session()
        self.persist_session_state()
        thread_id = self._active_thread_id
        if thread_id in self._child_runtimes:
            self.steer_active_child_thread(question)
            return
        agent = self._agent_for_thread(thread_id)
        history = self._history_for_thread(thread_id)
        llm_question = message_with_citations(question, citations)
        self._active_run = agent.create_resume_run(
            llm_question,
            history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        self._thread_runs[thread_id] = self._active_run
        self._active_run.run_state.tools.approved_tool_names = (
            self._approved_tool_names_for_thread(thread_id)
        )
        self._last_context_event_id = self._session_service.record_user_message(
            question,
            citations=citations,
            run_id=self._active_run.run_id,
            thread_id=thread_id,
            agent_id="" if thread_id == MAIN_THREAD_ID else agent.agent_id,
        )
        self._last_context_event_ids[thread_id] = self._last_context_event_id
        async for event in self._consume_run_stream_events(
            self._active_run,
            agent.execute(self._active_run),
            thread_id=thread_id,
            agent_id="" if thread_id == MAIN_THREAD_ID else agent.agent_id,
        ):
            yield event

    async def approve_tool(
        self,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        target_thread_id = thread_id or self._active_thread_id
        request = self.pending_approval_request(thread_id=target_thread_id)
        if request is None:
            raise RuntimeError("No pending tool approval")
        self._validate_approval_target(
            thread_id=target_thread_id,
            request=request,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )
        async for event in self._resume_approval(
            ToolApprovalDecision.approve(request),
            thread_id=target_thread_id,
        ):
            yield event

    async def reject_tool(
        self,
        reason: str,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        target_thread_id = thread_id or self._active_thread_id
        request = self.pending_approval_request(thread_id=target_thread_id)
        if request is None:
            raise RuntimeError("No pending tool approval")
        self._validate_approval_target(
            thread_id=target_thread_id,
            request=request,
            run_id=run_id,
            tool_call_id=tool_call_id,
        )
        if reason == "":
            reason = "rejected by caller"
        async for event in self._resume_approval(
            ToolApprovalDecision.reject(request, reason=reason),
            thread_id=target_thread_id,
        ):
            yield event

    async def _resume_approval(
        self,
        decision: ToolApprovalDecision,
        *,
        thread_id: str,
    ) -> AsyncGenerator[AgentEvent, None]:
        run = self._current_run(thread_id=thread_id)
        async for event in self._consume_run_stream(
            run,
            self._agent_for_thread(thread_id).resume_approval(run, decision),
            thread_id=thread_id,
            agent_id="" if thread_id == MAIN_THREAD_ID else run.agent_id,
        ):
            yield event

    async def _consume_run_stream(
        self,
        run: AgentRunContext,
        stream: AsyncGenerator[AgentEvent, None],
        *,
        thread_id: str,
        agent_id: str,
    ) -> AsyncGenerator[AgentEvent, None]:
        async for app_event in self._consume_run_stream_events(
            run,
            stream,
            thread_id=thread_id,
            agent_id=agent_id,
        ):
            if app_event.thread_id == thread_id:
                yield app_event.event

    async def _consume_run_stream_events(
        self,
        run: AgentRunContext,
        stream: AsyncGenerator[AgentEvent, None],
        *,
        thread_id: str,
        agent_id: str,
    ) -> AsyncGenerator[AgentAppEvent, None]:
        queue: asyncio.Queue[AgentAppEvent | None] = asyncio.Queue()
        self._app_event_subscribers.add(queue)

        async def produce_events() -> None:
            recorder_token = set_delegated_child_run_recorder(self)
            try:
                async for event in stream:
                    if isinstance(event, ContextCompressedEvent):
                        self._record_context_checkpoint(event, thread_id=thread_id)
                    event = self._archive_subagent_artifact(event)
                    persisted_event_id: str | None = None
                    if not isinstance(
                        event,
                        ContextCompactionFailedEvent
                        | ContextCompactionStartedEvent
                        | ContextCompressedEvent,
                    ):
                        persisted_event_id = self._session_service.record_agent_event(
                            event,
                            thread_id=thread_id,
                            agent_id=agent_id,
                        )
                    if persisted_event_id is not None and _agent_event_updates_context(
                        event
                    ):
                        self._last_context_event_ids[thread_id] = persisted_event_id
                        if thread_id == MAIN_THREAD_ID:
                            self._last_context_event_id = persisted_event_id
                    if isinstance(event, RunCompletedEvent):
                        self._finish_run_turn(
                            run,
                            event.final_answer,
                            thread_id=thread_id,
                        )
                    self._emit_app_event(
                        thread_id=thread_id,
                        agent_id=agent_id,
                        event=event,
                    )
            finally:
                reset_delegated_child_run_recorder(recorder_token)
                await stream.aclose()
                queue.put_nowait(None)

        producer = asyncio.create_task(produce_events())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            self._app_event_subscribers.discard(queue)
            if not producer.done():
                producer.cancel()
            await producer

    def _run_for_thread(self, thread_id: str) -> AgentRunContext | None:
        return self._thread_runs.get(thread_id)

    def _approved_tool_names_for_thread(self, thread_id: str) -> set[str]:
        if thread_id not in self._approved_tool_names_by_thread:
            session_id = self.session_id
            if session_id is None:
                self._approved_tool_names_by_thread[thread_id] = set()
            else:
                snapshot = self._session_service.snapshot_thread(session_id, thread_id)
                self._approved_tool_names_by_thread[thread_id] = (
                    _approved_tool_names_from_event_log(snapshot.event_log)
                )
        return self._approved_tool_names_by_thread[thread_id]

    def _current_run(self, *, thread_id: str | None = None) -> AgentRunContext:
        run = self._run_for_thread(thread_id or self._active_thread_id)
        if run is None:
            raise RuntimeError("AceAI run is not active")
        return run

    def _validate_approval_target(
        self,
        *,
        thread_id: str,
        request: ToolApprovalRequest,
        run_id: str | None,
        tool_call_id: str | None,
    ) -> None:
        run = self._current_run(thread_id=thread_id)
        if run_id is not None and run.run_id != run_id:
            raise RuntimeError("approval target run_id does not match pending run")
        if tool_call_id is not None and request.call.call_id != tool_call_id:
            raise RuntimeError("approval target tool_call_id does not match pending tool call")

    def _finish_run_turn(
        self,
        run: AgentRunContext,
        answer: str,
        *,
        thread_id: str,
    ) -> None:
        if thread_id != MAIN_THREAD_ID:
            return
        self._llm_history = list(run.context.context[1:])
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

    def _record_context_checkpoint(
        self,
        event: ContextCompressedEvent,
        *,
        thread_id: str,
    ) -> None:
        included_event_id = self._last_context_event_ids.get(thread_id)
        if included_event_id is None:
            raise RuntimeError("Context checkpoint has no included transcript event")
        if thread_id == MAIN_THREAD_ID:
            self._session_service.record_context_checkpoint(
                event,
                included_event_id=included_event_id,
            )
            return
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("AceAI session is not active")
        self._session_service.context_checkpoint_store.record_checkpoint(
            session_id=session_id,
            thread_id=thread_id,
            run_id=event.run_id,
            step_id=event.step_id,
            reason=event.reason,
            compression_count=event.compression_count,
            included_event_id=included_event_id,
            history=event.history,
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
        return {**self._request_meta, "model": self._selected_model}

    def start_child_thread(
        self,
        *,
        task: str,
        instructions: str,
        context_brief: str,
        allowed_tools: list[str],
        child_agent: Agent,
        agent_id: str,
        run_id: str,
        child_question: str,
    ) -> str:
        self.ensure_session()
        thread = self._session_service.create_subagent_thread(
            task=task,
            agent_id=agent_id,
            parent_run_id=self._current_run().run_id,
            instructions=instructions,
            context_brief=context_brief,
            allowed_tools=allowed_tools,
        )
        self._thread_agents[thread.thread_id] = child_agent
        event_id = self._session_service.record_user_message(
            child_question,
            run_id=run_id,
            thread_id=thread.thread_id,
            agent_id=agent_id,
        )
        self._child_context_event_ids[thread.thread_id] = event_id
        self._last_context_event_ids[thread.thread_id] = event_id
        return thread.thread_id

    async def run_child_thread(
        self,
        *,
        task: str,
        instructions: str,
        context_brief: str,
        allowed_tools: list[str],
        child_agent: Agent,
        child_run: AgentRunContext,
        child_question: str,
    ) -> ChildAgentResult:
        thread_id = self.start_child_thread(
            task=task,
            instructions=instructions,
            context_brief=context_brief,
            allowed_tools=allowed_tools,
            child_agent=child_agent,
            agent_id=child_agent.agent_id,
            run_id=child_run.run_id,
            child_question=child_question,
        )
        loop = asyncio.get_running_loop()
        handoff: asyncio.Future[ChildAgentResult] = loop.create_future()
        runtime_task = asyncio.create_task(
            self._run_child_thread_runtime(
                thread_id=thread_id,
                child_agent=child_agent,
                child_run=child_run,
                handoff=handoff,
            )
        )
        self._child_runtimes[thread_id] = ChildThreadRuntime(
            thread_id=thread_id,
            agent_id=child_agent.agent_id,
            run=child_run,
            handoff=handoff,
            task=runtime_task,
        )
        runtime_task.add_done_callback(
            lambda completed_task: self._complete_child_runtime_task(
                thread_id=thread_id,
                task=completed_task,
                handoff=handoff,
            )
        )
        try:
            return await handoff
        except asyncio.CancelledError:
            runtime = self._child_runtimes.get(thread_id)
            task_to_cancel = runtime.task if runtime is not None else runtime_task
            if not task_to_cancel.done():
                task_to_cancel.cancel()
            self.finish_child_thread(
                thread_id=thread_id,
                status="failed",
            )
            try:
                await task_to_cancel
            except asyncio.CancelledError:
                pass
            raise

    async def _run_child_thread_runtime(
        self,
        *,
        thread_id: str,
        child_agent: Agent,
        child_run: AgentRunContext,
        handoff: asyncio.Future[ChildAgentResult],
    ) -> None:
        events: list[AgentEvent] = []
        final_answer = ""
        try:
            async for event in child_agent.execute(child_run):
                events.append(event)
                self.record_child_event(
                    thread_id=thread_id,
                    agent_id=child_agent.agent_id,
                    event=event,
                )
                if isinstance(event, RunCompletedEvent):
                    final_answer = event.final_answer
                    self.finish_child_thread(
                        thread_id=thread_id,
                        status=child_run.status,
                    )
                    if not handoff.done():
                        handoff.set_result(
                            build_child_agent_result(
                                thread_id=thread_id,
                                child_agent=child_agent,
                                child_run=child_run,
                                events=events,
                                final_answer=final_answer,
                            )
                        )
                    return
                if isinstance(event, RunSuspendedEvent):
                    self.finish_child_thread(
                        thread_id=thread_id,
                        status="suspended",
                    )
                    if not handoff.done():
                        handoff.set_exception(
                            ToolExecutionError(
                                "delegated subagent suspended for approval; "
                                "delegate_to_subagent only supports approval-free child tools"
                            )
                        )
                    return
                if isinstance(event, RunFailedEvent):
                    self.finish_child_thread(
                        thread_id=thread_id,
                        status="failed",
                    )
                    if not handoff.done():
                        handoff.set_exception(ToolExecutionError(event.error))
                    return
        except asyncio.CancelledError:
            if child_run.run_id in self._steered_child_run_ids:
                self._steered_child_run_ids.remove(child_run.run_id)
                raise
            error = "delegated subagent run was cancelled before a terminal event"
            child_run.run_state.status = "failed"
            event = _cancelled_child_run_failed_event(
                child_run=child_run,
                events=events,
                error=error,
            )
            events.append(event)
            self.record_child_event(
                thread_id=thread_id,
                agent_id=child_agent.agent_id,
                event=event,
            )
            self.finish_child_thread(
                thread_id=thread_id,
                status="failed",
            )
            if not handoff.done():
                handoff.set_exception(ToolExecutionError(error))
            raise
        if child_run.status == "running":
            if child_run.run_id in self._steered_child_run_ids:
                self._steered_child_run_ids.remove(child_run.run_id)
                return
            error = "delegated subagent run was cancelled before a terminal event"
            child_run.run_state.status = "failed"
            event = _cancelled_child_run_failed_event(
                child_run=child_run,
                events=events,
                error=error,
            )
            events.append(event)
            self.record_child_event(
                thread_id=thread_id,
                agent_id=child_agent.agent_id,
                event=event,
            )
            self.finish_child_thread(
                thread_id=thread_id,
                status="failed",
            )
            if not handoff.done():
                handoff.set_exception(ToolExecutionError(error))
            return
        self.finish_child_thread(
            thread_id=thread_id,
            status=child_run.status,
        )
        if not handoff.done():
            handoff.set_result(
                build_child_agent_result(
                    thread_id=thread_id,
                    child_agent=child_agent,
                    child_run=child_run,
                    events=events,
                    final_answer=final_answer,
                )
            )

    def _complete_child_runtime_task(
        self,
        *,
        thread_id: str,
        task: asyncio.Task[None],
        handoff: asyncio.Future[ChildAgentResult],
    ) -> None:
        if task in self._stale_child_runtime_tasks:
            self._stale_child_runtime_tasks.remove(task)
            return
        runtime = self._child_runtimes.get(thread_id)
        if runtime is not None and runtime.task is not task:
            return
        self._child_runtimes.pop(thread_id, None)
        if task.cancelled():
            if not handoff.done():
                handoff.cancel()
            return
        exception = task.exception()
        if exception is not None and not handoff.done():
            self.finish_child_thread(
                thread_id=thread_id,
                status="failed",
            )
            handoff.set_exception(exception)

    def _record_child_steer_event(
        self,
        *,
        thread_id: str,
        agent_id: str,
        run_id: str,
        question: str,
    ) -> str:
        event_id = self._session_service.record_session_event(
            SessionEvent(
                event_id=uuid_str(),
                thread_id=thread_id,
                agent_id=agent_id,
                run_id=run_id,
                step_id=None,
                step_index=None,
                kind="user_steer",
                payload={"content": question},
            )
        )
        if event_id is None:
            raise RuntimeError("user_steer did not persist a session event")
        return event_id

    def record_child_event(
        self,
        *,
        thread_id: str,
        agent_id: str,
        event: AgentEvent,
    ) -> None:
        if isinstance(event, ContextCompressedEvent):
            included_event_id = self._child_context_event_ids[thread_id]
            session_id = self._session_service.session_id
            if session_id is None:
                raise RuntimeError("AceAI session is not active")
            self._session_service.context_checkpoint_store.record_checkpoint(
                session_id=session_id,
                thread_id=thread_id,
                run_id=event.run_id,
                step_id=event.step_id,
                reason=event.reason,
                compression_count=event.compression_count,
                included_event_id=included_event_id,
                history=event.history,
            )
            self._emit_app_event(
                thread_id=thread_id,
                agent_id=agent_id,
                event=event,
            )
            return
        if isinstance(
            event,
            ContextCompactionFailedEvent
            | ContextCompactionStartedEvent,
        ):
            self._emit_app_event(
                thread_id=thread_id,
                agent_id=agent_id,
                event=event,
            )
            return
        persisted_event_id = self._session_service.record_agent_event(
            event,
            thread_id=thread_id,
            agent_id=agent_id,
        )
        if persisted_event_id is not None and _agent_event_updates_context(event):
            self._child_context_event_ids[thread_id] = persisted_event_id
            self._last_context_event_ids[thread_id] = persisted_event_id
        self._emit_app_event(
            thread_id=thread_id,
            agent_id=agent_id,
            event=event,
        )

    def finish_child_thread(
        self,
        *,
        thread_id: str,
        status: str,
    ) -> None:
        if status not in ("idle", "running", "suspended", "completed", "failed"):
            raise ValueError("Unsupported child thread status")
        self._session_service.update_thread_status(
            thread_id=thread_id,
            status=status,
        )

    def _emit_app_event(
        self,
        *,
        thread_id: str,
        agent_id: str,
        event: AgentEvent,
    ) -> None:
        app_event = AgentAppEvent(
            thread_id=thread_id,
            agent_id=agent_id,
            event=event,
        )
        for queue in tuple(self._app_event_subscribers):
            queue.put_nowait(app_event)

    def _queued_questions_for_active_thread(self) -> list[str]:
        if self._active_thread_id not in self._queued_questions:
            self._queued_questions[self._active_thread_id] = []
        return self._queued_questions[self._active_thread_id]

    def _agent_for_thread(self, thread_id: str) -> Agent:
        if thread_id in self._thread_agents:
            return self._thread_agents[thread_id]
        agent = self._restore_agent_for_thread(thread_id)
        self._thread_agents[thread_id] = agent
        return agent

    def _restore_agent_for_thread(self, thread_id: str) -> Agent:
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("AceAI session is not active")
        thread = self._session_service.store.get_thread(session_id, thread_id)
        if thread.role != "subagent":
            raise RuntimeError("Only subagent threads can be restored")
        executor = self._agent.executor
        if not isinstance(executor, Executor):
            raise RuntimeError("Subagent thread restore requires an Executor")
        metadata = thread.metadata
        instructions = metadata["instructions"]
        context_brief = metadata["context_brief"]
        allowed_tools = metadata["allowed_tools"]
        if type(instructions) is not str:
            raise TypeError("subagent thread instructions must be str")
        if type(context_brief) is not str:
            raise TypeError("subagent thread context_brief must be str")
        if type(allowed_tools) is not list:
            raise TypeError("subagent thread allowed_tools must be list")
        for tool_name in allowed_tools:
            if type(tool_name) is not str:
                raise TypeError("subagent thread allowed_tools entries must be str")
        available_tools = [
            tool
            for tool_name, tool in executor.tools.items()
            if tool_name != "delegate_to_subagent"
        ]
        return build_restored_delegated_child_agent(
            llm_service=self._agent.llm_service,
            default_model=self._agent.default_model,
            instructions=instructions,
            allowed_tools=allowed_tools,
            available_tools=available_tools,
            available_hosted_tools=executor.hosted_tools,
            agent_id=thread.agent_id,
        )

    def _history_for_thread(self, thread_id: str) -> list[LLMMessage]:
        if thread_id == MAIN_THREAD_ID:
            return self._llm_history
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("AceAI session is not active")
        snapshot = self._session_service.snapshot_thread(session_id, thread_id)
        latest_context_event_id = _last_context_source_event_id(snapshot.event_log)
        if latest_context_event_id is not None:
            self._last_context_event_ids[thread_id] = latest_context_event_id
        return snapshot.history


def effective_reasoning_level(
    provider_name: str,
    model: str,
    level: ReasoningLevel,
) -> ReasoningLevel:
    if level == "auto":
        return "auto"
    if not supports_reasoning_effort(provider_name, model):
        return "auto"
    return level


def model_options_text_for(provider_name: str) -> str:
    names = ", ".join(option[1] for option in model_options(provider_name))
    return f"Available models: {names}"


def is_model_supported(provider_name: str, model: str) -> bool:
    return model in supported_models(provider_name)


def resolve_provider_api_key(provider_name: str) -> str:
    """Resolve an API key for the provider from env var, then default auth file.

    Returns "" if neither source has a key.
    """

    env_name = api_key_env(provider_name)
    api_key = os.environ.get(env_name, "")
    if api_key != "":
        return api_key
    return default_api_key_for_provider(provider_name)


def normalize_user_config(
    config: AgentAppConfig,
    *,
    persist: bool = False,
) -> AgentAppConfig:
    """Validate the config and optionally persist it to the project file."""

    next_config = replace_config(config)
    if persist:
        save_config(next_config)
    return next_config


def context_window_for_model(model: str | None) -> int | None:
    if model is None or model == "":
        return None
    return context_window_for_model_any_provider(model)


def supports_reasoning_for_model(model: str | None) -> bool:
    if model is None or model == "":
        return False
    return supports_reasoning_effort_any_provider(model)


def _agent_event_updates_context(event: AgentEvent) -> bool:
    return isinstance(event, LLMCompletedEvent | ToolCompletedEvent | ToolFailedEvent)


def _cancelled_child_run_failed_event(
    *,
    child_run: AgentRunContext,
    events: list[AgentEvent],
    error: str,
) -> RunFailedEvent:
    if events:
        step_index = events[-1].step_index
        step_id = events[-1].step_id
        failed_step = AgentStep(
            step_id=step_id,
            llm_response=LLMResponse(text="", status="failed"),
        )
    elif child_run.steps:
        step_index = len(child_run.steps)
        step_id = child_run.steps[-1].step_id
        failed_step = AgentStep(
            step_id=step_id,
            llm_response=LLMResponse(text="", status="failed"),
        )
    else:
        step_index = len(child_run.steps)
        failed_step = AgentStep(
            llm_response=LLMResponse(text="", status="failed"),
        )
        step_id = failed_step.step_id
    return RunFailedEvent(
        run_id=child_run.run_id,
        step_index=step_index,
        step_id=failed_step.step_id,
        step=failed_step,
        error=error,
    )


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
