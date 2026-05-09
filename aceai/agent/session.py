import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from msgspec import Struct
from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import delete as sql_delete
from sqlalchemy import inspect as sql_inspect
from sqlalchemy import insert as sql_insert
from sqlalchemy import select as sql_select
from sqlalchemy import update as sql_update
from sqlalchemy.engine import RowMapping
from typing_extensions import Self

from aceai.agent.cost import CostEstimate
from aceai.agent.citations import (
    citation_origin_name,
    citations_from_payload,
    message_with_citations,
)
from aceai.agent.project import ProjectMetadata, ProjectStore, default_project
from aceai.core.helpers.string import uuid_str
from aceai.llm.models import (
    LLMMessage,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
    LLMUsage,
)

SESSION_EVENT_VERSION = 1
SESSION_STATE_VERSION = 1
MAIN_THREAD_ID = "main"

SessionEventKind = Literal[
    "assistant_delta",
    "assistant_message",
    "assistant_tool_call",
    "error",
    "llm_completed",
    "llm_retrying",
    "llm_started",
    "media",
    "reasoning_summary",
    "run_completed",
    "run_failed",
    "run_suspended",
    "session_notice",
    "step_completed",
    "step_failed",
    "step_started",
    "thinking_delta",
    "tool_call_delta",
    "tool_approval_requested",
    "tool_approval_resolved",
    "tool_completed",
    "tool_failed",
    "tool_output",
    "tool_result",
    "tool_started",
    "user_message",
]


class SessionMetadata(Struct, frozen=True, kw_only=True):
    session_id: str
    project_id: str
    project_name: str
    created_at: datetime
    updated_at: datetime
    title: str
    path: str

    @classmethod
    def from_row(cls, row: RowMapping, *, files_dir: Path) -> Self:
        return cls(
            session_id=row["session_id"],
            project_id=row["project_id"],
            project_name=row["project_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row["title"],
            path=str(files_dir / row["path"]),
        )


AgentThreadRole = Literal["main", "subagent"]
AgentThreadStatus = Literal["idle", "running", "suspended", "completed", "failed"]


class AgentThreadMetadata(Struct, frozen=True, kw_only=True):
    session_id: str
    thread_id: str
    agent_id: str
    role: AgentThreadRole
    title: str
    status: AgentThreadStatus
    parent_thread_id: str | None
    parent_run_id: str | None
    parent_tool_call_id: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: RowMapping) -> Self:
        metadata = json.loads(row["metadata_json"])
        if not isinstance(metadata, dict):
            raise TypeError("thread metadata must be a mapping")
        return cls(
            session_id=row["session_id"],
            thread_id=row["thread_id"],
            agent_id=row["agent_id"],
            role=row["role"],
            title=row["title"],
            status=row["status"],
            parent_thread_id=row["parent_thread_id"],
            parent_run_id=row["parent_run_id"],
            parent_tool_call_id=row["parent_tool_call_id"],
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class SessionState(Struct, frozen=True, kw_only=True):
    selected_provider: str = ""
    selected_model: str = ""
    version: int = SESSION_STATE_VERSION

    @classmethod
    def empty(cls) -> Self:
        return cls()

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        if payload == {}:
            return cls.empty()
        selected_provider = payload["selected_provider"]
        selected_model = payload["selected_model"]
        version = payload["version"]
        if type(selected_provider) is not str:
            raise TypeError("Session state selected_provider must be str")
        if type(selected_model) is not str:
            raise TypeError("Session state selected_model must be str")
        if type(version) is not int:
            raise TypeError("Session state version must be int")
        return cls(
            selected_provider=selected_provider,
            selected_model=selected_model,
            version=version,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "selected_provider": self.selected_provider,
            "selected_model": self.selected_model,
        }


class SessionEvent(Struct, frozen=True, kw_only=True):
    kind: SessionEventKind
    payload: dict[str, Any]
    version: int = SESSION_EVENT_VERSION
    event_id: str = ""
    session_id: str = ""
    thread_id: str = MAIN_THREAD_ID
    agent_id: str = ""
    run_id: str = ""
    step_id: str | None = None
    step_index: int | None = None
    created_at: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        kind = payload["kind"]
        if kind not in (
            "assistant_delta",
            "assistant_message",
            "assistant_tool_call",
            "error",
            "llm_completed",
            "llm_retrying",
            "llm_started",
            "media",
            "reasoning_summary",
            "run_completed",
            "run_failed",
            "run_suspended",
            "session_notice",
            "step_completed",
            "step_failed",
            "step_started",
            "thinking_delta",
            "tool_call_delta",
            "tool_approval_requested",
            "tool_approval_resolved",
            "tool_completed",
            "tool_failed",
            "tool_output",
            "tool_result",
            "tool_started",
            "user_message",
        ):
            raise ValueError("Unsupported session event kind")
        if "thread_id" in payload:
            thread_id = payload["thread_id"]
        else:
            thread_id = MAIN_THREAD_ID
        if "agent_id" in payload:
            agent_id = payload["agent_id"]
        else:
            agent_id = ""
        return cls(
            version=payload["version"],
            event_id=payload["event_id"],
            session_id=payload["session_id"],
            thread_id=thread_id,
            agent_id=agent_id,
            run_id=payload["run_id"],
            step_id=payload["step_id"],
            step_index=payload["step_index"],
            kind=kind,
            created_at=payload["created_at"],
            payload=payload["payload"],
        )

    def with_session_defaults(self, *, session_id: str) -> Self:
        return type(self)(
            version=self.version,
            event_id=self.event_id or uuid_str(),
            session_id=self.session_id or session_id,
            thread_id=self.thread_id,
            agent_id=self.agent_id,
            run_id=self.run_id,
            step_id=self.step_id,
            step_index=self.step_index,
            kind=self.kind,
            created_at=self.created_at or _utc_now().isoformat(),
            payload=self.payload,
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "kind": self.kind,
            "created_at": self.created_at,
            "payload": self.payload,
        }


class _ToolBuffer(Struct, kw_only=True):
    name: str | None = None
    arguments: str = ""
    output: str = ""
    call: LLMToolCall | None = None


AgentRunStatus = Literal["running", "suspended", "completed", "failed"]


class AgentRunSummary(Struct, frozen=True, kw_only=True):
    run_id: str
    question: str
    status: AgentRunStatus
    final_answer: str
    step_count: int
    tool_call_count: int


class AgentRunLog(Struct, frozen=True, kw_only=True):
    run_id: str
    question: str
    status: AgentRunStatus
    final_answer: str
    events: list[SessionEvent]


class EventLog:
    def __init__(self, events: list[SessionEvent]) -> None:
        self.events = events

    def append(self, event: SessionEvent) -> None:
        self.events.append(event)

    def for_thread(self, thread_id: str) -> Self:
        return EventLog([event for event in self.events if event.thread_id == thread_id])

    def has_transcript(self) -> bool:
        for event in self.events:
            if event.kind in (
                "assistant_message",
                "assistant_tool_call",
                "error",
                "tool_result",
                "user_message",
            ):
                return True
        return False

    def get_run(self, run_id: str) -> AgentRunLog:
        if run_id == "":
            raise ValueError("run_id cannot be empty")
        events = [event for event in self.events if event.run_id == run_id]
        if not events:
            raise KeyError(run_id)
        return _run_log_from_events(run_id, events)

    def list_runs(self) -> list[AgentRunSummary]:
        run_ids: list[str] = []
        seen: set[str] = set()
        for event in self.events:
            if event.run_id == "" or event.run_id in seen:
                continue
            seen.add(event.run_id)
            run_ids.append(event.run_id)
        summaries: list[AgentRunSummary] = []
        for run_id in run_ids:
            run_log = self.get_run(run_id)
            summaries.append(
                AgentRunSummary(
                    run_id=run_log.run_id,
                    question=run_log.question,
                    status=run_log.status,
                    final_answer=run_log.final_answer,
                    step_count=_step_count(run_log.events),
                    tool_call_count=_tool_call_count(run_log.events),
                )
            )
        return summaries

    def replay_llm_history(self) -> list[LLMMessage]:
        history: list[LLMMessage] = []
        pending_tool_call: LLMToolCallMessage | None = None
        pending_tool_call_ids: set[str] = set()
        pending_tool_call_recorded = False
        for event in self.events:
            if event.kind == "user_message":
                pending_tool_call = None
                pending_tool_call_ids = set()
                pending_tool_call_recorded = False
                history.append(
                    LLMMessage.build(
                        role="user",
                        content=_user_message_history_content(event.payload),
                    )
                )
            elif event.kind == "assistant_message":
                pending_tool_call = None
                pending_tool_call_ids = set()
                pending_tool_call_recorded = False
                history.append(
                    LLMMessage.build(role="assistant", content=event.payload["content"])
                )
            elif event.kind == "assistant_tool_call":
                tool_calls = LLMToolCall.list_from_payload(event.payload)
                pending_tool_call = LLMToolCallMessage.from_content(
                    content=[]
                    if event.payload["content"] == ""
                    else event.payload["content"],
                    tool_calls=tool_calls,
                )
                pending_tool_call_ids = {call.call_id for call in tool_calls}
                pending_tool_call_recorded = False
            elif event.kind == "tool_result":
                if event.payload["tool_call_id"] not in pending_tool_call_ids:
                    continue
                if not pending_tool_call_recorded:
                    if pending_tool_call is None:
                        raise RuntimeError("Pending tool call history is missing")
                    history.append(pending_tool_call)
                    pending_tool_call_recorded = True
                history.append(
                    LLMToolUseMessage.from_content(
                        content=event.payload.get("model_output")
                        or event.payload["output"],
                        name=event.payload["tool_name"],
                        call_id=event.payload["tool_call_id"],
                    )
                )
                pending_tool_call_ids.remove(event.payload["tool_call_id"])
                if len(pending_tool_call_ids) == 0:
                    pending_tool_call = None
                    pending_tool_call_recorded = False
        return history

    def replay_export_text(self, metadata: SessionMetadata) -> str:
        lines = [
            f"# AceAI session {metadata.session_id}",
            f"project_id: {metadata.project_id}",
            f"project: {metadata.project_name}",
            f"title: {metadata.title}",
            f"created_at: {metadata.created_at.isoformat()}",
            f"updated_at: {metadata.updated_at.isoformat()}",
            "",
        ]
        for event in self.events:
            event_lines = _event_to_export_lines(event)
            if not event_lines:
                continue
            lines.extend(event_lines)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def replay_thread_export_text(
        self,
        metadata: SessionMetadata,
        thread: AgentThreadMetadata,
    ) -> str:
        lines = [
            f"# AceAI thread {thread.thread_id}",
            f"session_id: {metadata.session_id}",
            f"project_id: {metadata.project_id}",
            f"project: {metadata.project_name}",
            f"thread_id: {thread.thread_id}",
            f"agent_id: {thread.agent_id}",
            f"role: {thread.role}",
            f"title: {thread.title}",
            f"status: {thread.status}",
        ]
        if thread.parent_thread_id is not None:
            lines.append(f"parent_thread_id: {thread.parent_thread_id}")
        if thread.parent_run_id is not None:
            lines.append(f"parent_run_id: {thread.parent_run_id}")
        if thread.parent_tool_call_id is not None:
            lines.append(f"parent_tool_call_id: {thread.parent_tool_call_id}")
        lines.extend(
            [
                f"created_at: {thread.created_at.isoformat()}",
                f"updated_at: {thread.updated_at.isoformat()}",
                "",
            ]
        )
        for event in self.events:
            event_lines = _event_to_export_lines(event)
            if not event_lines:
                continue
            lines.extend(_event_diagnostic_lines(event))
            lines.extend(event_lines)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def total_cost_usd(self) -> float:
        total = 0.0
        for event in self.events:
            cost = _cost_from_event_payload(event.payload)
            if cost is not None:
                total += cost.total_cost_usd
        return total

    def last_user_message_created_at(self) -> datetime | None:
        for event in reversed(self.events):
            if event.kind == "user_message":
                return datetime.fromisoformat(event.created_at)
        return None

    def title_source(self) -> str:
        for event in self.events:
            if event.kind == "user_message" and event.payload["content"] != "":
                return event.payload["content"][:40]
        return "Empty session"


from aceai.agent.event_store import EventStore, JsonlEventStore  # noqa: E402


_metadata = MetaData()
_sessions_table = Table(
    "sessions",
    _metadata,
    Column("session_id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("project_name", String, nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, index=True),
    Column("title", String, nullable=False),
    Column("path", String, nullable=False),
    Column("state_json", String, nullable=False),
)

_threads_table = Table(
    "agent_threads",
    _metadata,
    Column("session_id", String, primary_key=True),
    Column("thread_id", String, primary_key=True),
    Column("agent_id", String, nullable=False),
    Column("role", String, nullable=False),
    Column("title", String, nullable=False),
    Column("status", String, nullable=False),
    Column("parent_thread_id", String, nullable=True),
    Column("parent_run_id", String, nullable=True),
    Column("parent_tool_call_id", String, nullable=True),
    Column("metadata_json", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class SessionStore:
    def __init__(
        self,
        root: Path | None = None,
        event_store: EventStore | None = None,
        project: ProjectMetadata | None = None,
    ) -> None:
        self.root = root or default_session_root()
        if project is not None:
            self.project = project
        elif root is not None:
            self.project = ProjectStore(self.root / "projects").resolve_project()
        else:
            self.project = default_project()
        self.project_id = self.project.project_id
        self.project_name = self.project.name
        self.db_path = self.root / "sessions.sqlite3"
        self.files_dir = self.root / "files"
        self.event_store = event_store or JsonlEventStore(self.files_dir)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_session(self) -> SessionMetadata:
        session_id = uuid_str()
        now = _utc_now()
        path = self.event_store.create_event_log(session_id)
        with self.engine.begin() as conn:
            conn.execute(
                sql_insert(_sessions_table).values(
                    session_id=session_id,
                    project_id=self.project_id,
                    project_name=self.project_name,
                    created_at=now,
                    updated_at=now,
                    title="New session",
                    path=path,
                    state_json=json.dumps(SessionState.empty().as_json()),
                )
            )
            conn.execute(
                sql_insert(_threads_table).values(
                    session_id=session_id,
                    thread_id=MAIN_THREAD_ID,
                    agent_id="",
                    role="main",
                    title="Main",
                    status="idle",
                    parent_thread_id=None,
                    parent_run_id=None,
                    parent_tool_call_id=None,
                    metadata_json="{}",
                    created_at=now,
                    updated_at=now,
                )
            )
        return SessionMetadata(
            session_id=session_id,
            project_id=self.project_id,
            project_name=self.project_name,
            created_at=now,
            updated_at=now,
            title="New session",
            path=str(self.files_dir / path),
        )

    def get_thread(
        self,
        session_id: str,
        thread_id: str = MAIN_THREAD_ID,
    ) -> AgentThreadMetadata:
        self.ensure_main_thread(session_id)
        query = sql_select(_threads_table).where(
            _threads_table.c.session_id == session_id,
            _threads_table.c.thread_id == thread_id,
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is None:
            raise KeyError(thread_id)
        return AgentThreadMetadata.from_row(row)

    def list_threads(self, session_id: str) -> list[AgentThreadMetadata]:
        self.ensure_main_thread(session_id)
        query = (
            sql_select(_threads_table)
            .where(_threads_table.c.session_id == session_id)
            .order_by(_threads_table.c.created_at.asc())
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        return [AgentThreadMetadata.from_row(row) for row in rows]

    def ensure_main_thread(self, session_id: str) -> AgentThreadMetadata:
        self.get_session(session_id)
        query = sql_select(_threads_table).where(
            _threads_table.c.session_id == session_id,
            _threads_table.c.thread_id == MAIN_THREAD_ID,
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is not None:
            return AgentThreadMetadata.from_row(row)
        now = _utc_now()
        with self.engine.begin() as conn:
            conn.execute(
                sql_insert(_threads_table).values(
                    session_id=session_id,
                    thread_id=MAIN_THREAD_ID,
                    agent_id="",
                    role="main",
                    title="Main",
                    status="idle",
                    parent_thread_id=None,
                    parent_run_id=None,
                    parent_tool_call_id=None,
                    metadata_json="{}",
                    created_at=now,
                    updated_at=now,
                )
            )
        return self.get_thread(session_id, MAIN_THREAD_ID)

    def create_thread(
        self,
        *,
        session_id: str,
        thread_id: str,
        agent_id: str,
        role: AgentThreadRole,
        title: str,
        status: AgentThreadStatus = "idle",
        parent_thread_id: str | None = None,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentThreadMetadata:
        self.get_session(session_id)
        now = _utc_now()
        if metadata is None:
            metadata = {}
        with self.engine.begin() as conn:
            conn.execute(
                sql_insert(_threads_table).values(
                    session_id=session_id,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    role=role,
                    title=title,
                    status=status,
                    parent_thread_id=parent_thread_id,
                    parent_run_id=parent_run_id,
                    parent_tool_call_id=parent_tool_call_id,
                    metadata_json=json.dumps(metadata, ensure_ascii=False),
                    created_at=now,
                    updated_at=now,
                )
            )
        return self.get_thread(session_id, thread_id)

    def update_thread_status(
        self,
        *,
        session_id: str,
        thread_id: str,
        status: AgentThreadStatus,
    ) -> None:
        self.get_thread(session_id, thread_id)
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_threads_table)
                .where(
                    _threads_table.c.session_id == session_id,
                    _threads_table.c.thread_id == thread_id,
                )
                .values(status=status, updated_at=_utc_now())
            )

    def get_session(self, session_id: str) -> SessionMetadata:
        query = sql_select(_sessions_table).where(
            _sessions_table.c.session_id == session_id
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is None:
            raise KeyError(session_id)
        return SessionMetadata.from_row(row, files_dir=self.files_dir)

    def list_sessions(self, project_id: str | None = None) -> list[SessionMetadata]:
        query = sql_select(_sessions_table).order_by(_sessions_table.c.created_at.desc())
        if project_id is not None:
            query = query.where(_sessions_table.c.project_id == project_id)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        sessions = [
            SessionMetadata.from_row(row, files_dir=self.files_dir) for row in rows
        ]
        if project_id is not None:
            return sessions
        sessions.sort(
            key=lambda session: (
                0 if session.project_id == self.project_id else 1,
                session.project_name,
                -session.created_at.timestamp(),
            )
        )
        return sessions

    def update_session_title(self, session_id: str, title: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(title=title, updated_at=_utc_now())
            )

    def get_session_state(self, session_id: str) -> SessionState:
        query = sql_select(_sessions_table.c.state_json).where(
            _sessions_table.c.session_id == session_id
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is None:
            raise KeyError(session_id)
        payload = json.loads(row["state_json"])
        if not isinstance(payload, dict):
            raise TypeError("Session state must be a mapping")
        return SessionState.from_json(payload)

    def update_session_state(self, session_id: str, state: SessionState) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(
                    state_json=json.dumps(state.as_json(), ensure_ascii=False),
                    updated_at=_utc_now(),
                )
            )

    def delete_session(self, session_id: str) -> None:
        metadata = self.get_session(session_id)
        with self.engine.begin() as conn:
            conn.execute(
                sql_delete(_threads_table).where(
                    _threads_table.c.session_id == session_id
                )
            )
            conn.execute(
                sql_delete(_sessions_table).where(
                    _sessions_table.c.session_id == session_id
                )
            )
        self.event_store.delete_event_log(metadata)
        shutil.rmtree(self.root / session_id, ignore_errors=True)

    def finalize_session_title(self, session_id: str) -> str:
        title_source = self.load_event_log(session_id).title_source()
        self.update_session_title(session_id, title_source)
        return title_source

    def append_event(self, session_id: str, event: SessionEvent) -> None:
        metadata = self.get_session(session_id)
        if event.thread_id == MAIN_THREAD_ID:
            self.ensure_main_thread(session_id)
        else:
            self.get_thread(session_id, event.thread_id)
        self.event_store.append_event(metadata, event)
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(updated_at=_utc_now())
            )

    def load_event_log(self, session_id: str) -> EventLog:
        metadata = self.get_session(session_id)
        return self.event_store.load_event_log(metadata)

    def load_thread_event_log(self, session_id: str, thread_id: str) -> EventLog:
        self.get_thread(session_id, thread_id)
        return self.load_event_log(session_id).for_thread(thread_id)

    def export_text(self, session_id: str, *, include_threads: bool = False) -> str:
        metadata = self.get_session(session_id)
        if not include_threads:
            return self.load_thread_event_log(
                session_id,
                MAIN_THREAD_ID,
            ).replay_export_text(metadata)
        parts: list[str] = []
        for thread in self.list_threads(session_id):
            parts.append(
                self.load_thread_event_log(
                    session_id,
                    thread.thread_id,
                ).replay_thread_export_text(metadata, thread)
            )
        return "\n---\n\n".join(parts)

    def total_cost_usd(self) -> float:
        total = 0.0
        for session in self.list_sessions():
            total += self.load_event_log(session.session_id).total_cost_usd()
        return total

    def _init_db(self) -> None:
        _metadata.create_all(self.engine)
        column_names = {
            column["name"] for column in sql_inspect(self.engine).get_columns("sessions")
        }
        if "state_json" not in column_names:
            with self.engine.begin() as conn:
                conn.exec_driver_sql(
                    "ALTER TABLE sessions ADD COLUMN state_json TEXT NOT NULL DEFAULT '{}'"
                )
        if "project_id" not in column_names:
            with self.engine.begin() as conn:
                conn.exec_driver_sql(
                    "ALTER TABLE sessions ADD COLUMN project_id TEXT NOT NULL DEFAULT ''"
                )
                conn.execute(
                    sql_update(_sessions_table).values(project_id=self.project_id)
                )
        if "project_name" not in column_names:
            with self.engine.begin() as conn:
                conn.exec_driver_sql(
                    "ALTER TABLE sessions ADD COLUMN project_name TEXT NOT NULL DEFAULT ''"
                )
                conn.execute(
                    sql_update(_sessions_table).values(project_name=self.project_name)
                )
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.project_id == "")
                .values(project_id=self.project_id)
            )
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.project_name == "")
                .values(project_name=self.project_name)
            )
        thread_column_names = {
            column["name"]
            for column in sql_inspect(self.engine).get_columns("agent_threads")
        }
        if "metadata_json" not in thread_column_names:
            with self.engine.begin() as conn:
                conn.exec_driver_sql(
                    "ALTER TABLE agent_threads ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )


class SessionRecorder:
    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._assistant_buffers: dict[str, str] = {}
        self._tools: dict[tuple[str, str], _ToolBuffer] = {}
        self._finalized = False
        self._saved = True
        self._last_recorded_event_id: str | None = None

    @property
    def saved(self) -> bool:
        return self._saved

    @property
    def last_recorded_event_id(self) -> str | None:
        return self._last_recorded_event_id

    def record(self, event: SessionEvent) -> str | None:
        if event.kind == "session_notice":
            return None
        if event.kind == "assistant_delta":
            self._assistant_buffers[event.thread_id] = (
                self._assistant_buffer(event.thread_id) + event.payload["content"]
            )
            return None
        if event.kind == "tool_call_delta":
            self._tool_for(event).arguments += event.payload["content"]
            return None
        if event.kind == "tool_started":
            return self._record_tool_started(event)
        if event.kind == "tool_output":
            self._tool_for(event).output += event.payload["content"]
            return None
        if event.kind == "tool_approval_requested":
            return self._record_tool_approval_requested(event)
        if event.kind == "llm_completed":
            return self._record_llm_completed(event)

        self.flush_assistant(event.thread_id)
        if event.kind == "user_message":
            return self._append_event(
                "user_message",
                _user_message_payload_for_record(event.payload),
                event,
            )
        elif event.kind in ("tool_completed", "tool_failed"):
            return self._record_tool_result(event)
        elif event.kind in ("run_failed", "step_failed"):
            return self._append_event(
                "error",
                {
                    "content": event.payload.get("error") or event.payload["content"],
                    "status": "failed",
                },
                event,
            )
        elif event.kind == "tool_approval_resolved":
            return self._record_tool_approval_resolved(event)
        elif event.kind in ("run_completed", "run_suspended", "step_completed", "step_started"):
            return self._append_event(event.kind, dict(event.payload), event)
        elif event.kind in (
            "llm_retrying",
            "media",
            "reasoning_summary",
            "thinking_delta",
        ):
            return self._append_event(event.kind, dict(event.payload), event)
        return None

    def flush_assistant(
        self,
        thread_id: str = MAIN_THREAD_ID,
        usage: LLMUsage | None = None,
        cost: CostEstimate | None = None,
    ) -> str | None:
        assistant_buffer = self._assistant_buffer(thread_id)
        if assistant_buffer == "":
            return None
        payload: dict[str, Any] = {"content": assistant_buffer}
        payload.update(_usage_payload(usage))
        payload.update(_cost_payload(cost))
        event_id = self._append_event(
            "assistant_message",
            payload,
            SessionEvent(
                event_id=uuid_str(),
                session_id=self.session_id,
                thread_id=thread_id,
                kind="assistant_message",
                payload=payload,
            ),
        )
        self._assistant_buffers[thread_id] = ""
        return event_id

    def finalize(self) -> bool:
        if self._finalized:
            return self._saved
        for thread_id in list(self._assistant_buffers):
            self.flush_assistant(thread_id)
        if not self.store.load_event_log(self.session_id).has_transcript():
            self.store.delete_session(self.session_id)
            self._saved = False
            self._finalized = True
            return self._saved
        self.store.finalize_session_title(self.session_id)
        self._saved = True
        self._finalized = True
        return self._saved

    def _record_llm_completed(self, event: SessionEvent) -> str | None:
        content = self._assistant_buffer(event.thread_id) or event.payload["content"]
        self._assistant_buffers[event.thread_id] = ""
        tool_calls = LLMToolCall.list_from_payload(event.payload)
        if tool_calls:
            return self._append_event(
                "assistant_tool_call",
                {
                    "content": "",
                    "tool_calls": [call.asdict() for call in tool_calls],
                    **_usage_payload(_usage_from_event_payload(event.payload)),
                    **_cost_payload(_cost_from_event_payload(event.payload)),
                },
                event,
            )
        if content != "":
            return self._append_event(
                "assistant_message",
                {
                    "content": content,
                    **_usage_payload(_usage_from_event_payload(event.payload)),
                    **_cost_payload(_cost_from_event_payload(event.payload)),
                },
                event,
            )
        return None

    def _record_tool_started(self, event: SessionEvent) -> str:
        tool_buffer = self._tool_for(event)
        tool_buffer.name = event.payload["tool_name"]
        call = LLMToolCall.from_payload(event.payload["tool_call"])
        tool_buffer.call = call
        tool_buffer.arguments = call.arguments
        return self._append_event("tool_started", dict(event.payload), event)

    def _record_tool_approval_requested(self, event: SessionEvent) -> str | None:
        tool_buffer = self._tool_for(event)
        if "tool_name" in event.payload:
            tool_buffer.name = event.payload["tool_name"]
        if "tool_call" in event.payload:
            call = LLMToolCall.from_payload(event.payload["tool_call"])
            tool_buffer.call = call
            tool_buffer.arguments = call.arguments
        if tool_buffer.name is None:
            return None
        return self._append_event(
            "tool_approval_requested",
            {
                "content": event.payload["content"],
                "tool_name": tool_buffer.name,
                "tool_call_id": event.payload["tool_call_id"],
                "tool_arguments": tool_buffer.arguments,
                "tool_call": event.payload["tool_call"],
            },
            event,
        )

    def _record_tool_approval_resolved(self, event: SessionEvent) -> str | None:
        tool_buffer = self._tool_for(event)
        if "tool_name" in event.payload:
            tool_buffer.name = event.payload["tool_name"]
        if "tool_call" in event.payload:
            call = LLMToolCall.from_payload(event.payload["tool_call"])
            tool_buffer.call = call
            tool_buffer.arguments = call.arguments
        if tool_buffer.name is None:
            return None
        return self._append_event(
            "tool_approval_resolved",
            {
                "content": event.payload["content"],
                "tool_name": tool_buffer.name,
                "tool_call_id": event.payload["tool_call_id"],
                "tool_arguments": tool_buffer.arguments,
                "tool_call": event.payload["tool_call"],
            },
            event,
        )

    def _record_tool_result(self, event: SessionEvent) -> str | None:
        tool_buffer = self._tool_for(event)
        if "tool_name" in event.payload:
            tool_buffer.name = event.payload["tool_name"]
        if "tool_call" in event.payload:
            call = LLMToolCall.from_payload(event.payload["tool_call"])
            tool_buffer.call = call
            tool_buffer.arguments = call.arguments
        if "tool_result" in event.payload:
            tool_buffer.output = event.payload["tool_result"]["output"]
        elif event.payload["content"] != "":
            tool_buffer.output = event.payload["content"]
        if tool_buffer.name is None:
            return None
        event_id = self._append_event(
            "tool_result",
            {
                "content": _tool_content(
                    event.kind,
                    event.payload.get("error") or event.payload["content"],
                    tool_buffer.output,
                ),
                "tool_name": tool_buffer.name,
                "tool_call_id": event.payload["tool_call_id"],
                "tool_arguments": tool_buffer.arguments,
                "output": tool_buffer.output,
                "model_output": event.payload.get("tool_result", {}).get(
                    "model_output",
                    tool_buffer.output,
                ),
                "status": "failed" if event.kind == "tool_failed" else "completed",
            },
            event,
        )
        self._tools.pop((event.thread_id, event.payload["tool_call_id"]), None)
        return event_id

    def _tool_for(self, event: SessionEvent) -> _ToolBuffer:
        if "tool_call_id" not in event.payload:
            raise ValueError("tool event must include tool_call_id")
        call_id = event.payload["tool_call_id"]
        key = (event.thread_id, call_id)
        tool_buffer = self._tools.get(key)
        if tool_buffer is None:
            tool_buffer = _ToolBuffer()
            self._tools[key] = tool_buffer
        return tool_buffer

    def _assistant_buffer(self, thread_id: str) -> str:
        if thread_id not in self._assistant_buffers:
            self._assistant_buffers[thread_id] = ""
        return self._assistant_buffers[thread_id]

    def _append_event(
        self,
        kind: SessionEventKind,
        payload: dict[str, Any],
        source_event: SessionEvent | None,
    ) -> str:
        if source_event is None:
            event_id = uuid_str()
        else:
            event_id = source_event.event_id
            if event_id == "":
                raise ValueError("source event must include event_id")
        self.store.append_event(
            self.session_id,
            SessionEvent(
                event_id=event_id,
                session_id=self.session_id,
                thread_id=MAIN_THREAD_ID if source_event is None else source_event.thread_id,
                agent_id="" if source_event is None else source_event.agent_id,
                run_id="" if source_event is None else source_event.run_id,
                step_id=None if source_event is None else source_event.step_id,
                step_index=None if source_event is None else source_event.step_index,
                kind=kind,
                payload=payload,
            ),
        )
        self._last_recorded_event_id = event_id
        return event_id


def default_session_root() -> Path:
    return Path.home() / ".aceai" / "sessions"


def _run_log_from_events(run_id: str, events: list[SessionEvent]) -> AgentRunLog:
    return AgentRunLog(
        run_id=run_id,
        question=_run_question(events),
        status=_run_status(events),
        final_answer=_run_final_answer(events),
        events=events,
    )


def _run_question(events: list[SessionEvent]) -> str:
    for event in events:
        if event.kind == "user_message":
            return event.payload["content"]
    return ""


def _run_status(events: list[SessionEvent]) -> AgentRunStatus:
    status: AgentRunStatus = "running"
    for event in events:
        if event.kind == "run_completed":
            status = "completed"
        elif event.kind in ("run_failed", "error"):
            status = "failed"
        elif event.kind == "run_suspended" and status != "completed":
            status = "suspended"
    return status


def _run_final_answer(events: list[SessionEvent]) -> str:
    for event in reversed(events):
        if event.kind == "run_completed":
            return event.payload["content"]
    return ""


def _step_count(events: list[SessionEvent]) -> int:
    step_ids: set[str] = set()
    for event in events:
        if event.step_id is not None:
            step_ids.add(event.step_id)
    return len(step_ids)


def _tool_call_count(events: list[SessionEvent]) -> int:
    tool_call_ids: set[str] = set()
    for event in events:
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is str and tool_call_id != "":
            tool_call_ids.add(tool_call_id)
    return len(tool_call_ids)


def _user_message_payload_for_record(payload: dict[str, Any]) -> dict[str, Any]:
    recorded: dict[str, Any] = {"content": payload["content"]}
    if "citations" in payload:
        recorded["citations"] = payload["citations"]
    return recorded


def _user_message_history_content(payload: dict[str, Any]) -> str:
    content = payload["content"]
    if type(content) is not str:
        raise TypeError("User message content must be str")
    if "citations" not in payload:
        return content
    citations = citations_from_payload(payload["citations"])
    return message_with_citations(content, citations)


def _event_to_export_lines(event: SessionEvent) -> list[str]:
    if event.kind == "user_message":
        lines = ["## user", event.payload["content"]]
        if "citations" in event.payload:
            lines.append("")
            lines.append("cited context:")
            for citation in citations_from_payload(event.payload["citations"]):
                lines.append(f"- {citation_origin_name(citation.origin)}")
                lines.append(citation.content)
        return lines
    if event.kind == "assistant_message":
        return ["## assistant", event.payload["content"]]
    if event.kind == "assistant_tool_call":
        lines = ["## assistant tool calls"]
        for call in LLMToolCall.list_from_payload(event.payload):
            lines.extend([f"tool: {call.name}", "arguments:", call.arguments])
        return lines
    if event.kind == "tool_result":
        name = event.payload["tool_name"]
        lines = [f"## tool: {name} ({event.payload['status']})"]
        if event.payload["tool_arguments"] != "":
            lines.extend(["arguments:", event.payload["tool_arguments"]])
        if event.payload["output"] != "":
            lines.extend(["output:", event.payload["output"]])
        return lines
    if event.kind == "tool_approval_requested":
        name = event.payload["tool_name"]
        lines = [f"## tool approval requested: {name}"]
        if event.payload["content"] != "":
            lines.append(event.payload["content"])
        if event.payload["tool_arguments"] != "":
            lines.extend(["arguments:", event.payload["tool_arguments"]])
        return lines
    if event.kind == "tool_approval_resolved":
        name = event.payload["tool_name"]
        return [f"## tool approval resolved: {name}", event.payload["content"]]
    if event.kind == "error":
        status = event.payload["status"]
        return [f"## error ({status})", event.payload["content"]]
    return []


def _event_diagnostic_lines(event: SessionEvent) -> list[str]:
    lines = [
        f"event_id: {event.event_id}",
        f"thread_id: {event.thread_id}",
        f"agent_id: {event.agent_id}",
        f"run_id: {event.run_id}",
    ]
    if event.step_id is not None:
        lines.append(f"step_id: {event.step_id}")
    if event.step_index is not None:
        lines.append(f"step_index: {event.step_index}")
    tool_call_id = event.payload.get("tool_call_id")
    if type(tool_call_id) is str:
        lines.append(f"tool_call_id: {tool_call_id}")
    lines.append("")
    return lines


def _usage_payload(usage: LLMUsage | None) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "usage": {
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_miss_input_tokens": usage.cache_miss_input_tokens,
            "input_cache_hit_rate": usage.input_cache_hit_rate,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
    }


def _cost_payload(cost: CostEstimate | None) -> dict[str, Any]:
    if cost is None:
        return {}
    return {
        "cost": {
            "model": cost.model,
            "input_cost_usd": cost.input_cost_usd,
            "cached_input_cost_usd": cost.cached_input_cost_usd,
            "output_cost_usd": cost.output_cost_usd,
            "total_cost_usd": cost.total_cost_usd,
            "input_usd_per_million": cost.input_usd_per_million,
            "cached_input_usd_per_million": cost.cached_input_usd_per_million,
            "output_usd_per_million": cost.output_usd_per_million,
            "pricing_source": cost.pricing_source,
        }
    }


def _usage_from_event_payload(payload: dict[str, Any]) -> LLMUsage | None:
    if "usage" not in payload:
        return None
    return LLMUsage.from_payload(payload["usage"])


def _cost_from_event_payload(payload: dict[str, Any]) -> CostEstimate | None:
    if "cost" not in payload:
        return None
    return CostEstimate.from_payload(payload["cost"])


def _tool_content(kind: SessionEventKind, error: str | None, output: str) -> str:
    if kind == "tool_failed":
        return error or output
    if '"entries":[' in output:
        entry_count = output.count('"name"')
        return f"completed - {entry_count} entries"
    if '"bytes_written":' in output:
        return "completed - file written"
    if '"matches":' in output:
        return "completed - search finished"
    if '"exit_code":' in output:
        return "completed - " + _shell_output_summary(output)
    return "completed"


def _shell_output_summary(output: str) -> str:
    payload = json.loads(output)
    exit_code = payload["exit_code"]
    stdout = payload["stdout"]
    stderr = payload["stderr"]
    if type(exit_code) is not int:
        raise TypeError("shell tool exit_code must be int")
    if type(stdout) is not str:
        raise TypeError("shell tool stdout must be str")
    if type(stderr) is not str:
        raise TypeError("shell tool stderr must be str")
    if exit_code == 0:
        return "succeeded"
    return f"exit {exit_code}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
