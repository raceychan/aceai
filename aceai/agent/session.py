import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from msgspec import Struct
from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import delete as sql_delete
from sqlalchemy import insert as sql_insert
from sqlalchemy import select as sql_select
from sqlalchemy import update as sql_update
from sqlalchemy.engine import RowMapping
from typing_extensions import Self

from aceai.agent.cost import CostEstimate
from aceai.core.helpers.string import uuid_str
from aceai.llm.models import (
    LLMMessage,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
    LLMUsage,
)

SESSION_EVENT_VERSION = 1

SessionEventKind = Literal[
    "assistant_delta",
    "assistant_message",
    "assistant_tool_call",
    "error",
    "llm_completed",
    "llm_started",
    "media",
    "reasoning_summary",
    "run_completed",
    "run_failed",
    "session_notice",
    "step_completed",
    "step_failed",
    "step_started",
    "thinking_delta",
    "tool_call_delta",
    "tool_completed",
    "tool_failed",
    "tool_output",
    "tool_result",
    "tool_started",
    "user_message",
]


class SessionMetadata(Struct, frozen=True, kw_only=True):
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: str
    path: str

    @classmethod
    def from_row(cls, row: RowMapping, *, files_dir: Path) -> Self:
        return cls(
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row["title"],
            path=str(files_dir / row["path"]),
        )


class SessionEvent(Struct, frozen=True, kw_only=True):
    kind: SessionEventKind
    payload: dict[str, Any]
    version: int = SESSION_EVENT_VERSION
    event_id: str = ""
    session_id: str = ""
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
            "llm_started",
            "media",
            "reasoning_summary",
            "run_completed",
            "run_failed",
            "session_notice",
            "step_completed",
            "step_failed",
            "step_started",
            "thinking_delta",
            "tool_call_delta",
            "tool_completed",
            "tool_failed",
            "tool_output",
            "tool_result",
            "tool_started",
            "user_message",
        ):
            raise ValueError("Unsupported session event kind")
        return cls(
            version=payload["version"],
            event_id=payload["event_id"],
            session_id=payload["session_id"],
            run_id=payload["run_id"],
            step_id=payload["step_id"],
            step_index=payload["step_index"],
            kind=kind,
            created_at=payload["created_at"],
            payload=payload["payload"],
        )

    def with_session_defaults(self, *, session_id: str) -> Self:
        return SessionEvent(
            version=self.version,
            event_id=self.event_id or uuid_str(),
            session_id=self.session_id or session_id,
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


class EventLog:
    def __init__(self, events: list[SessionEvent]) -> None:
        self.events = events

    def append(self, event: SessionEvent) -> None:
        self.events.append(event)

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

    def replay_llm_history(self) -> list[LLMMessage]:
        history: list[LLMMessage] = []
        for event in self.events:
            if event.kind == "user_message":
                history.append(
                    LLMMessage.build(role="user", content=event.payload["content"])
                )
            elif event.kind == "assistant_message":
                history.append(
                    LLMMessage.build(role="assistant", content=event.payload["content"])
                )
            elif event.kind == "assistant_tool_call":
                history.append(
                    LLMToolCallMessage.from_content(
                        content=event.payload["content"],
                        tool_calls=LLMToolCall.list_from_payload(event.payload),
                    )
                )
            elif event.kind == "tool_result":
                history.append(
                    LLMToolUseMessage.from_content(
                        content=event.payload["output"],
                        name=event.payload["tool_name"],
                        call_id=event.payload["tool_call_id"],
                    )
                )
        return history

    def replay_export_text(self, metadata: SessionMetadata) -> str:
        lines = [
            f"# AceAI session {metadata.session_id}",
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

    def total_cost_usd(self) -> float:
        total = 0.0
        for event in self.events:
            cost = _cost_from_event_payload(event.payload)
            if cost is not None:
                total += cost.total_cost_usd
        return total

    def title_source(self) -> str:
        for event in self.events:
            if event.kind == "user_message" and event.payload["content"] != "":
                return event.payload["content"][:40]
        return "Empty session"


_metadata = MetaData()
_sessions_table = Table(
    "sessions",
    _metadata,
    Column("session_id", String, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, index=True),
    Column("title", String, nullable=False),
    Column("path", String, nullable=False),
)


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_session_root()
        self.db_path = self.root / "sessions.sqlite3"
        self.files_dir = self.root / "files"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_session(self) -> SessionMetadata:
        session_id = uuid_str()
        now = _utc_now()
        path = self.files_dir / f"{session_id}.events.jsonl"
        path.write_text("", encoding="utf-8")
        with self.engine.begin() as conn:
            conn.execute(
                sql_insert(_sessions_table).values(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    title="New session",
                    path=path.name,
                )
            )
        return SessionMetadata(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            title="New session",
            path=str(path),
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

    def list_sessions(self) -> list[SessionMetadata]:
        query = sql_select(_sessions_table).order_by(
            _sessions_table.c.updated_at.desc()
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        return [SessionMetadata.from_row(row, files_dir=self.files_dir) for row in rows]

    def update_session_title(self, session_id: str, title: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(title=title, updated_at=_utc_now())
            )

    def delete_session(self, session_id: str) -> None:
        metadata = self.get_session(session_id)
        with self.engine.begin() as conn:
            conn.execute(
                sql_delete(_sessions_table).where(
                    _sessions_table.c.session_id == session_id
                )
            )
        Path(metadata.path).unlink()

    def finalize_session_title(self, session_id: str) -> str:
        title_source = self.load_event_log(session_id).title_source()
        self.update_session_title(session_id, title_source)
        return title_source

    def append_event(self, session_id: str, event: SessionEvent) -> None:
        metadata = self.get_session(session_id)
        persisted_event = event.with_session_defaults(session_id=session_id)
        with Path(metadata.path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(persisted_event.as_json(), ensure_ascii=False))
            stream.write("\n")
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(updated_at=_utc_now())
            )

    def load_event_log(self, session_id: str) -> EventLog:
        metadata = self.get_session(session_id)
        path = Path(metadata.path)
        events: list[SessionEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "":
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("Session event must be a mapping")
            events.append(SessionEvent.from_json(payload))
        return EventLog(events)

    def export_text(self, session_id: str) -> str:
        metadata = self.get_session(session_id)
        return self.load_event_log(session_id).replay_export_text(metadata)

    def total_cost_usd(self) -> float:
        total = 0.0
        for session in self.list_sessions():
            total += self.load_event_log(session.session_id).total_cost_usd()
        return total

    def _init_db(self) -> None:
        _metadata.create_all(self.engine)

class SessionRecorder:
    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._assistant_buffer = ""
        self._tools: dict[str, _ToolBuffer] = {}
        self._finalized = False
        self._saved = True

    @property
    def saved(self) -> bool:
        return self._saved

    def record(self, event: SessionEvent) -> None:
        if event.kind == "session_notice":
            return
        if event.kind == "assistant_delta":
            self._assistant_buffer += event.payload["content"]
            return
        if event.kind == "thinking_delta":
            return
        if event.kind == "tool_call_delta":
            self._tool_for(event).arguments += event.payload["content"]
            return
        if event.kind == "tool_started":
            self._record_tool_started(event)
            return
        if event.kind == "tool_output":
            self._tool_for(event).output += event.payload["content"]
            return
        if event.kind == "llm_completed":
            self._record_llm_completed(event)
            return

        self.flush_assistant()
        if event.kind == "user_message":
            self._append_event(
                "user_message",
                {"content": event.payload["content"]},
                event,
            )
        elif event.kind in ("tool_completed", "tool_failed"):
            self._record_tool_result(event)
        elif event.kind in ("run_failed", "step_failed"):
            self._append_event(
                "error",
                {
                    "content": event.payload.get("error") or event.payload["content"],
                    "status": "failed",
                },
                event,
            )
        elif event.kind in ("run_completed", "step_completed", "step_started"):
            self._append_event(event.kind, dict(event.payload), event)
        elif event.kind in ("media", "reasoning_summary"):
            self._append_event(event.kind, dict(event.payload), event)

    def flush_assistant(
        self,
        usage: LLMUsage | None = None,
        cost: CostEstimate | None = None,
    ) -> None:
        if self._assistant_buffer == "":
            return
        payload: dict[str, Any] = {"content": self._assistant_buffer}
        payload.update(_usage_payload(usage))
        payload.update(_cost_payload(cost))
        self._append_event("assistant_message", payload, None)
        self._assistant_buffer = ""

    def finalize(self) -> bool:
        if self._finalized:
            return self._saved
        self.flush_assistant()
        if not self.store.load_event_log(self.session_id).has_transcript():
            self.store.delete_session(self.session_id)
            self._saved = False
            self._finalized = True
            return self._saved
        self.store.finalize_session_title(self.session_id)
        self._saved = True
        self._finalized = True
        return self._saved

    def _record_llm_completed(self, event: SessionEvent) -> None:
        content = self._assistant_buffer or event.payload["content"]
        self._assistant_buffer = ""
        tool_calls = LLMToolCall.list_from_payload(event.payload)
        if tool_calls:
            self._append_event(
                "assistant_tool_call",
                {
                    "content": content,
                    "tool_calls": [call.asdict() for call in tool_calls],
                    **_usage_payload(_usage_from_event_payload(event.payload)),
                    **_cost_payload(_cost_from_event_payload(event.payload)),
                },
                event,
            )
            return
        if content != "":
            self._append_event(
                "assistant_message",
                {
                    "content": content,
                    **_usage_payload(_usage_from_event_payload(event.payload)),
                    **_cost_payload(_cost_from_event_payload(event.payload)),
                },
                event,
            )

    def _record_tool_started(self, event: SessionEvent) -> None:
        tool_buffer = self._tool_for(event)
        tool_buffer.name = event.payload["tool_name"]
        call = LLMToolCall.from_payload(event.payload["tool_call"])
        tool_buffer.call = call
        tool_buffer.arguments = call.arguments
        self._append_event("tool_started", dict(event.payload), event)

    def _record_tool_result(self, event: SessionEvent) -> None:
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
            return
        self._append_event(
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
                "status": "failed" if event.kind == "tool_failed" else "completed",
            },
            event,
        )
        self._tools.pop(event.payload["tool_call_id"], None)

    def _tool_for(self, event: SessionEvent) -> _ToolBuffer:
        if "tool_call_id" not in event.payload:
            raise ValueError("tool event must include tool_call_id")
        call_id = event.payload["tool_call_id"]
        tool_buffer = self._tools.get(call_id)
        if tool_buffer is None:
            tool_buffer = _ToolBuffer()
            self._tools[call_id] = tool_buffer
        return tool_buffer

    def _append_event(
        self,
        kind: SessionEventKind,
        payload: dict[str, Any],
        source_event: SessionEvent | None,
    ) -> None:
        self.store.append_event(
            self.session_id,
            SessionEvent(
                session_id=self.session_id,
                run_id="" if source_event is None else source_event.run_id,
                step_id=None if source_event is None else source_event.step_id,
                step_index=None if source_event is None else source_event.step_index,
                kind=kind,
                payload=payload,
            ),
        )


def default_session_root() -> Path:
    return Path.home() / ".aceai" / "sessions"


def _event_to_export_lines(event: SessionEvent) -> list[str]:
    if event.kind == "user_message":
        return ["## user", event.payload["content"]]
    if event.kind == "assistant_message":
        return ["## assistant", event.payload["content"]]
    if event.kind == "assistant_tool_call":
        lines = ["## assistant tool calls"]
        if event.payload["content"] != "":
            lines.append(event.payload["content"])
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
    if event.kind == "error":
        status = event.payload["status"]
        return [f"## error ({status})", event.payload["content"]]
    return []


def _usage_payload(usage: LLMUsage | None) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "usage": {
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
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
    if '"exit_code":0' in output:
        return "completed - command exited 0"
    if '"exit_code":' in output:
        return "completed - command finished"
    if '"matches":' in output:
        return "completed - search finished"
    return "completed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
