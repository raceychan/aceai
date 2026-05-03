import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from msgspec import Struct
from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import insert as sql_insert
from sqlalchemy import select as sql_select
from sqlalchemy import update as sql_update
from sqlalchemy.engine import RowMapping

from aceai.core.models import ToolExecutionResult
from aceai.core.helpers.string import uuid_str
from aceai.llm.models import LLMMessage, LLMToolCall

from .tui.events import TUIEvent, user_message_event

SessionMessageKind = Literal["user", "assistant", "tool", "error"]


class SessionMetadata(Struct, frozen=True, kw_only=True):
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: str
    path: str


class SessionMessage(Struct, frozen=True, kw_only=True):
    kind: SessionMessageKind
    content: str
    created_at: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: str = ""
    tool_output: str = ""
    status: str = ""


class _ToolBuffer(Struct, kw_only=True):
    name: str | None = None
    arguments: str = ""
    output: str = ""


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
        path = self.files_dir / f"{session_id}.jsonl"
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
        return self._metadata_from_row(row)

    def list_sessions(self) -> list[SessionMetadata]:
        query = sql_select(_sessions_table).order_by(
            _sessions_table.c.updated_at.desc()
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        return [self._metadata_from_row(row) for row in rows]

    def update_session_title(self, session_id: str, title: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(title=title, updated_at=_utc_now())
            )

    def finalize_session_title(self, session_id: str) -> str:
        messages = self.load_messages(session_id)
        title_source = "Empty session"
        for message in messages:
            if message.kind == "user" and message.content != "":
                title_source = message.content[:40]
                break
        title = f"{title_source} - {_local_second()}"
        self.update_session_title(session_id, title)
        return title

    def append_message(self, session_id: str, message: SessionMessage) -> None:
        metadata = self.get_session(session_id)
        with Path(metadata.path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_message_to_json(message), ensure_ascii=False))
            stream.write("\n")
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_sessions_table)
                .where(_sessions_table.c.session_id == session_id)
                .values(updated_at=_utc_now())
            )

    def load_messages(self, session_id: str) -> list[SessionMessage]:
        metadata = self.get_session(session_id)
        path = Path(metadata.path)
        messages: list[SessionMessage] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "":
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("Session message must be a mapping")
            messages.append(_message_from_json(payload))
        return messages

    def load_tui_events(self, session_id: str) -> list[TUIEvent]:
        return messages_to_tui_events(self.load_messages(session_id))

    def export_text(self, session_id: str) -> str:
        metadata = self.get_session(session_id)
        messages = self.load_messages(session_id)
        return messages_to_export_text(metadata, messages)

    def _init_db(self) -> None:
        _metadata.create_all(self.engine)

    def _metadata_from_row(self, row: RowMapping) -> SessionMetadata:
        return SessionMetadata(
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row["title"],
            path=str(self.files_dir / row["path"]),
        )


class SessionRecorder:
    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id
        self._assistant_buffer = ""
        self._tools: dict[str, _ToolBuffer] = {}

    def record(self, event: TUIEvent) -> None:
        if event.kind == "session_notice":
            return
        if event.kind == "assistant_delta":
            self._assistant_buffer += event.content
            return
        if event.kind == "tool_call_delta":
            self._tool_for(event).arguments += event.content
            return
        if event.kind == "tool_started":
            tool_buffer = self._tool_for(event)
            tool_buffer.name = event.tool_name
            if event.tool_call is not None:
                tool_buffer.arguments = event.tool_call.arguments
            return
        if event.kind == "tool_output":
            self._tool_for(event).output += event.content
            return

        flushed_assistant = self._assistant_buffer != ""
        self.flush_assistant()
        if event.kind == "user_message":
            self.store.append_message(
                self.session_id,
                SessionMessage(
                    kind="user",
                    content=event.content,
                    created_at=_utc_now().isoformat(),
                ),
            )
        elif event.kind == "llm_completed" and event.content != "" and not flushed_assistant:
            self.store.append_message(
                self.session_id,
                SessionMessage(
                    kind="assistant",
                    content=event.content,
                    created_at=_utc_now().isoformat(),
                ),
            )
        elif event.kind in ("tool_completed", "tool_failed"):
            self._record_tool(event)
        elif event.kind in ("run_failed", "step_failed"):
            self.store.append_message(
                self.session_id,
                SessionMessage(
                    kind="error",
                    content=event.error or event.content,
                    created_at=_utc_now().isoformat(),
                    status="failed",
                ),
            )

    def flush_assistant(self) -> None:
        if self._assistant_buffer == "":
            return
        self.store.append_message(
            self.session_id,
            SessionMessage(
                kind="assistant",
                content=self._assistant_buffer,
                created_at=_utc_now().isoformat(),
            ),
        )
        self._assistant_buffer = ""

    def finalize(self) -> None:
        self.flush_assistant()
        self.store.finalize_session_title(self.session_id)

    def _record_tool(self, event: TUIEvent) -> None:
        tool_buffer = self._tool_for(event)
        if event.tool_name is not None:
            tool_buffer.name = event.tool_name
        if event.tool_call is not None:
            tool_buffer.arguments = event.tool_call.arguments
        if event.tool_result is not None:
            tool_buffer.output = event.tool_result.output
        elif event.content != "":
            tool_buffer.output = event.content
        if tool_buffer.name is None:
            return
        self.store.append_message(
            self.session_id,
            SessionMessage(
                kind="tool",
                content=_tool_content(event, tool_buffer.output),
                created_at=_utc_now().isoformat(),
                tool_name=tool_buffer.name,
                tool_call_id=event.tool_call_id,
                tool_arguments=tool_buffer.arguments,
                tool_output=tool_buffer.output,
                status="failed" if event.kind == "tool_failed" else "completed",
            ),
        )
        if event.tool_call_id is not None:
            self._tools.pop(event.tool_call_id, None)

    def _tool_for(self, event: TUIEvent) -> _ToolBuffer:
        if event.tool_call_id is None:
            raise ValueError("tool event must include tool_call_id")
        tool_buffer = self._tools.get(event.tool_call_id)
        if tool_buffer is None:
            tool_buffer = _ToolBuffer()
            self._tools[event.tool_call_id] = tool_buffer
        return tool_buffer


def default_session_root() -> Path:
    return Path.home() / ".aceai" / "sessions"


def messages_to_tui_events(messages: list[SessionMessage]) -> list[TUIEvent]:
    events: list[TUIEvent] = []
    for message in messages:
        if message.kind == "user":
            events.append(user_message_event(message.content))
        elif message.kind == "assistant":
            events.append(
                TUIEvent(
                    kind="assistant_delta",
                    step_index=-1,
                    step_id=uuid_str(),
                    title="assistant",
                    content=message.content,
                    raw_event=None,
                )
            )
        elif message.kind == "tool":
            call_id = message.tool_call_id or uuid_str()
            tool_name = message.tool_name or "tool"
            tool_call = LLMToolCall(
                name=tool_name,
                arguments=message.tool_arguments,
                call_id=call_id,
            )
            events.append(
                TUIEvent(
                    kind="tool_failed" if message.status == "failed" else "tool_completed",
                    step_index=-1,
                    step_id=uuid_str(),
                    title=f"tool {tool_name}",
                    content=message.tool_output,
                    tool_name=tool_name,
                    tool_call_id=call_id,
                    tool_call=tool_call,
                    tool_result=ToolExecutionResult(
                        call=tool_call,
                        output=message.tool_output,
                        error=message.content if message.status == "failed" else None,
                    ),
                    error=message.content if message.status == "failed" else None,
                    raw_event=None,
                )
            )
        elif message.kind == "error":
            events.append(
                TUIEvent(
                    kind="run_failed",
                    step_index=-1,
                    step_id=uuid_str(),
                    title="run failed",
                    content=message.content,
                    error=message.content,
                    raw_event=None,
                )
            )
    return events


def messages_to_llm_history(messages: list[SessionMessage]) -> list[LLMMessage]:
    history: list[LLMMessage] = []
    for message in messages:
        if message.kind == "user":
            history.append(LLMMessage.build(role="user", content=message.content))
        elif message.kind == "assistant":
            history.append(LLMMessage.build(role="assistant", content=message.content))
    return history


def messages_to_export_text(
    metadata: SessionMetadata, messages: list[SessionMessage]
) -> str:
    lines = [
        f"# AceAI session {metadata.session_id}",
        f"title: {metadata.title}",
        f"created_at: {metadata.created_at.isoformat()}",
        f"updated_at: {metadata.updated_at.isoformat()}",
        "",
    ]
    for message in messages:
        lines.extend(_message_to_export_lines(message))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _message_to_json(message: SessionMessage) -> dict[str, Any]:
    return {
        "kind": message.kind,
        "content": message.content,
        "created_at": message.created_at,
        "tool_name": message.tool_name,
        "tool_call_id": message.tool_call_id,
        "tool_arguments": message.tool_arguments,
        "tool_output": message.tool_output,
        "status": message.status,
    }


def _message_from_json(payload: dict[str, Any]) -> SessionMessage:
    kind = payload["kind"]
    if kind not in ("user", "assistant", "tool", "error"):
        raise ValueError("Unsupported session message kind")
    return SessionMessage(
        kind=kind,
        content=payload["content"],
        created_at=payload["created_at"],
        tool_name=payload["tool_name"],
        tool_call_id=payload["tool_call_id"],
        tool_arguments=payload["tool_arguments"],
        tool_output=payload["tool_output"],
        status=payload["status"],
    )


def _message_to_export_lines(message: SessionMessage) -> list[str]:
    if message.kind == "user":
        return ["## user", message.content]
    if message.kind == "assistant":
        return ["## assistant", message.content]
    if message.kind == "tool":
        name = message.tool_name or "tool"
        lines = [f"## tool: {name} ({message.status})"]
        if message.tool_arguments != "":
            lines.extend(["arguments:", message.tool_arguments])
        if message.tool_output != "":
            lines.extend(["output:", message.tool_output])
        return lines
    if message.kind == "error":
        status = message.status or "failed"
        return [f"## error ({status})", message.content]
    raise ValueError("Unsupported session message kind")


def _tool_content(event: TUIEvent, output: str) -> str:
    if event.kind == "tool_failed":
        return event.error or event.content
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


def _local_second() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
