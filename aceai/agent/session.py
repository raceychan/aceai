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

from aceai.agent.cost import CostEstimate
from aceai.core.models import ToolExecutionResult
from aceai.core.helpers.string import uuid_str
from aceai.llm.models import LLMMessage, LLMToolCall, LLMUsage

SessionMessageKind = Literal["user", "assistant", "tool", "error"]
SessionEventKind = Literal[
    "session_notice",
    "assistant_delta",
    "tool_call_delta",
    "tool_started",
    "tool_output",
    "llm_completed",
    "user_message",
    "tool_completed",
    "tool_failed",
    "run_failed",
    "step_failed",
]


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
    usage_input_tokens: int | None = None
    usage_cached_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    usage_total_tokens: int | None = None
    cost_model: str | None = None
    cost_input_usd: float | None = None
    cost_cached_input_usd: float | None = None
    cost_output_usd: float | None = None
    cost_total_usd: float | None = None
    cost_input_usd_per_million: float | None = None
    cost_cached_input_usd_per_million: float | None = None
    cost_output_usd_per_million: float | None = None
    cost_pricing_source: str | None = None


class SessionEvent(Struct, frozen=True, kw_only=True):
    """Data-layer event consumed by SessionRecorder."""

    kind: SessionEventKind
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_call: LLMToolCall | None = None
    tool_result: ToolExecutionResult | None = None
    error: str | None = None
    usage: LLMUsage | None = None
    cost: CostEstimate | None = None


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
        messages = self.load_messages(session_id)
        title_source = "Empty session"
        for message in messages:
            if message.kind == "user" and message.content != "":
                title_source = message.content[:40]
                break
        self.update_session_title(session_id, title_source)
        return title_source

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

    def export_text(self, session_id: str) -> str:
        metadata = self.get_session(session_id)
        messages = self.load_messages(session_id)
        return messages_to_export_text(metadata, messages)

    def total_cost_usd(self) -> float:
        total = 0.0
        for session in self.list_sessions():
            for message in self.load_messages(session.session_id):
                if message.cost_total_usd is not None:
                    total += message.cost_total_usd
        return total

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
        self._finalized = False
        self._saved = True

    @property
    def saved(self) -> bool:
        return self._saved

    def record(self, event: SessionEvent) -> None:
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
        if event.kind == "llm_completed":
            if self._assistant_buffer != "":
                self.flush_assistant(event.usage, event.cost)
            elif event.content != "":
                self._append_assistant_message(event.content, event.usage, event.cost)
            return

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

    def flush_assistant(
        self,
        usage: LLMUsage | None = None,
        cost: CostEstimate | None = None,
    ) -> None:
        if self._assistant_buffer == "":
            return
        self._append_assistant_message(self._assistant_buffer, usage, cost)
        self._assistant_buffer = ""

    def _append_assistant_message(
        self,
        content: str,
        usage: LLMUsage | None,
        cost: CostEstimate | None,
    ) -> None:
        self.store.append_message(
            self.session_id,
            SessionMessage(
                kind="assistant",
                content=content,
                created_at=_utc_now().isoformat(),
                usage_input_tokens=None if usage is None else usage.input_tokens,
                usage_cached_input_tokens=(
                    None if usage is None else usage.cached_input_tokens
                ),
                usage_output_tokens=None if usage is None else usage.output_tokens,
                usage_total_tokens=None if usage is None else usage.total_tokens,
                cost_model=None if cost is None else cost.model,
                cost_input_usd=None if cost is None else cost.input_cost_usd,
                cost_cached_input_usd=(
                    None if cost is None else cost.cached_input_cost_usd
                ),
                cost_output_usd=None if cost is None else cost.output_cost_usd,
                cost_total_usd=None if cost is None else cost.total_cost_usd,
                cost_input_usd_per_million=(
                    None if cost is None else cost.input_usd_per_million
                ),
                cost_cached_input_usd_per_million=(
                    None if cost is None else cost.cached_input_usd_per_million
                ),
                cost_output_usd_per_million=(
                    None if cost is None else cost.output_usd_per_million
                ),
                cost_pricing_source=None if cost is None else cost.pricing_source,
            ),
        )

    def finalize(self) -> bool:
        if self._finalized:
            return self._saved
        self.flush_assistant()
        if not self.store.load_messages(self.session_id):
            self.store.delete_session(self.session_id)
            self._saved = False
            self._finalized = True
            return self._saved
        self.store.finalize_session_title(self.session_id)
        self._saved = True
        self._finalized = True
        return self._saved

    def _record_tool(self, event: SessionEvent) -> None:
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
                content=_tool_content(
                    event.kind,
                    event.error or event.content,
                    tool_buffer.output,
                ),
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

    def _tool_for(self, event: SessionEvent) -> _ToolBuffer:
        if event.tool_call_id is None:
            raise ValueError("tool event must include tool_call_id")
        tool_buffer = self._tools.get(event.tool_call_id)
        if tool_buffer is None:
            tool_buffer = _ToolBuffer()
            self._tools[event.tool_call_id] = tool_buffer
        return tool_buffer


def default_session_root() -> Path:
    return Path.home() / ".aceai" / "sessions"


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
        "usage_input_tokens": message.usage_input_tokens,
        "usage_cached_input_tokens": message.usage_cached_input_tokens,
        "usage_output_tokens": message.usage_output_tokens,
        "usage_total_tokens": message.usage_total_tokens,
        "cost_model": message.cost_model,
        "cost_input_usd": message.cost_input_usd,
        "cost_cached_input_usd": message.cost_cached_input_usd,
        "cost_output_usd": message.cost_output_usd,
        "cost_total_usd": message.cost_total_usd,
        "cost_input_usd_per_million": message.cost_input_usd_per_million,
        "cost_cached_input_usd_per_million": (
            message.cost_cached_input_usd_per_million
        ),
        "cost_output_usd_per_million": message.cost_output_usd_per_million,
        "cost_pricing_source": message.cost_pricing_source,
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
        usage_input_tokens=payload.get("usage_input_tokens"),
        usage_cached_input_tokens=payload.get("usage_cached_input_tokens"),
        usage_output_tokens=payload.get("usage_output_tokens"),
        usage_total_tokens=payload.get("usage_total_tokens"),
        cost_model=payload.get("cost_model"),
        cost_input_usd=payload.get("cost_input_usd"),
        cost_cached_input_usd=payload.get("cost_cached_input_usd"),
        cost_output_usd=payload.get("cost_output_usd"),
        cost_total_usd=payload.get("cost_total_usd"),
        cost_input_usd_per_million=payload.get("cost_input_usd_per_million"),
        cost_cached_input_usd_per_million=payload.get(
            "cost_cached_input_usd_per_million"
        ),
        cost_output_usd_per_million=payload.get("cost_output_usd_per_million"),
        cost_pricing_source=payload.get("cost_pricing_source"),
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
