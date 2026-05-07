import json
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import insert as sql_insert

from aceai.agent.cost import estimate_usage_cost
from aceai.agent.session import (
    EventLog,
    SessionEvent,
    SessionMetadata,
    SessionRecorder,
    SessionState,
    SessionStore,
)
from aceai.agent.event_store import JsonlEventStore
from aceai.agent.project import ProjectStore
from aceai.core.models import ToolExecutionResult
from aceai.llm.models import (
    LLMMessage,
    LLMToolCall,
    LLMUsage,
)


def test_session_store_creates_sqlite_index_and_message_file(tmp_path) -> None:
    store = SessionStore(tmp_path)

    metadata = store.create_session()

    assert metadata.session_id != ""
    assert type(metadata.created_at) is datetime
    assert type(metadata.updated_at) is datetime
    assert (tmp_path / "sessions.sqlite3").exists()
    assert (tmp_path / "files" / f"{metadata.session_id}.events.jsonl").exists()
    assert store.get_session(metadata.session_id).session_id == metadata.session_id


def test_session_store_lists_sessions_by_recent_update(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()

    store.update_session_title(first.session_id, "first")

    sessions = store.list_sessions()

    assert [session.session_id for session in sessions] == [
        first.session_id,
        second.session_id,
    ]


def test_session_store_lists_all_sessions_with_current_project_first(tmp_path) -> None:
    project_store = ProjectStore(tmp_path / "projects")
    ioa = project_store.resolve_project(tmp_path / "ioa")
    travel_butler = project_store.resolve_project(tmp_path / "travel_butler")
    ioa_store = SessionStore(tmp_path / "sessions", project=ioa)
    travel_store = SessionStore(tmp_path / "sessions", project=travel_butler)

    ioa_session = ioa_store.create_session()
    travel_session = travel_store.create_session()

    sessions = ioa_store.list_sessions()

    assert [session.session_id for session in sessions] == [
        ioa_session.session_id,
        travel_session.session_id,
    ]
    assert sessions[0].project_id == ioa.project_id
    assert sessions[0].project_name == "ioa"


def test_session_store_can_filter_sessions_by_project(tmp_path) -> None:
    project_store = ProjectStore(tmp_path / "projects")
    ioa = project_store.resolve_project(tmp_path / "ioa")
    travel_butler = project_store.resolve_project(tmp_path / "travel_butler")
    ioa_store = SessionStore(tmp_path / "sessions", project=ioa)
    travel_store = SessionStore(tmp_path / "sessions", project=travel_butler)

    ioa_session = ioa_store.create_session()
    travel_store.create_session()

    sessions = ioa_store.list_sessions(project_id=ioa.project_id)

    assert [session.session_id for session in sessions] == [ioa_session.session_id]


def test_session_store_backfills_empty_project_to_current_project(tmp_path) -> None:
    root = tmp_path / "sessions"
    files_dir = root / "files"
    files_dir.mkdir(parents=True)
    engine = create_engine(f"sqlite:///{root / 'sessions.sqlite3'}")
    metadata = MetaData()
    sessions_table = Table(
        "sessions",
        metadata,
        Column("session_id", String, primary_key=True),
        Column("project_id", String, nullable=False),
        Column("project_name", String, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("title", String, nullable=False),
        Column("path", String, nullable=False),
        Column("state_json", String, nullable=False),
    )
    metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            sql_insert(sessions_table).values(
                session_id="session-1",
                project_id="",
                project_name="",
                created_at=now,
                updated_at=now,
                title="old session",
                path="session-1.events.jsonl",
                state_json=json.dumps(SessionState.empty().as_json()),
            )
        )
    (files_dir / "session-1.events.jsonl").write_text("", encoding="utf-8")

    store = SessionStore(root)
    session = store.get_session("session-1")

    assert session.project_id == store.project_id
    assert session.project_name == "aceai"


def test_session_store_deletes_session_index_and_file(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    path = tmp_path / "files" / f"{metadata.session_id}.events.jsonl"

    store.delete_session(metadata.session_id)

    assert store.list_sessions() == []
    assert not path.exists()


def test_jsonl_event_store_round_trips_session_events(tmp_path) -> None:
    event_store = JsonlEventStore(tmp_path)
    session_id = "session-1"
    path = event_store.create_event_log(session_id)
    metadata = SessionMetadata(
        session_id=session_id,
        project_id="project-1",
        project_name="test-project",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        title="Test",
        path=str(tmp_path / path),
    )

    event_store.append_event(metadata, _user_message("hello"))
    event_log = event_store.load_event_log(metadata)

    assert (tmp_path / path).exists()
    assert [event.kind for event in event_log.events] == ["user_message"]
    assert event_log.events[0].session_id == session_id
    assert event_log.events[0].payload["content"] == "hello"

    event_store.delete_event_log(metadata)

    assert not (tmp_path / path).exists()


def test_session_store_delegates_event_log_operations_to_event_store(tmp_path) -> None:
    event_store = StubEventStore()
    store = SessionStore(tmp_path, event_store=event_store)

    metadata = store.create_session()
    store.append_event(metadata.session_id, _user_message("hello"))
    event_log = store.load_event_log(metadata.session_id)
    store.delete_session(metadata.session_id)

    assert event_store.created == [metadata.session_id]
    assert event_store.appended == ["hello"]
    assert event_log.events[0].payload["content"] == "hello"
    assert event_store.deleted == [metadata.session_id]


def test_session_store_persists_session_state(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()

    store.update_session_state(
        metadata.session_id,
        SessionState(
            selected_provider="deepseek",
            selected_model="deepseek-v4-pro",
        ),
    )

    state = store.get_session_state(metadata.session_id)

    assert state == SessionState(
        selected_provider="deepseek",
        selected_model="deepseek-v4-pro",
    )


def test_event_log_returns_full_run_by_run_id() -> None:
    run_id = "run-1"
    events = [
        SessionEvent(
            run_id=run_id,
            kind="user_message",
            payload={"content": "Question?"},
        ),
        SessionEvent(
            run_id=run_id,
            step_id="step-1",
            step_index=0,
            kind="assistant_tool_call",
            payload={
                "content": "checking",
                "tool_calls": [
                    LLMToolCall(
                        name="list_directory",
                        arguments='{"path":"."}',
                        call_id="call-1",
                    ).asdict()
                ],
            },
        ),
        SessionEvent(
            run_id=run_id,
            step_id="step-1",
            step_index=0,
            kind="tool_result",
            payload={
                "content": "completed - 1 entry",
                "tool_name": "list_directory",
                "tool_call_id": "call-1",
                "tool_arguments": '{"path":"."}',
                "output": '{"entries":["aceai"]}',
                "status": "completed",
            },
        ),
        SessionEvent(
            run_id=run_id,
            step_id="step-2",
            step_index=1,
            kind="run_completed",
            payload={"content": "Answer."},
        ),
    ]
    event_log = EventLog(events)

    run_log = event_log.get_run(run_id)
    summaries = event_log.list_runs()

    assert run_log.question == "Question?"
    assert run_log.status == "completed"
    assert run_log.final_answer == "Answer."
    assert [event.kind for event in run_log.events] == [
        "user_message",
        "assistant_tool_call",
        "tool_result",
        "run_completed",
    ]
    assert summaries[0].run_id == run_id
    assert summaries[0].step_count == 2
    assert summaries[0].tool_call_count == 1


def test_session_store_finalize_uses_first_question_as_title(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(_user_message("What files are here?"))

    title = store.finalize_session_title(metadata.session_id)

    assert title == "What files are here?"
    assert store.get_session(metadata.session_id).title == title


def test_session_recorder_finalize_deletes_empty_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)

    saved = recorder.finalize()

    assert saved is False
    assert recorder.saved is False
    assert store.list_sessions() == []
    assert not (tmp_path / "files" / f"{metadata.session_id}.events.jsonl").exists()


def test_session_recorder_finalize_keeps_non_empty_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(_user_message("hello"))

    saved = recorder.finalize()

    assert saved is True
    assert recorder.saved is True
    assert store.get_session(metadata.session_id).title == "hello"


def test_session_recorder_merges_streaming_assistant_deltas(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)

    recorder.record(_user_message("hello"))
    recorder.record(_assistant_delta("hel"))
    recorder.record(_assistant_delta("lo"))
    recorder.record(_llm_completed("hello"))

    events = store.load_event_log(metadata.session_id).events

    assert [event.kind for event in events] == ["user_message", "assistant_message"]
    assert events[1].payload["content"] == "hello"


def test_session_recorder_persists_reasoning_events(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)

    recorder.record(_user_message("hello"))
    recorder.record(_thinking_delta("checking code"))
    recorder.record(_reasoning_summary("Found the session replay path."))
    recorder.record(_llm_completed("answer"))

    events = store.load_event_log(metadata.session_id).events

    assert [event.kind for event in events] == [
        "user_message",
        "thinking_delta",
        "reasoning_summary",
        "assistant_message",
    ]
    assert events[1].payload["content"] == "checking code"
    assert events[2].payload["content"] == "Found the session replay path."


def test_session_recorder_persists_assistant_usage(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    usage = LLMUsage(
        input_tokens=12,
        cached_input_tokens=8,
        output_tokens=5,
        total_tokens=17,
    )

    recorder.record(_llm_completed("answer", model="gpt-5.5", usage=usage))

    events = store.load_event_log(metadata.session_id).events

    assert events[0].payload["usage"]["input_tokens"] == 12
    assert events[0].payload["usage"]["cached_input_tokens"] == 8
    assert events[0].payload["usage"]["cache_miss_input_tokens"] is None
    assert events[0].payload["usage"]["input_cache_hit_rate"] is None
    assert events[0].payload["usage"]["output_tokens"] == 5
    assert events[0].payload["usage"]["total_tokens"] == 17
    assert events[0].payload["cost"]["model"] == "gpt-5.5"
    assert events[0].payload["cost"]["total_cost_usd"] is not None
    assert events[0].payload["cost"]["cached_input_cost_usd"] is not None
    assert round(events[0].payload["cost"]["cached_input_cost_usd"], 6) == 0.000004
    assert round(events[0].payload["cost"]["total_cost_usd"], 6) == 0.000174


def test_session_store_sums_total_cost_across_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()

    SessionRecorder(store, first.session_id).record(
        _llm_completed(
            "first",
            model="gpt-5.5",
            usage=LLMUsage(
                input_tokens=1_000,
                cached_input_tokens=600,
                output_tokens=100,
                total_tokens=1_100,
            ),
        )
    )
    SessionRecorder(store, second.session_id).record(
        _llm_completed(
            "second",
            model="gpt-5.4-mini",
            usage=LLMUsage(
                input_tokens=1_000,
                output_tokens=100,
                total_tokens=1_100,
            ),
        )
    )

    assert round(store.total_cost_usd(), 6) == 0.0065


def test_session_recorder_saves_non_streaming_llm_completion(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)

    recorder.record(_llm_completed("answer"))

    events = store.load_event_log(metadata.session_id).events

    assert len(events) == 1
    assert events[0].kind == "assistant_message"
    assert events[0].payload["content"] == "answer"


def test_session_recorder_merges_tool_deltas_into_one_message(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    call = LLMToolCall(
        name="list_directory",
        arguments='{"path":"."}',
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"path":".","entries":['
            '{"name":"aceai","path":"aceai","kind":"directory"},'
            '{"name":"tests","path":"tests","kind":"directory"}'
            "]}"
        ),
    )

    recorder.record(_tool_call_delta("call-1", '{"path"'))
    recorder.record(_tool_call_delta("call-1", ':"."}'))
    recorder.record(_tool_started(call))
    recorder.record(_tool_completed(call, result))

    events = store.load_event_log(metadata.session_id).events

    assert [event.kind for event in events] == ["tool_started", "tool_result"]
    assert events[1].payload["tool_name"] == "list_directory"
    assert events[1].payload["tool_arguments"] == '{"path":"."}'
    assert events[1].payload["content"] == "completed - 2 entries"


def test_event_log_restores_tool_messages_in_llm_history(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(_user_message("hello"))
    call = LLMToolCall(
        name="list_directory",
        arguments='{"path":"."}',
        call_id="call-1",
    )
    result = ToolExecutionResult(call=call, output='{"entries":[]}')
    recorder.record(_llm_completed("", tool_calls=[call]))
    recorder.record(_tool_started(call))
    recorder.record(_tool_completed(call, result))

    history = store.load_event_log(metadata.session_id).replay_llm_history()

    assert history[0] == LLMMessage.build(role="user", content="hello")
    assert history[1].role == "assistant"
    assert history[1].tool_calls == [call]
    assert history[2].role == "tool"
    assert history[2].call_id == "call-1"
    assert history[2].content[0]["data"] == '{"entries":[]}'


def test_tool_call_assistant_content_is_not_recorded_or_replayed(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    call = LLMToolCall(
        name="list_directory",
        arguments='{"path":"."}',
        call_id="call-1",
    )
    result = ToolExecutionResult(call=call, output='{"entries":[]}')

    recorder.record(_user_message("inspect"))
    recorder.record(_assistant_delta("Need to inspect files."))
    recorder.record(_llm_completed("Need to inspect files.", tool_calls=[call]))
    recorder.record(_tool_started(call))
    recorder.record(_tool_completed(call, result))

    event_log = store.load_event_log(metadata.session_id)
    assistant_tool_call = [
        event for event in event_log.events if event.kind == "assistant_tool_call"
    ][0]
    history = event_log.replay_llm_history()
    export_text = store.export_text(metadata.session_id)

    assert assistant_tool_call.payload["content"] == ""
    assert history[1].role == "assistant"
    assert history[1].content == []
    assert "Need to inspect files" not in export_text


def test_event_log_omits_pending_tool_call_from_llm_history(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"python binary_search.py"}',
        call_id="call-1",
    )

    recorder.record(_user_message("run it"))
    recorder.record(_llm_completed("running it", tool_calls=[call]))
    recorder.record(_tool_started(call))
    recorder.record(
        SessionEvent(
            kind="tool_approval_requested",
            payload={
                "content": "Tool 'run_shell_command' requires approval (shell_command)",
                "tool_name": call.name,
                "tool_call_id": call.call_id,
                "tool_call": call.asdict(),
            },
        )
    )
    recorder.record(_user_message("send approval again"))

    history = store.load_event_log(metadata.session_id).replay_llm_history()

    assert [message.role for message in history] == ["user", "user"]
    assert history[0].content[0]["data"] == "run it"
    assert history[1].content[0]["data"] == "send approval again"


def test_session_store_exports_readable_text(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(_user_message("hello"))
    recorder.record(_assistant_delta("answer"))
    recorder.flush_assistant()

    text = store.export_text(metadata.session_id)

    assert text.startswith(f"# AceAI session {metadata.session_id}\n")
    assert "## user\nhello\n" in text
    assert "## assistant\nanswer\n" in text


def test_session_recorder_exports_tool_approval_events(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"x","content":"hello"}',
        call_id="call-1",
    )

    recorder.record(_tool_started(call))
    recorder.record(
        SessionEvent(
            kind="tool_approval_requested",
            payload={
                "content": "Tool 'write_text_file' requires approval (filesystem_write)",
                "tool_name": call.name,
                "tool_call_id": call.call_id,
                "tool_call": call.asdict(),
            },
        )
    )
    recorder.record(
        SessionEvent(
            kind="tool_approval_resolved",
            payload={
                "content": "approved",
                "tool_name": call.name,
                "tool_call_id": call.call_id,
                "tool_call": call.asdict(),
            },
        )
    )

    text = store.export_text(metadata.session_id)

    assert "## tool approval requested: write_text_file\n" in text
    assert "filesystem_write" in text
    assert "arguments:\n{\"path\":\"x\",\"content\":\"hello\"}" in text
    assert "## tool approval resolved: write_text_file\napproved" in text


class StubEventStore:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.appended: list[str] = []
        self.events: list[SessionEvent] = []
        self.deleted: list[str] = []

    def create_event_log(self, session_id: str) -> str:
        self.created.append(session_id)
        return f"{session_id}.events.jsonl"

    def append_event(self, metadata: SessionMetadata, event: SessionEvent) -> None:
        persisted_event = event.with_session_defaults(session_id=metadata.session_id)
        self.appended.append(persisted_event.payload["content"])
        self.events.append(persisted_event)

    def load_event_log(self, metadata: SessionMetadata) -> EventLog:
        return EventLog(list(self.events))

    def delete_event_log(self, metadata: SessionMetadata) -> None:
        self.deleted.append(metadata.session_id)


def _user_message(content: str) -> SessionEvent:
    return SessionEvent(kind="user_message", payload={"content": content})


def _assistant_delta(content: str) -> SessionEvent:
    return SessionEvent(kind="assistant_delta", payload={"content": content})


def _thinking_delta(content: str) -> SessionEvent:
    return SessionEvent(kind="thinking_delta", payload={"content": content})


def _reasoning_summary(content: str) -> SessionEvent:
    return SessionEvent(kind="reasoning_summary", payload={"content": content})


def _llm_completed(
    content: str,
    *,
    model: str | None = None,
    usage: LLMUsage | None = None,
    tool_calls: list[LLMToolCall] | None = None,
) -> SessionEvent:
    payload = {
        "content": content,
        "tool_calls": [] if tool_calls is None else [call.asdict() for call in tool_calls],
    }
    if usage is not None:
        payload["usage"] = {
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_miss_input_tokens": usage.cache_miss_input_tokens,
            "input_cache_hit_rate": usage.input_cache_hit_rate,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
    cost = estimate_usage_cost(model, usage)
    if cost is not None:
        payload["cost"] = cost.asdict()
    return SessionEvent(
        kind="llm_completed",
        payload=payload,
    )


def _tool_call_delta(call_id: str, content: str) -> SessionEvent:
    return SessionEvent(
        kind="tool_call_delta",
        payload={"content": content, "tool_call_id": call_id},
    )


def _tool_started(call: LLMToolCall) -> SessionEvent:
    return SessionEvent(
        kind="tool_started",
        payload={
            "content": "",
            "tool_name": call.name,
            "tool_call_id": call.call_id,
            "tool_call": call.asdict(),
        },
    )


def _tool_completed(
    call: LLMToolCall,
    result: ToolExecutionResult,
) -> SessionEvent:
    return SessionEvent(
        kind="tool_completed",
        payload={
            "content": result.output,
            "tool_name": call.name,
            "tool_call_id": call.call_id,
            "tool_call": call.asdict(),
            "tool_result": {
                "output": result.output,
                "error": result.error,
            },
        },
    )
