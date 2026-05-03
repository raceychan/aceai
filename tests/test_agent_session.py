from datetime import datetime

from aceai.agent.session import SessionRecorder, SessionStore, messages_to_llm_history
from aceai.agent.tui.events import adapt_agent_event, user_message_event
from aceai.core.events import AgentEventBuilder
from aceai.core.models import AgentStep, ToolExecutionResult
from aceai.llm.models import LLMMessage, LLMResponse, LLMToolCall, LLMToolCallDelta


def test_session_store_creates_sqlite_index_and_message_file(tmp_path) -> None:
    store = SessionStore(tmp_path)

    metadata = store.create_session()

    assert metadata.session_id != ""
    assert type(metadata.created_at) is datetime
    assert type(metadata.updated_at) is datetime
    assert (tmp_path / "sessions.sqlite3").exists()
    assert (tmp_path / "files" / f"{metadata.session_id}.jsonl").exists()
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


def test_session_store_finalize_uses_first_question_and_second_level_date(
    tmp_path, monkeypatch
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(user_message_event("What files are here?"))

    monkeypatch.setattr("aceai.agent.session._local_second", lambda: "2026-05-04 12:13:14")

    title = store.finalize_session_title(metadata.session_id)

    assert title == "What files are here? - 2026-05-04 12:13:14"
    assert store.get_session(metadata.session_id).title == title


def test_session_recorder_merges_streaming_assistant_deltas(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)

    recorder.record(user_message_event("hello"))
    recorder.record(
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
                text_delta="hel"
            )
        )
    )
    recorder.record(
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
                text_delta="lo"
            )
        )
    )
    recorder.record(
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
                step=AgentStep(llm_response=LLMResponse(text="hello"))
            )
        )
    )

    messages = store.load_messages(metadata.session_id)

    assert [message.kind for message in messages] == ["user", "assistant"]
    assert messages[1].content == "hello"


def test_session_recorder_saves_non_streaming_llm_completion(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)

    recorder.record(
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
                step=AgentStep(llm_response=LLMResponse(text="answer"))
            )
        )
    )

    messages = store.load_messages(metadata.session_id)

    assert len(messages) == 1
    assert messages[0].kind == "assistant"
    assert messages[0].content == "answer"


def test_session_recorder_merges_tool_deltas_into_one_message(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
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

    recorder.record(
        adapt_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='{"path"',
                )
            )
        )
    )
    recorder.record(
        adapt_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta=':"."}',
                )
            )
        )
    )
    recorder.record(adapt_agent_event(builder.tool_started(tool_call=call)))
    recorder.record(
        adapt_agent_event(builder.tool_completed(tool_call=call, tool_result=result))
    )

    messages = store.load_messages(metadata.session_id)

    assert len(messages) == 1
    assert messages[0].kind == "tool"
    assert messages[0].tool_name == "list_directory"
    assert messages[0].tool_arguments == '{"path":"."}'
    assert messages[0].content == "completed - 2 entries"


def test_session_store_restores_compact_messages_as_tui_events(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(user_message_event("hello"))
    recorder.record(
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
                text_delta="answer"
            )
        )
    )
    recorder.flush_assistant()

    events = store.load_tui_events(metadata.session_id)

    assert [event.kind for event in events] == ["user_message", "assistant_delta"]
    assert events[0].content == "hello"
    assert events[1].content == "answer"


def test_session_messages_restore_user_assistant_llm_history_only(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(user_message_event("hello"))
    recorder.record(
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
                text_delta="answer"
            )
        )
    )
    recorder.flush_assistant()

    history = messages_to_llm_history(store.load_messages(metadata.session_id))

    assert history == [
        LLMMessage.build(role="user", content="hello"),
        LLMMessage.build(role="assistant", content="answer"),
    ]
