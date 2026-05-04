from datetime import datetime

from aceai.agent.cost import estimate_usage_cost
from aceai.agent.session import (
    SessionEvent,
    SessionRecorder,
    SessionStore,
    messages_to_llm_history,
)
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


def test_session_store_deletes_session_index_and_file(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    path = tmp_path / "files" / f"{metadata.session_id}.jsonl"

    store.delete_session(metadata.session_id)

    assert store.list_sessions() == []
    assert not path.exists()


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
    assert not (tmp_path / "files" / f"{metadata.session_id}.jsonl").exists()


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

    messages = store.load_messages(metadata.session_id)

    assert [message.kind for message in messages] == ["user", "assistant"]
    assert messages[1].content == "hello"


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

    messages = store.load_messages(metadata.session_id)

    assert messages[0].usage_input_tokens == 12
    assert messages[0].usage_cached_input_tokens == 8
    assert messages[0].usage_output_tokens == 5
    assert messages[0].usage_total_tokens == 17
    assert messages[0].cost_model == "gpt-5.5"
    assert messages[0].cost_total_usd is not None
    assert messages[0].cost_cached_input_usd is not None
    assert round(messages[0].cost_cached_input_usd, 6) == 0.000004
    assert round(messages[0].cost_total_usd, 6) == 0.000174


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

    messages = store.load_messages(metadata.session_id)

    assert len(messages) == 1
    assert messages[0].kind == "assistant"
    assert messages[0].content == "answer"


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

    messages = store.load_messages(metadata.session_id)

    assert len(messages) == 1
    assert messages[0].kind == "tool"
    assert messages[0].tool_name == "list_directory"
    assert messages[0].tool_arguments == '{"path":"."}'
    assert messages[0].content == "completed - 2 entries"


def test_session_messages_restore_user_assistant_llm_history_only(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(_user_message("hello"))
    recorder.record(_assistant_delta("answer"))
    recorder.flush_assistant()

    history = messages_to_llm_history(store.load_messages(metadata.session_id))

    assert history == [
        LLMMessage.build(role="user", content="hello"),
        LLMMessage.build(role="assistant", content="answer"),
    ]


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


def _user_message(content: str) -> SessionEvent:
    return SessionEvent(kind="user_message", content=content)


def _assistant_delta(content: str) -> SessionEvent:
    return SessionEvent(kind="assistant_delta", content=content)


def _llm_completed(
    content: str,
    *,
    model: str | None = None,
    usage: LLMUsage | None = None,
) -> SessionEvent:
    return SessionEvent(
        kind="llm_completed",
        content=content,
        usage=usage,
        cost=estimate_usage_cost(model, usage),
    )


def _tool_call_delta(call_id: str, content: str) -> SessionEvent:
    return SessionEvent(
        kind="tool_call_delta",
        content=content,
        tool_call_id=call_id,
    )


def _tool_started(call: LLMToolCall) -> SessionEvent:
    return SessionEvent(
        kind="tool_started",
        tool_name=call.name,
        tool_call_id=call.call_id,
        tool_call=call,
    )


def _tool_completed(
    call: LLMToolCall,
    result: ToolExecutionResult,
) -> SessionEvent:
    return SessionEvent(
        kind="tool_completed",
        content=result.output,
        tool_name=call.name,
        tool_call_id=call.call_id,
        tool_call=call,
        tool_result=result,
    )
