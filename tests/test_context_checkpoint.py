import json

import pytest

from aceai.agent.app import AceAgentApp
from aceai.agent.memory.context_checkpoint_store import (
    ContextCheckpoint,
    ContextCheckpointStore,
    context_units_payload_from_messages,
    llm_message_from_payload,
    llm_message_to_payload,
)
from aceai.agent.memory.context_history import build_context_history
from aceai.agent.session import EventLog, SessionEvent, SessionStore
from aceai.core.agent import Agent
from aceai.llm import LLMResponse
from aceai.llm.models import (
    LLMMessage,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
)

from tests.test_agent_behavior import (
    CompressingLLMService,
    StubExecutor,
    make_stream,
)


def test_context_checkpoint_store_round_trips_latest_checkpoint(tmp_path) -> None:
    store = ContextCheckpointStore(tmp_path / "checkpoints")
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-1")
    history = [
        LLMMessage.build(role="user", content="hello"),
        LLMToolCallMessage.from_content(content=[], tool_calls=[call]),
        LLMToolUseMessage.from_content(
            content='{"ok":true}',
            name="lookup",
            call_id="call-1",
        ),
    ]

    first = store.record_checkpoint(
        session_id="session-1",
        run_id="run-1",
        step_id="step-1",
        reason="threshold",
        compression_count=1,
        included_event_id="event-1",
        history=[LLMMessage.build(role="user", content="older")],
    )
    second = store.record_checkpoint(
        session_id="session-1",
        run_id="run-2",
        step_id="step-2",
        reason="context_window_retry",
        compression_count=2,
        included_event_id="event-2",
        history=history,
    )

    latest = store.latest_checkpoint("session-1")

    assert latest is not None
    assert latest.checkpoint_id == second.checkpoint_id
    assert latest.checkpoint_id != first.checkpoint_id
    assert latest.reason == "context_window_retry"
    assert latest.included_event_id == "event-2"
    assert latest.history[0] == history[0]
    assert isinstance(latest.history[1], LLMToolCallMessage)
    assert latest.history[1].tool_calls == [call]
    assert isinstance(latest.history[2], LLMToolUseMessage)
    assert latest.history[2].call_id == "call-1"


def test_llm_message_checkpoint_payload_rejects_unknown_message_type() -> None:
    payload = llm_message_to_payload(LLMMessage.build(role="user", content="hello"))
    payload["message_type"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported context checkpoint message_type"):
        llm_message_from_payload(payload)


def test_context_history_uses_checkpoint_without_changing_transcript_replay() -> None:
    event_log = EventLog(
        [
            _session_event("event-1", "user_message", {"content": "old question"}),
            _session_event(
                "event-2",
                "assistant_message",
                {"content": "old answer"},
            ),
            _session_event("event-3", "user_message", {"content": "new question"}),
            _session_event(
                "event-4",
                "assistant_message",
                {"content": "new answer"},
            ),
        ]
    )
    checkpoint_history = [
        LLMMessage.build(
            role="system",
            content=(
                '<aceai_context_summary scope="prior_runs">\n'
                "summary of old context\n"
                "</aceai_context_summary>"
            ),
        ),
        LLMMessage.build(role="user", content="new question"),
    ]
    checkpoint = ContextCheckpoint(
        checkpoint_id="checkpoint-1",
        session_id="session-1",
        run_id="run-1",
        step_id="step-1",
        reason="threshold",
        compression_count=1,
        included_event_id="event-3",
        message_count=2,
        estimated_tokens=10,
        history=checkpoint_history,
        units=context_units_payload_from_messages(checkpoint_history),
    )

    transcript_history = event_log.replay_llm_history()
    context_history = build_context_history(
        event_log=event_log,
        checkpoint=checkpoint,
    )

    assert [message.content[0]["data"] for message in transcript_history] == [
        "old question",
        "old answer",
        "new question",
        "new answer",
    ]
    assert [message.content[0]["data"] for message in context_history] == [
        (
            '<aceai_context_summary scope="prior_runs">\n'
            "summary of old context\n"
            "</aceai_context_summary>"
        ),
        "new question",
        "new answer",
    ]


def test_context_checkpoint_store_rejects_malformed_payload(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    path = checkpoint_dir / "session-1.checkpoints.jsonl"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "checkpoint_id": "checkpoint-1",
                "session_id": "session-1",
                "run_id": "run-1",
                "step_id": "step-1",
                "reason": "threshold",
                "compression_count": 1,
                "included_event_id": "event-1",
                "message_count": 1,
                "estimated_tokens": 1,
                "units": {"bad": "shape"},
            }
        ),
        encoding="utf-8",
    )
    store = ContextCheckpointStore(checkpoint_dir)

    with pytest.raises(TypeError, match="context checkpoint units must be list"):
        store.latest_checkpoint("session-1")


@pytest.mark.anyio
async def test_agent_app_persists_checkpoint_and_restores_context_history(
    tmp_path,
) -> None:
    session_store = SessionStore(tmp_path / "sessions")
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold=1,
    )
    initial_history = [
        LLMMessage.build(role="user", content=f"history message {index}")
        for index in range(10)
    ]
    app = AceAgentApp(
        agent,
        provider_name="openai",
        selected_model="gpt-4o",
        initial_history=initial_history,
        session_store=session_store,
    )

    [event async for event in app.start_turn("new question")]

    session_id = app.session_id
    assert session_id is not None
    checkpoint = app.session_service.context_checkpoint_store.latest_checkpoint(
        session_id
    )
    assert checkpoint is not None
    assert checkpoint.reason == "threshold"
    assert (
        checkpoint.included_event_id
        == session_store.load_event_log(session_id).events[0].event_id
    )

    restored_app = AceAgentApp(
        Agent(
            prompt="Prompt",
            default_model="gpt-4o",
            llm_service=CompressingLLMService(
                make_stream(response=LLMResponse(text="later"), deltas=["later"])
            ),
            executor=StubExecutor(),
            max_steps=1,
        ),
        provider_name="openai",
        selected_model="gpt-4o",
        session_store=session_store,
        session_id=session_id,
    )
    restored_text = "\n".join(
        part["data"]
        for message in restored_app.llm_history
        for part in message.content
        if part["type"] == "text"
    )

    assert "Earlier discussion summary." in restored_text
    assert "done" in restored_text
    assert "history message 0" not in restored_text


def _session_event(
    event_id: str,
    kind: str,
    payload: dict,
) -> SessionEvent:
    return SessionEvent(
        event_id=event_id,
        session_id="session-1",
        run_id="run-1",
        kind=kind,
        payload=payload,
    )
