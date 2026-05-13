from typing import Any
from datetime import datetime, timezone

import pytest
import msgspec

lihil = pytest.importorskip("lihil")

from lihil.vendors import TestClient

from aceai.agent.app import AgentAppEvent, QueuedTurn
from aceai.agent.config import AgentAppConfig
from aceai.agent.gui.server import AceAIGuiRuntime, build_gui_app
from aceai.agent.memory.ideas import IdeaStore
from aceai.agent.project import ProjectMetadata
from aceai.agent.session import MAIN_THREAD_ID, SessionEvent, SessionStore
from aceai.agent.session_service import (
    SessionService,
    UserImageAttachment,
    agent_event_to_session_event,
)
from aceai.core.events import AgentEventBuilder
from aceai.core.models import ToolApprovalDecision, ToolApprovalRequest
from aceai.llm.models import LLMToolCall


APPROVAL_REQUEST = ToolApprovalRequest(
    call=LLMToolCall(
        name="write_file",
        arguments='{"path":"demo.txt"}',
        call_id="call-1",
    ),
    tool_name="write_file",
    reason="requires approval",
    policy="test_policy",
)


class FakeAgentApp:
    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.session_service = SessionService(store=store)
        if session_id == "new":
            self.session_service.ensure_session()
        else:
            self.session_service.attach_session(session_id)
        self.active_thread_id = MAIN_THREAD_ID
        self.cancelled = False
        self._queued_turns: list[QueuedTurn] = []
        self._pending_approval: ToolApprovalRequest | None = None
        self._active_run = None
        self.accepts_user_turn = True
        self.steered_questions: list[str] = []

    @property
    def session_id(self) -> str | None:
        return self.session_service.session_id

    @property
    def queued_questions(self) -> tuple[str, ...]:
        return tuple(turn.content for turn in self._queued_turns)

    @property
    def queued_turns(self) -> tuple[QueuedTurn, ...]:
        return tuple(self._queued_turns)

    @property
    def is_running_suspended(self) -> bool:
        return False

    @property
    def active_thread_accepts_user_turn(self) -> bool:
        return self.accepts_user_turn

    @property
    def active_run(self):
        return self._active_run

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def selected_model(self) -> str:
        return "gpt-5.5"

    @property
    def reasoning_level(self) -> str:
        return "medium"

    def pending_approval_request(self):
        return self._pending_approval

    async def start_turn_events(self, content: str, images=()):
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("fake session is not active")
        run_id = "run-1"
        self.session_service.record_user_message(
            content,
            run_id=run_id,
            thread_id=MAIN_THREAD_ID,
            images=images,
        )
        persisted_event_id = self.session_service.record_session_event(
            SessionEvent(
                event_id="evt-run-completed",
                run_id=run_id,
                step_id="step-1",
                step_index=0,
                kind="run_completed",
                payload={"content": "hello", "status": "completed"},
            )
        )
        if persisted_event_id is None:
            raise RuntimeError("fake event did not persist")
        event = self.session_service.store.load_event_log(session_id).events[-1]
        yield AgentAppEvent(thread_id=MAIN_THREAD_ID, agent_id="", event=event)

    async def approve_tool(self, **kwargs: Any):
        async for event in self.approve_tool_events(**kwargs):
            if event.raw_event is not None:
                yield event.raw_event

    async def approve_tool_events(self, **kwargs: Any):
        self._pending_approval = None
        builder = AgentEventBuilder(run_id="run-approval", step_id="step-1", step_index=0)
        raw_event = builder.tool_approval_resolved(
            request=APPROVAL_REQUEST,
            decision=ToolApprovalDecision.approve(APPROVAL_REQUEST),
        )
        session_event = agent_event_to_session_event(raw_event)
        yield AgentAppEvent(event=session_event, raw_event=raw_event, thread_id=MAIN_THREAD_ID, agent_id="")

    async def reject_tool(self, reason: str, **kwargs: Any):
        async for event in self.reject_tool_events(reason, **kwargs):
            if event.raw_event is not None:
                yield event.raw_event

    async def reject_tool_events(self, reason: str, **kwargs: Any):
        self._pending_approval = None
        builder = AgentEventBuilder(run_id="run-approval", step_id="step-1", step_index=0)
        raw_event = builder.tool_approval_resolved(
            request=APPROVAL_REQUEST,
            decision=ToolApprovalDecision.reject(APPROVAL_REQUEST, reason=reason),
        )
        session_event = agent_event_to_session_event(raw_event)
        yield AgentAppEvent(event=session_event, raw_event=raw_event, thread_id=MAIN_THREAD_ID, agent_id="")

    def switch_thread(self, thread_id: str):
        self.active_thread_id = thread_id
        session_id = self.session_id
        if session_id is None:
            raise RuntimeError("fake session is not active")
        return self.session_service.snapshot_thread(session_id, thread_id)

    def cancel_active_turn(self) -> None:
        self.cancelled = True

    def enqueue_turn(
        self,
        question: str,
        *,
        images: tuple[UserImageAttachment, ...] = (),
    ) -> int:
        if not self.accepts_user_turn:
            raise RuntimeError("Delegated subagent thread is still running")
        self._queued_turns.append(QueuedTurn(content=question, images=images))
        return len(self._queued_turns)

    def take_queued_turn(self, index: int) -> QueuedTurn:
        return self._queued_turns.pop(index)

    def cancel_queued_turn(self, index: int) -> QueuedTurn:
        return self._queued_turns.pop(index)

    def steer_active_child_thread(self, question: str) -> bool:
        if self.accepts_user_turn:
            return False
        self.steered_questions.append(question)
        return True


def _config() -> AgentAppConfig:
    return AgentAppConfig(
        provider="openai",
        api_key="test-key",
        model="gpt-5.5",
        default_model="gpt-5.5",
    )


def _runtime(store: SessionStore) -> AceAIGuiRuntime:
    def app_factory(session_id: str):
        return FakeAgentApp(store, session_id)

    return AceAIGuiRuntime(
        config=_config(),
        session_store=store,
        agent_app_factory=app_factory,
    )


def _runtime_with_app(store: SessionStore) -> tuple[AceAIGuiRuntime, dict[str, FakeAgentApp]]:
    apps: dict[str, FakeAgentApp] = {}

    def app_factory(session_id: str):
        app = FakeAgentApp(store, session_id)
        if app.session_id is None:
            raise RuntimeError("fake app did not create a session")
        apps[app.session_id] = app
        return app

    return (
        AceAIGuiRuntime(
            config=_config(),
            session_store=store,
            agent_app_factory=app_factory,
        ),
        apps,
    )


def _project(root) -> ProjectMetadata:
    return ProjectMetadata(
        project_id="project-1",
        name="project",
        root_path=str(root),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _encode(topic: str, event: str, payload=None, ref: str | None = None) -> bytes:
    return msgspec.json.encode(
        {
            "topic": topic,
            "event": event,
            "payload": payload,
            "ref": ref,
        }
    )


def _reply_payload(data: dict[str, Any]) -> dict[str, Any]:
    assert data["event"] == "reply"
    assert data["payload"]["status"] == "ok"
    return data["payload"]["response"]


def test_gui_session_channel_snapshot_and_message_stream(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            join_response = _reply_payload(ws.receive_json())
            assert join_response["replay_supported"] is True

            ws.send_bytes(_encode("session:new", "snapshot", {}, "snapshot-1"))
            snapshot = _reply_payload(ws.receive_json())
            session_id = snapshot["session"]["session_id"]
            assert session_id != "new"
            assert snapshot["active_thread_id"] == MAIN_THREAD_ID
            assert snapshot["threads"][0]["thread_id"] == MAIN_THREAD_ID
            assert snapshot["runtime"]["queued_questions"] == []
            assert snapshot["runtime"]["queued_turns"] == []
            assert snapshot["runtime"]["pending_approval"] is None

            ws.send_bytes(
                _encode(
                    "session:new",
                    "send_message",
                    {"content": "hi"},
                    "send-1",
                )
            )
            messages = [ws.receive_json(), ws.receive_json()]
            replies = [msg for msg in messages if msg["event"] == "reply"]
            events = [msg for msg in messages if msg["event"] == "agent.event"]
            assert _reply_payload(replies[0])["accepted"] is True
            assert events[0]["payload"]["kind"] == "session"
            assert events[0]["payload"]["event"]["kind"] == "run_completed"


def test_gui_session_channel_sends_image_attachments(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()

            ws.send_bytes(
                _encode(
                    "session:new",
                    "send_message",
                    {
                        "content": "look",
                        "attachments": [
                            {"mime_type": "image/png", "data": "cG5n"}
                        ],
                    },
                    "send-1",
                )
            )
            ws.receive_json()
            ws.receive_json()

    event_log = store.load_event_log(store.list_sessions()[0].session_id)
    user_message = next(event for event in event_log.events if event.kind == "user_message")
    assert user_message.payload["images"] == [
        {"mime_type": "image/png", "data": "cG5n"}
    ]


def test_gui_session_channel_queues_and_cancels_messages(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    runtime, apps = _runtime_with_app(store)
    app = build_gui_app(runtime)
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()
            fake_app = next(iter(apps.values()))

            ws.send_bytes(
                _encode(
                    "session:new",
                    "enqueue_message",
                    {"content": "later"},
                    "queue-1",
                )
            )
            queued = _reply_payload(ws.receive_json())
            assert queued["queued_questions"] == ["later"]
            assert queued["queued_turns"] == [
                {"content": "later", "images": []}
            ]
            assert fake_app.queued_questions == ("later",)

            ws.send_bytes(
                _encode(
                    "session:new",
                    "cancel_queued_message",
                    {"index": 0},
                    "cancel-1",
                )
            )
            cancelled = _reply_payload(ws.receive_json())
            assert cancelled["queued_questions"] == []
            assert cancelled["queued_turns"] == []
            assert fake_app.queued_questions == ()


def test_gui_session_channel_starts_queued_image_message(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    runtime, apps = _runtime_with_app(store)
    app = build_gui_app(runtime)
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()
            fake_app = next(iter(apps.values()))

            ws.send_bytes(
                _encode(
                    "session:new",
                    "enqueue_message",
                    {
                        "content": "look later",
                        "attachments": [
                            {"mime_type": "image/png", "data": "cG5n"}
                        ],
                    },
                    "queue-1",
                )
            )
            queued = _reply_payload(ws.receive_json())
            assert queued["queued_questions"] == ["look later"]
            assert queued["queued_turns"] == [
                {
                    "content": "look later",
                    "images": [{"mime_type": "image/png", "data": "cG5n"}],
                }
            ]
            assert fake_app.queued_questions == ("look later",)

            ws.send_bytes(
                _encode(
                    "session:new",
                    "start_queued_message",
                    {"index": 0},
                    "start-1",
                )
            )
            started = _reply_payload(ws.receive_json())
            assert started["queued_questions"] == []
            assert started["queued_turns"] == []
            ws.receive_json()

    event_log = store.load_event_log(store.list_sessions()[0].session_id)
    user_message = next(event for event in event_log.events if event.kind == "user_message")
    assert user_message.payload["content"] == "look later"
    assert user_message.payload["images"] == [
        {"mime_type": "image/png", "data": "cG5n"}
    ]


def test_gui_session_channel_steers_child_thread_without_run_task(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    runtime, apps = _runtime_with_app(store)
    app = build_gui_app(runtime)
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()
            fake_app = next(iter(apps.values()))
            fake_app.accepts_user_turn = False

            ws.send_bytes(
                _encode(
                    "session:new",
                    "send_message",
                    {"content": "redirect"},
                    "send-1",
                )
            )
            reply = _reply_payload(ws.receive_json())

    assert reply["mode"] == "steered"
    assert fake_app.steered_questions == ["redirect"]


def test_gui_session_channel_rejects_unknown_message_fields(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()

            ws.send_bytes(
                _encode(
                    "session:new",
                    "send_message",
                    {"content": "hi", "future_field": "must fail"},
                    "send-1",
                )
            )
            reply = ws.receive_json()

    assert reply["event"] == "reply"
    assert reply["payload"]["status"] == "error"
    assert reply["payload"]["error"]["code"] == "invalid_payload"
    assert store.list_sessions()[0].session_id
    assert store.load_event_log(store.list_sessions()[0].session_id).events == []


def test_gui_session_channel_rejects_unknown_image_fields(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()

            ws.send_bytes(
                _encode(
                    "session:new",
                    "send_message",
                    {
                        "content": "look",
                        "attachments": [
                            {
                                "mime_type": "image/png",
                                "data": "cG5n",
                                "filename": "old-client.png",
                            }
                        ],
                    },
                    "send-1",
                )
            )
            reply = ws.receive_json()

    assert reply["event"] == "reply"
    assert reply["payload"]["status"] == "error"
    assert reply["payload"]["error"]["code"] == "invalid_payload"
    assert store.load_event_log(store.list_sessions()[0].session_id).events == []


def test_gui_session_snapshot_filters_after_event_id(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()

            ws.send_bytes(
                _encode("session:new", "send_message", {"content": "hi"}, "send-1")
            )
            ws.receive_json()
            ws.receive_json()

            events = store.load_event_log(store.list_sessions()[0].session_id).events
            ws.send_bytes(
                _encode(
                    "session:new",
                    "snapshot",
                    {"after_event_id": events[0].event_id},
                    "snapshot-1",
                )
            )
            snapshot = _reply_payload(ws.receive_json())
            assert [event["event_id"] for event in snapshot["events"]] == [
                event.event_id for event in events[1:]
            ]


def test_gui_sessions_api_lists_saved_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first = store.create_session()
    second = store.create_session()
    service = SessionService(store=store)
    service.attach_session(first.session_id)
    service.record_user_message("hi", run_id="run-1", thread_id=MAIN_THREAD_ID)
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        response = client.get("/api/sessions")

    assert response.status_code == 200
    payload = response.json()
    session_ids = [session["session_id"] for session in payload["sessions"]]
    assert first.session_id in session_ids
    assert second.session_id in session_ids
    assert payload["sessions"][0]["session_id"] == first.session_id
    first_payload = next(
        session for session in payload["sessions"]
        if session["session_id"] == first.session_id
    )
    assert first_payload["event_count"] == 1
    assert first_payload["total_cost_usd"] == 0
    assert first_payload["active_thread"]["thread_id"] == MAIN_THREAD_ID


def test_gui_openapi_schema_includes_http_api(tmp_path) -> None:
    app = build_gui_app(_runtime(SessionStore(tmp_path / "sessions")))

    schema = app.genereate_oas()

    assert "/api/sessions" in schema.paths
    assert "/api/config" in schema.paths
    assert "/api/files" in schema.paths
    assert "/api/ideas" in schema.paths
    assert "SessionsPayload" in schema.components.schemas
    assert "GuiConfigPayload" in schema.components.schemas
    assert "FileResponsePayload" in schema.components.schemas
    assert "IdeaResponsePayload" in schema.components.schemas


def test_gui_config_api_includes_tools_and_skills(tmp_path) -> None:
    app = build_gui_app(_runtime(SessionStore(tmp_path / "sessions")))
    client = TestClient(app)

    with client:
        response = client.get("/api/config")

    assert response.status_code == 200
    payload = response.json()
    skill_names = {skill["name"] for skill in payload["skills"]}
    tool_names = {tool["name"] for tool in payload["tools"]}
    assert "developer" in skill_names
    assert "skill-creator" in skill_names
    assert "read_text_file" in tool_names
    assert "run_shell_command" in tool_names


def test_gui_api_allows_cross_origin_browser_clients(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.create_session()
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        response = client.get("/api/sessions", headers={"Origin": "null"})
        options = client.options(
            "/api/sessions",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.headers["access-control-allow-origin"] == "*"
    assert options.status_code == 200
    assert "GET" in options.headers["access-control-allow-methods"]


def test_gui_session_cleanup_deletes_only_empty_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    kept = store.create_session()
    deleted = store.create_session()
    service = SessionService(store=store)
    service.attach_session(kept.session_id)
    service.record_user_message("keep me", run_id="run-1", thread_id=MAIN_THREAD_ID)
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        response = client.delete("/api/session-cleanup/empty")
        sessions_response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json() == {
        "deleted": 1,
        "session_ids": [deleted.session_id],
    }
    session_ids = [
        session["session_id"]
        for session in sessions_response.json()["sessions"]
    ]
    assert kept.session_id in session_ids
    assert deleted.session_id not in session_ids


def test_gui_snapshot_includes_observability_payload(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create_session()
    store.append_event(
        metadata.session_id,
        SessionEvent(
            event_id="evt-tool-started",
            run_id="run-1",
            step_id="step-1",
            step_index=0,
            kind="tool_started",
            payload={
                "content": "started",
                "tool_name": "search_text",
                "tool_call_id": "call-1",
            },
        ),
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            event_id="evt-tool-completed",
            run_id="run-1",
            step_id="step-1",
            step_index=0,
            kind="tool_completed",
            payload={
                "content": "done",
                "tool_name": "search_text",
                "tool_call_id": "call-1",
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 4,
                    "output_tokens": 6,
                    "total_tokens": 16,
                    "input_cache_hit_rate": 0.4,
                },
                "cost": {
                    "total_cost_usd": 0.001,
                },
            },
        ),
    )
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode(f"session:{metadata.session_id}", "join", {}, "join-1"))
            ws.receive_json()
            ws.send_bytes(_encode(f"session:{metadata.session_id}", "snapshot", {}, "snapshot-1"))
            snapshot = _reply_payload(ws.receive_json())

    observability = snapshot["observability"]
    assert observability["usage"]["total_tokens"] == 16
    assert observability["usage"]["total_cost_usd"] == 0.001
    assert observability["event_counts"][0] == {"kind": "tool_completed", "count": 1}
    assert observability["tool_calls"] == [
        {
            "name": "search_text",
            "calls": 1,
            "succeeded": 1,
            "failed": 0,
            "approval_requests": 0,
        }
    ]
    assert observability["trajectory"][0]["summary"] == "started"
    assert observability["debug_events"][0]["payload"]["tool_name"] == "search_text"


def test_gui_session_channel_switches_thread(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create_session()
    child = store.create_thread(
        session_id=metadata.session_id,
        thread_id="thread-child",
        agent_id="agent-child",
        role="subagent",
        title="Research",
        parent_thread_id=MAIN_THREAD_ID,
        parent_run_id="run-1",
        parent_tool_call_id="tool-1",
    )
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            topic = f"session:{metadata.session_id}"
            ws.send_bytes(_encode(topic, "join", {}, "join-1"))
            ws.receive_json()

            ws.send_bytes(
                _encode(
                    topic,
                    "switch_thread",
                    {"thread_id": child.thread_id},
                    "switch-1",
                )
            )
            snapshot = _reply_payload(ws.receive_json())

    assert snapshot["active_thread_id"] == child.thread_id
    assert snapshot["threads"][1]["role"] == "subagent"


def test_gui_sessions_api_deletes_saved_session(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create_session()
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        response = client.delete(f"/api/sessions/{metadata.session_id}")

    assert response.status_code == 200
    assert response.json()["session_id"] == metadata.session_id
    assert store.list_sessions() == []


def test_gui_references_api_lists_files_and_ideas(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "app.py").write_text("print('hi')\n")
    (project_root / ".hidden.py").write_text("ignore\n")
    (project_root / "node_modules").mkdir()
    (project_root / "node_modules" / "ignored.js").write_text("ignore\n")
    store = SessionStore(tmp_path / "sessions", project=_project(project_root))
    idea_store = IdeaStore(tmp_path / "ideas.sqlite3")
    idea_store.capture(
        "Remember the GUI reference picker",
        project=store.project,
    )
    app = build_gui_app(
        AceAIGuiRuntime(
            config=_config(),
            session_store=store,
            idea_store=idea_store,
            agent_app_factory=lambda session_id: FakeAgentApp(store, session_id),
        )
    )
    client = TestClient(app)

    with client:
        response = client.get("/api/references?q=app")
        idea_response = client.get("/api/references?q=idea%3AGUI")
        ideas_response = client.get("/api/ideas")

    assert response.status_code == 200
    items = response.json()["items"]
    assert {
        "kind": "file",
        "value": "@app.py",
        "label": "app.py",
        "description": "file",
    } in items
    assert all(item["value"] != "@node_modules/ignored.js" for item in items)
    assert idea_response.status_code == 200
    assert idea_response.json()["items"][0]["kind"] == "idea"
    assert idea_response.json()["items"][0]["description"] == "Remember the GUI reference picker"
    assert ideas_response.status_code == 200
    assert ideas_response.json()["ideas"][0]["content"] == "Remember the GUI reference picker"


def test_gui_files_api_reads_and_saves_project_file(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    target = project_root / "app.py"
    target.write_text("print('old')\n")
    store = SessionStore(tmp_path / "sessions", project=_project(project_root))
    app = build_gui_app(_runtime(store))
    client = TestClient(app)

    with client:
        read_response = client.get("/api/files?path=app.py")
        save_response = client.put(
            "/api/files?path=app.py",
            json={"content": "print('new')\n"},
        )

    assert read_response.status_code == 200
    assert read_response.json()["file"]["path"] == "app.py"
    assert read_response.json()["file"]["content"] == "print('old')\n"
    assert save_response.status_code == 200
    assert save_response.json()["file"]["content"] == "print('new')\n"
    assert target.read_text() == "print('new')\n"


def test_gui_ideas_api_captures_idea(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    store = SessionStore(tmp_path / "sessions", project=_project(project_root))
    idea_store = IdeaStore(tmp_path / "ideas.sqlite3")
    idea_store.capture("Existing GUI idea", project=store.project)
    app = build_gui_app(
        AceAIGuiRuntime(
            config=_config(),
            session_store=store,
            idea_store=idea_store,
            agent_app_factory=lambda session_id: FakeAgentApp(store, session_id),
        )
    )
    client = TestClient(app)

    with client:
        response = client.post(
            "/api/ideas",
            json={"content": "Capture this GUI idea"},
        )

    assert response.status_code == 200
    assert response.json()["idea"]["content"] == "Capture this GUI idea"
    assert response.json()["idea"]["index"] == 2
    assert idea_store.list_for_display(current_project=store.project)[1].content == "Capture this GUI idea"


def test_gui_ideas_api_updates_and_deletes_displayed_idea(tmp_path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    store = SessionStore(tmp_path / "sessions", project=_project(project_root))
    idea_store = IdeaStore(tmp_path / "ideas.sqlite3")
    idea_store.capture("Original idea", project=store.project)
    app = build_gui_app(
        AceAIGuiRuntime(
            config=_config(),
            session_store=store,
            idea_store=idea_store,
            agent_app_factory=lambda session_id: FakeAgentApp(store, session_id),
        )
    )
    client = TestClient(app)

    with client:
        update_response = client.put(
            "/api/ideas/1",
            json={"content": "Updated idea"},
        )
        delete_response = client.delete("/api/ideas/1")

    assert update_response.status_code == 200
    assert update_response.json()["idea"]["content"] == "Updated idea"
    assert delete_response.status_code == 200
    assert delete_response.json()["idea"]["content"] == "Updated idea"
    assert idea_store.list_for_display(current_project=store.project) == []


def test_gui_approval_resume_events_use_app_event_envelope(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    fake_apps: list[FakeAgentApp] = []

    def app_factory(session_id: str):
        fake = FakeAgentApp(store, session_id)
        fake._pending_approval = APPROVAL_REQUEST
        fake_apps.append(fake)
        return fake

    runtime = AceAIGuiRuntime(
        config=_config(),
        session_store=store,
        agent_app_factory=app_factory,
    )
    app = build_gui_app(runtime)
    client = TestClient(app)

    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(_encode("session:new", "join", {}, "join-1"))
            ws.receive_json()

            ws.send_bytes(
                _encode(
                    "session:new",
                    "snapshot",
                    {},
                    "snapshot-1",
                )
            )
            snapshot = _reply_payload(ws.receive_json())
            assert snapshot["runtime"]["pending_approval"]["tool_name"] == "write_file"

            ws.send_bytes(
                _encode(
                    "session:new",
                    "approve_tool",
                    {"tool_call_id": "call-1"},
                    "approve-1",
                )
            )
            messages = [ws.receive_json(), ws.receive_json()]

    replies = [msg for msg in messages if msg["event"] == "reply"]
    events = [msg for msg in messages if msg["event"] == "agent.event"]
    assert _reply_payload(replies[0])["accepted"] is True
    assert events[0]["payload"]["kind"] == "session"
    assert events[0]["payload"]["event"]["kind"] == "tool_approval_resolved"
