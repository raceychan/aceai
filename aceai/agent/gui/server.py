"""Lihil-backed realtime adapter for AceAI GUI clients."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from msgspec import Struct

from aceai.agent.app import AceAgentApp, AgentAppEvent
from aceai.agent.config import AgentAppConfig
from aceai.agent.memory.ideas import Idea, IdeaStore
from aceai.agent.references import ReferenceCandidate, reference_candidates
from aceai.agent.session import SessionEvent, SessionStore
from aceai.agent.session_views import (
    agent_snapshot_payload,
    delete_empty_sessions as delete_empty_session_ids,
    events_after,
    idea_display_index,
    idea_payload,
    jsonable_value,
    project_file_path,
    project_file_payload,
    session_list_item_payload,
)
from aceai.agent.tui.cli import build_agent
from aceai.core import Agent

NEW_SESSION_TOPIC_ID = "new"
SESSION_EVENT = "session.event"
APP_EVENT = "agent.event"
RUN_CANCELLED_EVENT = "run.cancelled"
RUN_TASK_NAME = "active_run"


class SendMessageRequest(Struct, frozen=True, kw_only=True):
    content: str


class SnapshotRequest(Struct, frozen=True, kw_only=True):
    after_event_id: str | None = None


class SwitchThreadRequest(Struct, frozen=True, kw_only=True):
    thread_id: str


class ApprovalRequest(Struct, frozen=True, kw_only=True):
    thread_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None


class RejectToolRequest(Struct, frozen=True, kw_only=True):
    reason: str
    thread_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None


class IdeaCaptureRequest(Struct, frozen=True, kw_only=True):
    content: str


class IdeaUpdateRequest(Struct, frozen=True, kw_only=True):
    content: str


class FileSaveRequest(Struct, frozen=True, kw_only=True):
    content: str


class AceAIGuiRuntime:
    """Factory and session-store boundary shared by GUI channels."""

    def __init__(
        self,
        *,
        config: AgentAppConfig,
        session_store: SessionStore | None = None,
        idea_store: IdeaStore | None = None,
        agent_factory: Callable[[AgentAppConfig], Agent] = build_agent,
        agent_app_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._session_store = session_store or SessionStore()
        self._idea_store = idea_store or IdeaStore()
        self._agent_factory = agent_factory
        self._agent_app_factory = agent_app_factory

    @property
    def session_store(self) -> SessionStore:
        return self._session_store

    @property
    def project_root(self) -> Path:
        return Path(self._session_store.project.root_path).resolve()

    def list_ideas(self) -> list[Idea]:
        return self._idea_store.list_for_display(
            current_project=self._session_store.project,
        )

    def capture_idea(self, content: str) -> Idea:
        return self._idea_store.capture(
            content,
            project=self._session_store.project,
        )

    def update_idea(self, index: int, content: str) -> Idea:
        return self._idea_store.update_displayed(
            index,
            content,
            current_project=self._session_store.project,
        )

    def delete_idea(self, index: int) -> Idea:
        return self._idea_store.delete_displayed(
            index,
            current_project=self._session_store.project,
        )

    def create_agent_app(self, session_id: str) -> AceAgentApp:
        if self._agent_app_factory is not None:
            return self._agent_app_factory(session_id)
        agent = self._agent_factory(self._config)
        app = AceAgentApp(
            agent,
            provider_name=self._config.provider,
            selected_model=self._config.model,
            reasoning_level=self._config.reasoning_level,
            session_store=self._session_store,
            idea_store=self._idea_store,
        )
        if session_id == NEW_SESSION_TOPIC_ID:
            app.ensure_session()
        else:
            app.switch_session(session_id)
        return app


def build_gui_app(runtime: AceAIGuiRuntime) -> Any:
    """Build a Lihil ASGI app exposing AceAI realtime channels."""

    try:
        from lihil import (
            ChannelBase,
            ISocket,
            Lihil,
            MessageEnvelope,
            Resolver,
            SocketBus,
            SocketHub,
            Topic,
        )
        from lihil import Route
        from starlette.middleware.cors import CORSMiddleware
    except ModuleNotFoundError as exc:
        if exc.name == "lihil":
            raise RuntimeError(
                "AceAI GUI server requires the optional gui dependencies. "
                "Install with `aceai[gui]`."
            ) from None
        raise

    class AceAISessionChannel(ChannelBase):
        topic = Topic("session:{session_id}")

        def __init__(
            self,
            socket: ISocket,
            *,
            topic: str,
            bus: SocketBus,
            resolver: Resolver,
        ):
            super().__init__(socket, topic=topic, bus=bus, resolver=resolver)
            self._agent_app: AceAgentApp | None = None
            self._session_id: str | None = None

        @property
        def agent_app(self) -> AceAgentApp:
            if self._agent_app is None:
                raise RuntimeError("AceAI session channel is not joined")
            return self._agent_app

        @property
        def session_id(self) -> str:
            if self._session_id is None:
                raise RuntimeError("AceAI session channel is not joined")
            return self._session_id

        async def on_join(self, **params: str) -> None:
            session_id = params["session_id"]
            self._agent_app = runtime.create_agent_app(session_id)
            actual_session_id = self._agent_app.session_id
            if actual_session_id is None:
                raise RuntimeError("AceAI session was not initialized")
            self._session_id = actual_session_id
            await super().on_join(session_id=session_id)

        async def on_snapshot(self, payload: SnapshotRequest) -> dict[str, Any]:
            return agent_snapshot_payload(
                self.agent_app,
                after_event_id=payload.after_event_id,
            )

        async def on_switch_thread(
            self,
            payload: SwitchThreadRequest,
        ) -> dict[str, Any]:
            self.agent_app.switch_thread(payload.thread_id)
            return agent_snapshot_payload(self.agent_app, after_event_id=None)

        async def on_send_message(
            self,
            payload: SendMessageRequest,
        ) -> dict[str, Any]:
            self.start_task(RUN_TASK_NAME, self._stream_turn(payload.content))
            return {
                "accepted": True,
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
            }

        async def on_approve_tool(self, payload: ApprovalRequest) -> dict[str, Any]:
            self.start_task(
                RUN_TASK_NAME,
                self._stream_approval(payload),
            )
            return {
                "accepted": True,
                "session_id": self.session_id,
            }

        async def on_reject_tool(self, payload: RejectToolRequest) -> dict[str, Any]:
            self.start_task(
                RUN_TASK_NAME,
                self._stream_rejection(payload),
            )
            return {
                "accepted": True,
                "session_id": self.session_id,
            }

        async def on_cancel(self) -> dict[str, Any]:
            await self.cancel_task(RUN_TASK_NAME)
            self.agent_app.cancel_active_turn()
            await self.publish(
                {
                    "session_id": self.session_id,
                    "thread_id": self.agent_app.active_thread_id,
                },
                event=RUN_CANCELLED_EVENT,
            )
            return {
                "cancelled": True,
                "session_id": self.session_id,
            }

        async def replay_after(self, event_id: str | None) -> list[MessageEnvelope]:
            if self._session_id is None or event_id is None:
                return []
            events = events_after(
                runtime.session_store.load_event_log(self._session_id).events,
                event_id,
            )
            return [
                MessageEnvelope(
                    topic=self.resolved_topic,
                    event=SESSION_EVENT,
                    payload=event.as_json(),
                    event_id=event.event_id,
                )
                for event in events
            ]

        async def _stream_turn(self, content: str) -> None:
            async for event in self.agent_app.start_turn_events(content):
                await self.publish(_app_event_payload(event), event=APP_EVENT)

        async def _stream_approval(self, payload: ApprovalRequest) -> None:
            async for event in self.agent_app.approve_tool(
                thread_id=payload.thread_id,
                run_id=payload.run_id,
                tool_call_id=payload.tool_call_id,
            ):
                await self.publish(
                    _app_event_payload(
                        AgentAppEvent(
                            thread_id=payload.thread_id or self.agent_app.active_thread_id,
                            agent_id="",
                            event=event,
                        )
                    ),
                    event=APP_EVENT,
                )

        async def _stream_rejection(self, payload: RejectToolRequest) -> None:
            async for event in self.agent_app.reject_tool(
                payload.reason,
                thread_id=payload.thread_id,
                run_id=payload.run_id,
                tool_call_id=payload.tool_call_id,
            ):
                await self.publish(
                    _app_event_payload(
                        AgentAppEvent(
                            thread_id=payload.thread_id or self.agent_app.active_thread_id,
                            agent_id="",
                            event=event,
                        )
                    ),
                    event=APP_EVENT,
                )

    hub = SocketHub("/ws")
    hub.channel(AceAISessionChannel)

    sessions_route = Route("/api/sessions", in_schema=False)

    @sessions_route.get
    def list_sessions() -> dict[str, Any]:
        session_payloads = [
            (
                metadata,
                session_list_item_payload(runtime.session_store, metadata),
            )
            for metadata in runtime.session_store.list_sessions()
        ]
        session_payloads.sort(
            key=lambda item: (
                0 if item[1]["event_count"] > 0 else 1,
                -item[0].updated_at.timestamp(),
            )
        )
        return {
            "sessions": [payload for _metadata, payload in session_payloads]
        }

    session_cleanup_route = Route("/api/session-cleanup/empty", in_schema=False)

    @session_cleanup_route.delete
    def delete_empty_sessions() -> dict[str, Any]:
        deleted_session_ids = delete_empty_session_ids(runtime.session_store)
        return {
            "deleted": len(deleted_session_ids),
            "session_ids": deleted_session_ids,
        }

    session_route = Route("/api/sessions/{session_id}", in_schema=False)

    @session_route.delete
    def delete_session(session_id: str) -> dict[str, Any]:
        runtime.session_store.delete_session(session_id)
        return {"deleted": True, "session_id": session_id}

    references_route = Route("/api/references", in_schema=False)

    @references_route.get
    def list_references(q: str = "", kind: str = "all") -> dict[str, Any]:
        return {
            "items": [
                _reference_candidate_payload(candidate)
                for candidate in reference_candidates(
                    root=runtime.project_root,
                    ideas=runtime.list_ideas(),
                    query=q,
                    kind=kind,
                    limit=30,
                )
            ]
        }

    file_route = Route("/api/files", in_schema=False)

    @file_route.get
    def read_file(path: str) -> dict[str, Any]:
        return {"file": project_file_payload(runtime.project_root, path)}

    @file_route.put
    def save_file(path: str, payload: FileSaveRequest) -> dict[str, Any]:
        target = project_file_path(runtime.project_root, path)
        target.write_text(payload.content)
        return {"file": project_file_payload(runtime.project_root, path)}

    ideas_route = Route("/api/ideas", in_schema=False)

    @ideas_route.get
    def list_ideas() -> dict[str, Any]:
        return {
            "ideas": [
                idea_payload(idea, index)
                for index, idea in enumerate(runtime.list_ideas(), start=1)
            ]
        }

    @ideas_route.post
    def capture_idea(payload: IdeaCaptureRequest) -> dict[str, Any]:
        idea = runtime.capture_idea(payload.content)
        return {
            "idea": idea_payload(
                idea,
                idea_display_index(runtime.list_ideas(), idea.idea_id),
            )
        }

    idea_route = Route("/api/ideas/{index}", in_schema=False)

    @idea_route.put
    def update_idea(index: int, payload: IdeaUpdateRequest) -> dict[str, Any]:
        return {"idea": idea_payload(runtime.update_idea(index, payload.content), index)}

    @idea_route.delete
    def delete_idea(index: int) -> dict[str, Any]:
        return {"deleted": True, "idea": idea_payload(runtime.delete_idea(index), index)}

    app = Lihil(
        hub,
        sessions_route,
        session_cleanup_route,
        session_route,
        references_route,
        file_route,
        ideas_route,
        idea_route,
    )
    def cors_middleware(asgi_app):
        return CORSMiddleware(
            asgi_app,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    app.add_middleware(cors_middleware)
    return app


def _reference_candidate_payload(candidate: ReferenceCandidate) -> dict[str, Any]:
    payload = {
        "kind": candidate.kind,
        "value": candidate.value,
        "label": candidate.label,
        "description": candidate.description,
    }
    if candidate.idea_id is not None:
        payload["idea_id"] = candidate.idea_id
    return payload


def _app_event_payload(app_event: AgentAppEvent) -> dict[str, Any]:
    event = app_event.event
    if isinstance(event, SessionEvent):
        return {
            "kind": "session",
            "thread_id": app_event.thread_id,
            "agent_id": app_event.agent_id,
            "event": event.as_json(),
        }
    return {
        "kind": "agent",
        "thread_id": app_event.thread_id,
        "agent_id": app_event.agent_id,
        "event": _agent_event_payload(event),
    }


def _agent_event_payload(event) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "run_id": event.run_id,
        "step_id": event.step_id,
        "step_index": event.step_index,
        "payload": _jsonable_agent_event_fields(event),
    }


def _jsonable_agent_event_fields(event) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in event.__struct_fields__:
        if name in {"run_id", "step_id", "step_index"}:
            continue
        payload[name] = jsonable_value(getattr(event, name))
    return payload
