"""Lihil-backed realtime adapter for AceAI GUI clients."""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any

from msgspec import Struct, field

from aceai.agent.ace_agent import ACE_AGENT_BUILTIN_SKILL_PATHS
from aceai.agent.app import AceAgentApp, AgentAppEvent
from aceai.agent.config import AgentAppConfig, project_config_path, save_config
from aceai.agent.features.tools import default_agent_tools
from aceai.agent.memory.ideas import Idea, IdeaStore
from aceai.agent.provider_catalog import (
    api_key_env,
    auth_mode,
    model_options,
    provider_options,
    reasoning_effort_options,
)
from aceai.agent.references import ReferenceCandidate, reference_candidates
from aceai.agent.session import MAIN_THREAD_ID, SessionEvent, SessionStore
from aceai.agent.session_service import UserImageAttachment
from aceai.agent.session_views import (
    agent_snapshot_payload,
    agent_runtime_payload,
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
from aceai.core.skills import SkillLoader

NEW_SESSION_TOPIC_ID = "new"
SESSION_EVENT = "session.event"
APP_EVENT = "agent.event"
RUN_CANCELLED_EVENT = "run.cancelled"
RUN_TASK_NAME = "active_run"


class ImageAttachmentPayload(
    Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
):
    mime_type: str
    data: str


class SendMessageRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    content: str
    attachments: list[ImageAttachmentPayload] = field(default_factory=list)


class QueuedMessageRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    index: int


class SnapshotRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    after_event_id: str | None = None


class SwitchThreadRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    thread_id: str


class ApprovalRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    thread_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None


class RejectToolRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    reason: str
    thread_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None


class IdeaCaptureRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    content: str


class IdeaUpdateRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    content: str


class FileSaveRequest(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    content: str


def _user_image_attachments(
    attachments: list[ImageAttachmentPayload],
) -> tuple[UserImageAttachment, ...]:
    return tuple(
        UserImageAttachment(
            mime_type=attachment.mime_type,
            data=attachment.data,
        )
        for attachment in attachments
    )


def _draft_snapshot_payload(app: AceAgentApp) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "session": {
            "session_id": NEW_SESSION_TOPIC_ID,
            "project_id": app.session_service.store.project.project_id,
            "project_name": app.session_service.store.project.name,
            "created_at": now,
            "updated_at": now,
            "title": "New AceAI session",
            "path": "",
        },
        "state": {},
        "runtime": agent_runtime_payload(app),
        "observability": {
            "usage": {
                "context_tokens": None,
                "total_cost_usd": 0.0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "event_counts": [],
            "tool_calls": [],
            "trajectory": [],
            "debug_events": [],
        },
        "active_thread_id": MAIN_THREAD_ID,
        "threads": [
            {
                "session_id": NEW_SESSION_TOPIC_ID,
                "thread_id": MAIN_THREAD_ID,
                "agent_id": app.agent.agent_id,
                "role": "main",
                "title": "Main",
                "status": "idle",
                "parent_thread_id": None,
                "parent_run_id": None,
                "parent_tool_call_id": None,
                "metadata": {},
                "created_at": now,
                "updated_at": now,
            }
        ],
        "events": [],
    }


class ThreadMetadataPayload(Struct, frozen=True, kw_only=True):
    session_id: str
    thread_id: str
    role: str
    title: str
    status: str
    agent_id: str
    parent_thread_id: str | None
    parent_run_id: str | None
    parent_tool_call_id: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class SessionListItemPayload(Struct, frozen=True, kw_only=True, omit_defaults=True):
    session_id: str
    project_id: str
    project_name: str
    title: str
    created_at: str
    updated_at: str
    event_count: int
    total_cost_usd: float
    thread_count: int
    active_thread: ThreadMetadataPayload
    path: str | None = None


class SessionsPayload(Struct, frozen=True, kw_only=True):
    sessions: list[SessionListItemPayload]


class EmptySessionCleanupPayload(Struct, frozen=True, kw_only=True):
    deleted: int
    session_ids: list[str]


class SessionDeletePayload(Struct, frozen=True, kw_only=True):
    deleted: bool
    session_id: str


class ReferenceItemPayload(Struct, frozen=True, kw_only=True, omit_defaults=True):
    kind: str
    value: str
    label: str
    description: str
    idea_id: str | None = None


class ReferencesPayload(Struct, frozen=True, kw_only=True):
    items: list[ReferenceItemPayload]


class FilePayload(Struct, frozen=True, kw_only=True):
    path: str
    content: str
    size: int
    updated_at: str


class FileResponsePayload(Struct, frozen=True, kw_only=True):
    file: FilePayload


class IdeaItemPayload(Struct, frozen=True, kw_only=True):
    index: int
    idea_id: str
    created_at: str
    project_id: str
    project_name: str
    workspace: str
    content: str
    source_session_id: str | None


class IdeasPayload(Struct, frozen=True, kw_only=True):
    ideas: list[IdeaItemPayload]


class IdeaResponsePayload(Struct, frozen=True, kw_only=True):
    idea: IdeaItemPayload


class IdeaDeletePayload(Struct, frozen=True, kw_only=True):
    deleted: bool
    idea: IdeaItemPayload


class ProviderOptionPayload(Struct, frozen=True, kw_only=True):
    label: str
    value: str
    auth_mode: str
    api_key_env: str


class ModelOptionPayload(Struct, frozen=True, kw_only=True):
    label: str
    value: str


class ToolPermissionPayload(Struct, frozen=True, kw_only=True, omit_defaults=True):
    name: str
    description: str
    permission: str
    enabled: bool
    tags: list[str]
    max_calls_per_run: int | None = None


class SkillItemPayload(Struct, frozen=True, kw_only=True):
    name: str
    description: str
    location: str
    source: str
    builtin: bool
    enabled: bool


class GuiConfigPayload(Struct, frozen=True, kw_only=True):
    project_name: str
    git_branch: str
    provider: str
    model: str
    default_model: str
    reasoning_level: str
    compress_threshold: str
    api_timeout_seconds: float
    stream_start_timeout_seconds: float
    stream_event_timeout_seconds: float
    skill_selection_mode: str
    enabled_skills: list[str]
    disabled_providers: list[str]
    api_key_set: bool
    api_key_env: str
    config_path: str
    providers: list[ProviderOptionPayload]
    models: list[ModelOptionPayload]
    models_by_provider: dict[str, list[ModelOptionPayload]]
    reasoning_options: list[str]
    skills: list[SkillItemPayload]
    tools: list[ToolPermissionPayload]


class GuiConfigUpdateRequest(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    forbid_unknown_fields=True,
):
    provider: str
    model: str
    default_model: str
    reasoning_level: str
    compress_threshold: str
    api_timeout_seconds: float
    stream_start_timeout_seconds: float
    stream_event_timeout_seconds: float
    skill_selection_mode: str
    enabled_skills: list[str]
    disabled_providers: list[str]
    api_key: str | None = None
    tool_permissions: dict[str, str]
    tool_enabled: dict[str, bool]
    tool_max_calls: dict[str, int]


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

    @property
    def config(self) -> AgentAppConfig:
        return self._config

    def update_config(self, config: AgentAppConfig) -> AgentAppConfig:
        save_config(config, path=project_config_path(self.project_root))
        self._config = config
        return self._config

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
        if session_id != NEW_SESSION_TOPIC_ID:
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
            if actual_session_id is None and session_id != NEW_SESSION_TOPIC_ID:
                raise RuntimeError("AceAI session was not initialized")
            self._session_id = actual_session_id or NEW_SESSION_TOPIC_ID
            await super().on_join(session_id=session_id)

        async def on_snapshot(self, payload: SnapshotRequest) -> dict[str, Any]:
            if self.agent_app.session_id is None:
                return _draft_snapshot_payload(self.agent_app)
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
            if not self.agent_app.active_thread_accepts_user_turn:
                if payload.attachments:
                    raise ValueError("Child thread steering does not support image input")
                self.agent_app.steer_active_child_thread(payload.content)
                return {
                    "accepted": True,
                    "mode": "steered",
                    "session_id": self.session_id,
                    "thread_id": self.agent_app.active_thread_id,
                }
            if self.agent_app.session_id is None:
                self._session_id = self.agent_app.ensure_session()
                if payload.content:
                    runtime.session_store.update_session_title(
                        self._session_id,
                        payload.content[:40],
                    )
            self.start_task(
                RUN_TASK_NAME,
                self._stream_turn(
                    payload.content,
                    _user_image_attachments(payload.attachments),
                ),
            )
            return {
                "accepted": True,
                "mode": "started",
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
            }

        async def on_enqueue_message(
            self,
            payload: SendMessageRequest,
        ) -> dict[str, Any]:
            self.agent_app.enqueue_turn(
                payload.content,
                images=_user_image_attachments(payload.attachments),
            )
            return {
                "accepted": True,
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
                "queued_questions": list(self.agent_app.queued_questions),
                "queued_turns": jsonable_value(self.agent_app.queued_turns),
            }

        async def on_start_queued_message(
            self,
            payload: QueuedMessageRequest,
        ) -> dict[str, Any]:
            turn = self.agent_app.take_queued_turn(payload.index)
            self.start_task(RUN_TASK_NAME, self._stream_turn(turn.content, turn.images))
            return {
                "accepted": True,
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
                "queued_questions": list(self.agent_app.queued_questions),
                "queued_turns": jsonable_value(self.agent_app.queued_turns),
            }

        async def on_steer_message(
            self,
            payload: SendMessageRequest,
        ) -> dict[str, Any]:
            if not self.agent_app.active_thread_accepts_user_turn:
                if payload.attachments:
                    raise ValueError("Child thread steering does not support image input")
                self.agent_app.steer_active_child_thread(payload.content)
                return {
                    "accepted": True,
                    "mode": "steered",
                    "session_id": self.session_id,
                    "thread_id": self.agent_app.active_thread_id,
                }
            if self.agent_app.is_running_suspended:
                raise RuntimeError("Choose Approve or Reject before steering this run")
            await self.cancel_task(RUN_TASK_NAME)
            self.agent_app.cancel_active_turn()
            self.start_task(
                RUN_TASK_NAME,
                self._stream_turn(
                    payload.content,
                    _user_image_attachments(payload.attachments),
                ),
            )
            return {
                "accepted": True,
                "mode": "started",
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
            }

        async def on_steer_queued_message(
            self,
            payload: QueuedMessageRequest,
        ) -> dict[str, Any]:
            if self.agent_app.is_running_suspended:
                raise RuntimeError("Choose Approve or Reject before steering this run")
            turn = self.agent_app.take_queued_turn(payload.index)
            await self.cancel_task(RUN_TASK_NAME)
            self.agent_app.cancel_active_turn()
            self.start_task(RUN_TASK_NAME, self._stream_turn(turn.content, turn.images))
            return {
                "accepted": True,
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
                "queued_questions": list(self.agent_app.queued_questions),
                "queued_turns": jsonable_value(self.agent_app.queued_turns),
            }

        async def on_cancel_queued_message(
            self,
            payload: QueuedMessageRequest,
        ) -> dict[str, Any]:
            self.agent_app.cancel_queued_turn(payload.index)
            return {
                "cancelled": True,
                "session_id": self.session_id,
                "thread_id": self.agent_app.active_thread_id,
                "queued_questions": list(self.agent_app.queued_questions),
                "queued_turns": jsonable_value(self.agent_app.queued_turns),
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

        async def _stream_turn(
            self,
            content: str,
            attachments: tuple[UserImageAttachment, ...],
        ) -> None:
            async for event in self.agent_app.start_turn_events(
                content,
                images=attachments,
            ):
                await self.publish(_app_event_payload(event), event=APP_EVENT)

        async def _stream_approval(self, payload: ApprovalRequest) -> None:
            async for event in self.agent_app.approve_tool_events(
                thread_id=payload.thread_id,
                run_id=payload.run_id,
                tool_call_id=payload.tool_call_id,
            ):
                await self.publish(_app_event_payload(event), event=APP_EVENT)

        async def _stream_rejection(self, payload: RejectToolRequest) -> None:
            async for event in self.agent_app.reject_tool_events(
                payload.reason,
                thread_id=payload.thread_id,
                run_id=payload.run_id,
                tool_call_id=payload.tool_call_id,
            ):
                await self.publish(_app_event_payload(event), event=APP_EVENT)

    hub = SocketHub("/ws")
    hub.channel(AceAISessionChannel)

    sessions_route = Route("/api/sessions")

    @sessions_route.get
    def list_sessions() -> SessionsPayload:
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
        return SessionsPayload(
            sessions=[
                _session_list_item_response_payload(payload)
                for _metadata, payload in session_payloads
            ]
        )

    session_cleanup_route = Route("/api/session-cleanup/empty")

    @session_cleanup_route.delete
    def delete_empty_sessions() -> EmptySessionCleanupPayload:
        deleted_session_ids = delete_empty_session_ids(runtime.session_store)
        return EmptySessionCleanupPayload(
            deleted=len(deleted_session_ids),
            session_ids=deleted_session_ids,
        )

    session_route = Route("/api/sessions/{session_id}")

    @session_route.delete
    def delete_session(session_id: str) -> SessionDeletePayload:
        runtime.session_store.delete_session(session_id)
        return SessionDeletePayload(deleted=True, session_id=session_id)

    config_route = Route("/api/config")

    @config_route.get
    def read_config() -> GuiConfigPayload:
        return _gui_config_payload(runtime)

    @config_route.put
    def update_config(payload: GuiConfigUpdateRequest) -> GuiConfigPayload:
        current = runtime.config
        api_key = current.api_key if payload.api_key is None else payload.api_key
        api_keys = dict(current.api_keys)
        api_keys[payload.provider] = api_key
        next_config = AgentAppConfig(
            provider=payload.provider,
            api_key=api_key,
            model=payload.model,
            default_model=payload.default_model,
            skills=current.skills,
            skill_selection_mode=payload.skill_selection_mode,
            enabled_skills=payload.enabled_skills,
            api_keys=api_keys,
            tool_permissions=payload.tool_permissions,
            tool_enabled=payload.tool_enabled,
            tool_max_calls=payload.tool_max_calls,
            compress_threshold=payload.compress_threshold,
            reasoning_level=payload.reasoning_level,
            api_timeout_seconds=payload.api_timeout_seconds,
            stream_start_timeout_seconds=payload.stream_start_timeout_seconds,
            stream_event_timeout_seconds=payload.stream_event_timeout_seconds,
            disabled_providers=payload.disabled_providers,
        )
        runtime.update_config(next_config)
        return _gui_config_payload(runtime)

    references_route = Route("/api/references")

    @references_route.get
    def list_references(q: str = "", kind: str = "all") -> ReferencesPayload:
        return ReferencesPayload(
            items=[
                ReferenceItemPayload(**_reference_candidate_payload(candidate))
                for candidate in reference_candidates(
                    root=runtime.project_root,
                    ideas=runtime.list_ideas(),
                    query=q,
                    kind=kind,
                    limit=30,
                )
            ],
        )

    file_route = Route("/api/files")

    @file_route.get
    def read_file(path: str) -> FileResponsePayload:
        return FileResponsePayload(
            file=FilePayload(**project_file_payload(runtime.project_root, path))
        )

    @file_route.put
    def save_file(path: str, payload: FileSaveRequest) -> FileResponsePayload:
        target = project_file_path(runtime.project_root, path)
        target.write_text(payload.content)
        return FileResponsePayload(
            file=FilePayload(**project_file_payload(runtime.project_root, path))
        )

    ideas_route = Route("/api/ideas")

    @ideas_route.get
    def list_ideas() -> IdeasPayload:
        return IdeasPayload(
            ideas=[
                IdeaItemPayload(**idea_payload(idea, index))
                for index, idea in enumerate(runtime.list_ideas(), start=1)
            ],
        )

    @ideas_route.post
    def capture_idea(payload: IdeaCaptureRequest) -> IdeaResponsePayload:
        idea = runtime.capture_idea(payload.content)
        return IdeaResponsePayload(
            idea=IdeaItemPayload(
                **idea_payload(
                    idea,
                    idea_display_index(runtime.list_ideas(), idea.idea_id),
                )
            )
        )

    idea_route = Route("/api/ideas/{index}")

    @idea_route.put
    def update_idea(index: int, payload: IdeaUpdateRequest) -> IdeaResponsePayload:
        return IdeaResponsePayload(
            idea=IdeaItemPayload(
                **idea_payload(runtime.update_idea(index, payload.content), index)
            )
        )

    @idea_route.delete
    def delete_idea(index: int) -> IdeaDeletePayload:
        return IdeaDeletePayload(
            deleted=True,
            idea=IdeaItemPayload(**idea_payload(runtime.delete_idea(index), index)),
        )

    app = Lihil(
        hub,
        sessions_route,
        session_cleanup_route,
        session_route,
        config_route,
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


def _gui_config_payload(runtime: AceAIGuiRuntime) -> GuiConfigPayload:
    config = runtime.config
    provider_payloads = [
        ProviderOptionPayload(
            label=label,
            value=value,
            auth_mode=auth_mode(value),
            api_key_env=api_key_env(value),
        )
        for label, value in provider_options()
    ]
    model_payloads = [
        ModelOptionPayload(label=label, value=value)
        for label, value in model_options(config.provider)
    ]
    models_by_provider = {
        provider.value: [
            ModelOptionPayload(label=label, value=value)
            for label, value in model_options(provider.value)
        ]
        for provider in provider_payloads
    }
    reasoning_options = list(reasoning_effort_options(config.provider, config.model))
    if "auto" not in reasoning_options:
        reasoning_options.insert(0, "auto")
    return GuiConfigPayload(
        project_name=runtime.session_store.project.name,
        git_branch=_git_branch_name(runtime.project_root),
        provider=config.provider,
        model=config.model,
        default_model=config.default_model,
        reasoning_level=config.reasoning_level,
        compress_threshold=str(config.compress_threshold),
        api_timeout_seconds=config.api_timeout_seconds,
        stream_start_timeout_seconds=config.stream_start_timeout_seconds,
        stream_event_timeout_seconds=config.stream_event_timeout_seconds,
        skill_selection_mode=config.skill_selection_mode,
        enabled_skills=list(config.enabled_skills),
        disabled_providers=list(config.disabled_providers),
        api_key_set=config.api_key != "",
        api_key_env=api_key_env(config.provider),
        config_path=str(project_config_path(runtime.project_root)),
        providers=provider_payloads,
        models=model_payloads,
        models_by_provider=models_by_provider,
        reasoning_options=reasoning_options,
        skills=_skill_item_payloads(config),
        tools=_tool_permission_payloads(config),
    )


def _git_branch_name(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.removesuffix("\n")


def _skill_item_payloads(config: AgentAppConfig) -> list[SkillItemPayload]:
    registry = SkillLoader.load_registry(
        config.skills,
        extra_skill_paths=ACE_AGENT_BUILTIN_SKILL_PATHS,
    )
    enabled_names = set(config.enabled_skills)
    skills: list[SkillItemPayload] = []
    for skill in registry.get_skills():
        source = _skill_source(skill.skill_file)
        skills.append(
            SkillItemPayload(
                name=skill.name,
                description=skill.description,
                location=str(skill.skill_file),
                source=source,
                builtin=source == "aceai builtin",
                enabled=config.skill_selection_mode == "all" or skill.name in enabled_names,
            )
        )
    return skills


def _skill_source(skill_file: Path) -> str:
    resolved = skill_file.resolve()
    if any(
        resolved.is_relative_to(builtin_path.resolve())
        for builtin_path in ACE_AGENT_BUILTIN_SKILL_PATHS
    ):
        return "aceai builtin"
    if _is_under(skill_file, Path.cwd() / ".agents" / "skills"):
        return "project"
    if _is_under(skill_file, Path.home() / ".aceai" / "skills"):
        return "global"
    return "project"


def _is_under(path: Path, root: Path) -> bool:
    absolute_path = path.expanduser().absolute()
    absolute_root = root.expanduser().absolute()
    return absolute_path.is_relative_to(absolute_root)


def _tool_permission_payloads(config: AgentAppConfig) -> list[ToolPermissionPayload]:
    items: list[ToolPermissionPayload] = []
    for configured_tool in default_agent_tools():
        permission = config.tool_permissions.get(configured_tool.name)
        if permission is None:
            permission = "ask" if configured_tool.metadata.require_approval else "always"
        items.append(
            ToolPermissionPayload(
                name=configured_tool.name,
                description=configured_tool.description,
                permission=permission,
                enabled=config.tool_enabled.get(configured_tool.name, True),
                max_calls_per_run=config.tool_max_calls.get(configured_tool.name),
                tags=list(configured_tool.metadata.tags),
            )
        )
    return items


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


def _session_list_item_response_payload(
    payload: dict[str, Any],
) -> SessionListItemPayload:
    active_thread = payload["active_thread"]
    return SessionListItemPayload(
        session_id=payload["session_id"],
        project_id=payload["project_id"],
        project_name=payload["project_name"],
        title=payload["title"],
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
        event_count=payload["event_count"],
        total_cost_usd=payload["total_cost_usd"],
        thread_count=payload["thread_count"],
        active_thread=ThreadMetadataPayload(
            session_id=active_thread["session_id"],
            thread_id=active_thread["thread_id"],
            role=active_thread["role"],
            title=active_thread["title"],
            status=active_thread["status"],
            agent_id=active_thread["agent_id"],
            parent_thread_id=active_thread["parent_thread_id"],
            parent_run_id=active_thread["parent_run_id"],
            parent_tool_call_id=active_thread["parent_tool_call_id"],
            metadata=active_thread["metadata"],
            created_at=active_thread["created_at"],
            updated_at=active_thread["updated_at"],
        ),
        path=payload.get("path"),
    )


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
