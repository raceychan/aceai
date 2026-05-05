"""Live runner that bridges AgentBase events into the Textual app."""

import os
from typing import AsyncGenerator, Callable, cast

from opentelemetry.context import Context
from textual.widgets import Input
from textual.worker import Worker

from aceai.agent.provider_catalog import api_key_env, model_options, supported_models
from aceai.agent.session import SessionRecorder, SessionState
from aceai.agent.config import AceAITUIConfig, replace_config
from aceai.core import AgentBase, AgentRuntime, ToolApprovalDecision
from aceai.core.events import AgentEvent, RunCompletedEvent, RunSuspendedEvent
from aceai.core.executor import ToolExecutor
from aceai.core.skills import SkillLoader, SkillRegistry
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel
from aceai.llm.models import LLMRequestMeta

from .app import AceAITUI
from .events import TUIEvent
from .metadata import MetadataSection
from .setup import ConfigSelection, ConfigScreen, ProviderSetupScreen, SkillConfigItem
from .widgets import ApprovalWidget
from .widgets import CommandInput

AgentFactory = Callable[[AceAITUIConfig], AgentBase]


def _as_model(provider_name: str, model: str) -> OpenAIModel:
    if model not in supported_models(provider_name):
        raise ValueError("Unsupported model")
    return cast(OpenAIModel, model)


def _model_options_text(provider_name: str) -> str:
    model_names = ", ".join(option[1] for option in model_options(provider_name))
    return f"Available models: {model_names}"


def _model_from_request_meta(
    request_meta: LLMRequestMeta,
    default_model: str,
    provider_name: str,
) -> OpenAIModel:
    if "model" in request_meta:
        return _as_model(provider_name, request_meta["model"])
    return _as_model(provider_name, default_model)


def _skill_config_items(registry: SkillRegistry) -> tuple[SkillConfigItem, ...]:
    return tuple(
        SkillConfigItem(
            name=skill.name,
            description=skill.description,
            location=str(skill.skill_file),
        )
        for skill in registry.get_skills()
    )


def _system_prompt_text(agent: AgentBase) -> str:
    parts: list[str] = []
    for part in agent.system_message.content:
        if part["type"] == "text":
            parts.append(part["data"])
    return "".join(parts)


class _RuntimeStreamMixin:
    _active_runtime: AgentRuntime | None

    async def _stream_active_runtime(self) -> None:
        runtime = self._current_runtime()
        await self._consume_runtime_stream(runtime, runtime.execute())

    async def _stream_approval_decision(
        self,
        decision: ToolApprovalDecision,
    ) -> None:
        runtime = self._current_runtime()
        await self._consume_runtime_stream(runtime, runtime.resume_approval(decision))

    async def _consume_runtime_stream(
        self,
        runtime: AgentRuntime,
        stream: AsyncGenerator[AgentEvent, None],
    ) -> None:
        try:
            async for event in stream:
                self.append_agent_event(event)
                if isinstance(event, RunCompletedEvent):
                    self._finish_runtime_turn(runtime, event.final_answer)
                elif isinstance(event, RunSuspendedEvent):
                    self.show_pending_approval()
        finally:
            await stream.aclose()

    def _current_runtime(self) -> AgentRuntime:
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("AceAI runtime is not active")
        return runtime


class AceAIInteractiveTUI(_RuntimeStreamMixin, AceAITUI):
    """Textual app that runs an AgentBase from submitted questions."""

    def __init__(
        self,
        agent: AgentBase,
        *,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._request_meta: LLMRequestMeta = dict(request_meta or {})
        self._provider_name = "openai"
        self._selected_model = _model_from_request_meta(
            self._request_meta,
            agent.default_model,
            self._provider_name,
        )
        self._request_meta["model"] = self._selected_model
        super().__init__(
            events=initial_events or [],
            model=self._selected_model,
            session_recorder=session_recorder,
            session_id=session_id,
        )
        self._persist_session_state()
        self._agent = agent
        self._trace_ctx = trace_ctx
        self._llm_history = list(initial_history or [])
        self._active_worker: Worker[None] | None = None
        self._active_runtime: AgentRuntime | None = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, CommandInput):
            return
        question = event.value
        if question == "/quit":
            self.exit()
            return
        if question == "/clear":
            self.load_events([])
            self.exit_command_input(event.input)
            return
        if question == "/sessions":
            self.open_session_selector()
            self.exit_command_input(event.input)
            return
        if question in ("/metadata", "/info"):
            self.open_metadata_screen()
            self.exit_command_input(event.input)
            return
        if question == "/config":
            self.open_config_screen()
            self.exit_command_input(event.input)
            return
        if question.startswith("/resume "):
            self.switch_session(question.removeprefix("/resume "))
            self.exit_command_input(event.input)
            return
        if question == "/model":
            self.open_config_screen()
            self.exit_command_input(event.input)
            return
        if question.startswith("/model "):
            self.switch_model(question.removeprefix("/model "))
            self.exit_command_input(event.input)
            return
        if question == "":
            return
        self.start_run(question)
        self.exit_command_input(event.input)

    def start_run(self, question: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        if self._active_runtime is not None and self._active_runtime.status == "suspended":
            self.append_event(
                TUIEvent.session_notice("Choose Approve or Reject before starting another run.")
            )
            return
        self.ensure_session()
        self._persist_session_state()
        self.append_event(TUIEvent.user_message(question))
        self._active_runtime = self._agent.create_resume_run(
            question,
            self._llm_history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        self._active_worker = self.run_worker(
            self._stream_active_runtime(),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )

    def approve_pending_tool(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(ToolApprovalDecision.approve(request)),
            name="aceai-approval",
            description="Resume AceAI agent after tool approval",
            exit_on_error=True,
        )

    def show_pending_approval(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.show_approval_request(request)

    def on_approval_widget_selected(self, event: ApprovalWidget.Selected) -> None:
        event.stop()
        if event.approved:
            self.approve_pending_tool()
            return
        self.reject_pending_tool("rejected by caller")

    def reject_pending_tool(self, reason: str) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        if reason == "":
            reason = "rejected by caller"
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(
                ToolApprovalDecision.reject(request, reason=reason)
            ),
            name="aceai-approval",
            description="Resume AceAI agent after tool rejection",
            exit_on_error=True,
        )

    def _pending_approval_request(self):
        runtime = self._active_runtime
        if runtime is None:
            return None
        pending = runtime.run_state.pending_approval
        if pending is None:
            return None
        return pending.request

    def switch_session(self, session_id: str) -> None:
        previous_session_id = self._session_id
        super().switch_session(session_id)
        if self._session_id != previous_session_id:
            self._restore_session_state()
        self._reload_llm_history()

    def _finish_runtime_turn(self, runtime: AgentRuntime, answer: str) -> None:
        self._llm_history = list(runtime.context.context[1:])
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

    def show_model(self) -> None:
        self.append_event(
            TUIEvent.session_notice(
                f"Current model: {self._selected_model}\n{_model_options_text(self._provider_name)}"
            )
        )

    def action_config(self) -> None:
        self.open_config_screen()

    def open_config_screen(self) -> None:
        self.push_screen(
            ConfigScreen(
                provider_name=self._provider_name,
                current_model=self._selected_model,
                default_model=cast(OpenAIModel, self._agent.default_model),
                skills="auto",
                skill_items=_skill_config_items(self._agent.skill_registry),
                skill_selection_mode="all",
                enabled_skills=(),
                api_keys={},
                system_prompt=_system_prompt_text(self._agent),
            ),
            self._handle_config_selection,
        )

    def _handle_config_selection(self, selection: ConfigSelection | None) -> None:
        if selection is None:
            return
        if type(selection) is str:
            self.switch_model(selection)
            return
        if selection.provider != self._provider_name:
            self.append_event(
                TUIEvent.session_notice(
                    "Provider changes are only available in the configured AceAI app."
                )
            )
            return
        self.switch_model(selection.model)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                TUIEvent.session_notice("Model changes apply after the current run finishes.")
            )
            return
        if model not in supported_models(self._provider_name):
            self.append_event(
                TUIEvent.session_notice(_model_options_text(self._provider_name))
            )
            return
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta["model"] = self._selected_model
        self._persist_session_state()
        self.set_status_model(self._selected_model)
        self.append_event(TUIEvent.session_notice(f"Switched model to {self._selected_model}"))

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta: LLMRequestMeta = dict(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        event_log = self._session_recorder.store.load_event_log(self._session_id)
        self._llm_history = event_log.replay_llm_history()
        self._active_runtime = None

    def _persist_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        self._session_recorder.store.update_session_state(
            self._session_id,
            SessionState(
                selected_provider=self._provider_name,
                selected_model=self._selected_model,
            ),
        )

    def _restore_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        state = self._session_recorder.store.get_session_state(self._session_id)
        if state.selected_model == "":
            return
        if state.selected_provider != "" and state.selected_provider != self._provider_name:
            return
        if state.selected_model not in supported_models(self._provider_name):
            return
        self._selected_model = cast(OpenAIModel, state.selected_model)
        self._request_meta["model"] = self._selected_model
        self.set_status_model(self._selected_model)

    def _metadata_sections(self) -> list[MetadataSection]:
        return [
            *super()._metadata_sections(),
            *_agent_metadata_sections(
                self._agent,
                provider_name=self._provider_name,
                selected_model=self._selected_model,
            ),
        ]


class AceAIConfiguredTUI(_RuntimeStreamMixin, AceAITUI):
    """TUI that asks for provider settings before creating the agent."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        initial_config: AceAITUIConfig | None,
        initial_question: str,
        default_model: OpenAIModel,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._request_meta: LLMRequestMeta = dict(request_meta or {})
        self._provider_name = initial_config.provider if initial_config is not None else "openai"
        self._current_config = initial_config
        initial_model = (
            initial_config.model
            if initial_config is not None
            else _model_from_request_meta(
                self._request_meta,
                default_model,
                self._provider_name,
            )
        )
        self._request_meta["model"] = initial_model
        super().__init__(
            events=initial_events or [],
            model=initial_model,
            session_recorder=session_recorder,
            session_id=session_id,
        )
        self._agent_factory = agent_factory
        self._initial_config = initial_config
        self._initial_question = initial_question
        self._default_model: OpenAIModel = default_model
        self._trace_ctx = trace_ctx
        self._selected_model: OpenAIModel = initial_model
        self._llm_history = list(initial_history or [])
        self._agent: AgentBase | None = None
        self._active_worker: Worker[None] | None = None
        self._active_runtime: AgentRuntime | None = None

    def on_mount(self) -> None:
        super().on_mount()
        if self._initial_config is not None:
            self.apply_config(self._initial_config)
            return
        self.push_screen(
            ProviderSetupScreen(default_model=self._default_model),
            self._handle_setup_config,
        )

    def _handle_setup_config(self, config: AceAITUIConfig | None) -> None:
        if config is None:
            return
        self.apply_config(config)

    def apply_config(self, config: AceAITUIConfig) -> None:
        next_config = replace_config(config)
        self._provider_name = next_config.provider
        self._current_config = next_config
        self._agent = self._agent_factory(next_config)
        self._selected_model = next_config.model
        self._request_meta["model"] = self._selected_model
        self._persist_session_state()
        self.set_status_model(self._selected_model)
        if self._initial_question != "":
            self.start_run(self._initial_question)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, CommandInput):
            return
        question = event.value
        if question == "/quit":
            self.exit()
            return
        if question == "/clear":
            self.load_events([])
            self.exit_command_input(event.input)
            return
        if question == "/sessions":
            self.open_session_selector()
            self.exit_command_input(event.input)
            return
        if question in ("/metadata", "/info"):
            self.open_metadata_screen()
            self.exit_command_input(event.input)
            return
        if question == "/config":
            self.open_config_screen()
            self.exit_command_input(event.input)
            return
        if question.startswith("/resume "):
            self.switch_session(question.removeprefix("/resume "))
            self.exit_command_input(event.input)
            return
        if question == "/model":
            self.open_config_screen()
            self.exit_command_input(event.input)
            return
        if question.startswith("/model "):
            self.switch_model(question.removeprefix("/model "))
            self.exit_command_input(event.input)
            return
        if question == "":
            return
        self.start_run(question)
        self.exit_command_input(event.input)

    def start_run(self, question: str) -> None:
        if self._agent is None:
            self.query_one(CommandInput).value = question
            return
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        if self._active_runtime is not None and self._active_runtime.status == "suspended":
            self.append_event(
                TUIEvent.session_notice("Choose Approve or Reject before starting another run.")
            )
            return
        self.ensure_session()
        self._persist_session_state()
        self.append_event(TUIEvent.user_message(question))
        self._active_runtime = self._agent.create_resume_run(
            question,
            self._llm_history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        self._active_worker = self.run_worker(
            self._stream_active_runtime(),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )

    def switch_session(self, session_id: str) -> None:
        previous_session_id = self._session_id
        super().switch_session(session_id)
        if self._session_id != previous_session_id:
            self._restore_session_state()
        self._reload_llm_history()

    def _finish_runtime_turn(self, runtime: AgentRuntime, answer: str) -> None:
        self._llm_history = list(runtime.context.context[1:])
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

    def approve_pending_tool(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(ToolApprovalDecision.approve(request)),
            name="aceai-approval",
            description="Resume AceAI agent after tool approval",
            exit_on_error=True,
        )

    def show_pending_approval(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.show_approval_request(request)

    def on_approval_widget_selected(self, event: ApprovalWidget.Selected) -> None:
        event.stop()
        if event.approved:
            self.approve_pending_tool()
            return
        self.reject_pending_tool("rejected by caller")

    def reject_pending_tool(self, reason: str) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        if reason == "":
            reason = "rejected by caller"
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(
                ToolApprovalDecision.reject(request, reason=reason)
            ),
            name="aceai-approval",
            description="Resume AceAI agent after tool rejection",
            exit_on_error=True,
        )

    def _pending_approval_request(self):
        runtime = self._active_runtime
        if runtime is None:
            return None
        pending = runtime.run_state.pending_approval
        if pending is None:
            return None
        return pending.request

    def show_model(self) -> None:
        self.append_event(
            TUIEvent.session_notice(
                f"Current model: {self._selected_model}\n{_model_options_text(self._provider_name)}"
            )
        )

    def action_config(self) -> None:
        self.open_config_screen()

    def open_config_screen(self) -> None:
        api_keys: dict[str, str] = {}
        if self._current_config is not None:
            api_keys.update(self._current_config.api_keys)
            api_keys[self._current_config.provider] = self._current_config.api_key
        self.push_screen(
            ConfigScreen(
                provider_name=self._provider_name,
                current_model=self._selected_model,
                default_model=self._current_config.default_model
                if self._current_config is not None
                else self._selected_model,
                skills=self._current_config.skills if self._current_config is not None else "",
                skill_items=self._available_skill_items(),
                skill_selection_mode=self._current_config.skill_selection_mode
                if self._current_config is not None
                else "all",
                enabled_skills=tuple(self._current_config.enabled_skills)
                if self._current_config is not None
                else (),
                api_keys=api_keys,
                system_prompt=_system_prompt_text(self._agent),
            ),
            self._handle_config_selection,
        )

    def _handle_config_selection(self, selection: ConfigSelection | None) -> None:
        if selection is None:
            return
        if type(selection) is str:
            self.switch_model(selection)
            return
        if selection.provider == self._provider_name:
            if selection.api_key != "" or self._current_config is not None:
                api_keys = {}
                if self._current_config is not None:
                    api_keys.update(self._current_config.api_keys)
                api_key = (
                    selection.api_key
                    if selection.api_key != ""
                    else self._current_config.api_key
                )
                api_keys[selection.provider] = api_key
                self.apply_config(
                    AceAITUIConfig(
                        provider=selection.provider,
                        api_key=api_key,
                        model=selection.model,
                        default_model=selection.default_model,
                        skills=selection.skills,
                        skill_selection_mode=selection.skill_selection_mode,
                        enabled_skills=list(selection.enabled_skills),
                        api_keys=api_keys,
                    )
                )
                self.append_event(
                    TUIEvent.session_notice(
                        f"Updated provider credentials and switched model to {selection.model}"
                    )
                )
                return
            self.switch_model(selection.model)
            return
        api_key = selection.api_key
        if api_key == "":
            env_name = api_key_env(selection.provider)
            if env_name in os.environ:
                api_key = os.environ[env_name]
        if api_key == "":
            self.append_event(
                TUIEvent.session_notice(
                    f"API key required for provider {selection.provider}."
                )
            )
            return
        api_keys = {}
        if self._current_config is not None:
            api_keys.update(self._current_config.api_keys)
        api_keys[selection.provider] = api_key
        self.apply_config(
            AceAITUIConfig(
                provider=selection.provider,
                api_key=api_key,
                model=selection.model,
                default_model=selection.default_model,
                skills=selection.skills,
                skill_selection_mode=selection.skill_selection_mode,
                enabled_skills=list(selection.enabled_skills),
                api_keys=api_keys,
            )
        )
        self.append_event(
            TUIEvent.session_notice(
                f"Switched provider to {selection.provider} and model to {selection.model}"
            )
        )

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                TUIEvent.session_notice("Model changes apply after the current run finishes.")
            )
            return
        if model not in supported_models(self._provider_name):
            self.append_event(
                TUIEvent.session_notice(_model_options_text(self._provider_name))
            )
            return
        if self._current_config is not None:
            self._current_config = replace_config(
                AceAITUIConfig(
                    provider=self._current_config.provider,
                    api_key=self._current_config.api_key,
                    model=cast(OpenAIModel, model),
                    default_model=self._current_config.default_model,
                    skills=self._current_config.skills,
                    skill_selection_mode=self._current_config.skill_selection_mode,
                    enabled_skills=self._current_config.enabled_skills,
                    api_keys=self._current_config.api_keys,
                )
            )
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta["model"] = self._selected_model
        self._persist_session_state()
        self.set_status_model(self._selected_model)
        self.append_event(TUIEvent.session_notice(f"Switched model to {self._selected_model}"))

    def _available_skill_items(self) -> tuple[SkillConfigItem, ...]:
        if self._current_config is None:
            return _skill_config_items(self._agent.skill_registry)
        registry = SkillLoader.load_registry(self._current_config.skills)
        return _skill_config_items(registry)

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta: LLMRequestMeta = dict(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        event_log = self._session_recorder.store.load_event_log(self._session_id)
        self._llm_history = event_log.replay_llm_history()
        self._active_runtime = None

    def _persist_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        self._session_recorder.store.update_session_state(
            self._session_id,
            SessionState(
                selected_provider=self._provider_name,
                selected_model=self._selected_model,
            ),
        )

    def _restore_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        state = self._session_recorder.store.get_session_state(self._session_id)
        if state.selected_model == "":
            return
        if state.selected_provider == self._provider_name:
            self.switch_model(state.selected_model)

    def _metadata_sections(self) -> list[MetadataSection]:
        sections = super()._metadata_sections()
        if self._agent is None:
            return [
                *sections,
                MetadataSection(
                    title="Agent",
                    lines=[
                        f"provider: {self._provider_name}",
                        f"model: {self._selected_model}",
                        "configured: no",
                    ],
                ),
            ]
        return [
            *sections,
            *_agent_metadata_sections(
                self._agent,
                provider_name=self._provider_name,
                selected_model=self._selected_model,
            ),
        ]


class AceAILiveTUI(AceAIInteractiveTUI):
    """Textual app that starts a single run on mount."""

    def __init__(
        self,
        agent: AgentBase,
        question: str,
        *,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        super().__init__(
            agent,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=session_recorder,
            session_id=session_id,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
        )
        self._initial_question = question

    def on_mount(self) -> None:
        super().on_mount()
        self.start_run(self._initial_question)


def run_interactive_tui(
    agent: AgentBase,
    *,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAIInteractiveTUI(
        agent,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def run_configured_tui(
    agent_factory: AgentFactory,
    *,
    initial_config: AceAITUIConfig | None,
    initial_question: str,
    default_model: OpenAIModel,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAIConfiguredTUI(
        agent_factory,
        initial_config=initial_config,
        initial_question=initial_question,
        default_model=default_model,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def run_agent_tui(
    agent: AgentBase,
    question: str,
    *,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAILiveTUI(
        agent,
        question,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def _agent_metadata_sections(
    agent: AgentBase,
    *,
    provider_name: str,
    selected_model: str,
) -> list[MetadataSection]:
    skills = agent.skill_registry.get_skills()
    skill_lines = [
        f"{skill.name}: {skill.description} ({skill.skill_file})"
        for skill in skills
    ]
    executor = agent.executor
    tool_lines: list[str] = []
    if isinstance(executor, ToolExecutor):
        for tool in executor.tools.values():
            tags = ", ".join(tool.metadata.tags)
            tag_text = f" [{tags}]" if tags else ""
            tool_lines.append(f"{tool.name}{tag_text}: {tool.description}")
    hosted_lines = [
        f"{tool.provider_name}:{tool.native_name}"
        for tool in agent.hosted_tools
    ]
    return [
        MetadataSection(
            title="Agent",
            lines=[
                f"provider: {provider_name}",
                f"selected model: {selected_model}",
                f"default model: {agent.default_model}",
                f"max steps: {agent.max_steps}",
            ],
        ),
        MetadataSection(title=f"Skills ({len(skill_lines)})", lines=skill_lines),
        MetadataSection(title=f"Tools ({len(tool_lines)})", lines=tool_lines),
        MetadataSection(
            title=f"Hosted Tools ({len(hosted_lines)})",
            lines=hosted_lines,
        ),
    ]
