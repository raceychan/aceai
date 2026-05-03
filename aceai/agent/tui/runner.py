"""Live runner that bridges AgentBase events into the Textual app."""

from typing import Callable, cast

from opentelemetry.context import Context
from textual.widgets import Input
from textual.worker import Worker

from aceai.agent.session import SessionRecorder, messages_to_llm_history
from aceai.core import AgentBase
from aceai.core.events import RunCompletedEvent
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel
from aceai.llm.models import LLMRequestMeta

from .app import AceAITUI
from .config import AceAITUIConfig, OPENAI_MODEL_OPTIONS, SUPPORTED_OPENAI_MODELS
from .events import TUIEvent, session_notice_event, user_message_event
from .setup import ModelSelectScreen, ProviderSetupScreen
from .widgets import CommandInput

AgentFactory = Callable[[AceAITUIConfig], AgentBase]


def _as_openai_model(model: str) -> OpenAIModel:
    if model not in SUPPORTED_OPENAI_MODELS:
        raise ValueError("Unsupported OpenAI model")
    return cast(OpenAIModel, model)


def _model_options_text() -> str:
    model_names = ", ".join(option[1] for option in OPENAI_MODEL_OPTIONS)
    return f"Available models: {model_names}"


def _model_from_request_meta(
    request_meta: LLMRequestMeta,
    default_model: str,
) -> OpenAIModel:
    if "model" in request_meta:
        return _as_openai_model(request_meta["model"])
    return _as_openai_model(default_model)


class AceAIInteractiveTUI(AceAITUI):
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
        self._selected_model = _model_from_request_meta(
            self._request_meta,
            agent.default_model,
        )
        self._request_meta["model"] = self._selected_model
        super().__init__(
            events=initial_events or [],
            model=self._selected_model,
            session_recorder=session_recorder,
            session_id=session_id,
        )
        self._agent = agent
        self._trace_ctx = trace_ctx
        self._llm_history = list(initial_history or [])
        self._active_worker: Worker[None] | None = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, CommandInput):
            return
        question = event.value
        if question == "/quit":
            self.exit()
            return
        if question == "/clear":
            self.load_events([])
            event.input.value = ""
            return
        if question == "/sessions":
            self.open_session_selector()
            event.input.value = ""
            return
        if question.startswith("/resume "):
            self.switch_session(question.removeprefix("/resume "))
            event.input.value = ""
            return
        if question == "/model":
            self.open_model_selector()
            event.input.value = ""
            return
        if question.startswith("/model "):
            self.switch_model(question.removeprefix("/model "))
            event.input.value = ""
            return
        if question == "":
            return
        self.start_run(question)
        event.input.value = ""

    def start_run(self, question: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        self.append_event(user_message_event(question))
        self._active_worker = self.run_worker(
            self._stream_agent_events(question),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )

    async def _stream_agent_events(self, question: str) -> None:
        stream = self._agent.resume(
            question,
            self._llm_history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        try:
            async for event in stream:
                self.append_agent_event(event)
                if isinstance(event, RunCompletedEvent):
                    self._append_history_turn(question, event.final_answer)
        finally:
            await stream.aclose()

    def switch_session(self, session_id: str) -> None:
        super().switch_session(session_id)
        self._reload_llm_history()

    def _append_history_turn(self, question: str, answer: str) -> None:
        self._llm_history.append(LLMMessage.build(role="user", content=question))
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

    def show_model(self) -> None:
        self.append_event(
            session_notice_event(
                f"Current model: {self._selected_model}\n{_model_options_text()}"
            )
        )

    def action_model_switcher(self) -> None:
        self.open_model_selector()

    def open_model_selector(self) -> None:
        self.push_screen(
            ModelSelectScreen(current_model=self._selected_model),
            self._handle_model_selection,
        )

    def _handle_model_selection(self, model: OpenAIModel | None) -> None:
        if model is None:
            return
        self.switch_model(model)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                session_notice_event("Model changes apply after the current run finishes.")
            )
            return
        if model not in SUPPORTED_OPENAI_MODELS:
            self.append_event(session_notice_event(_model_options_text()))
            return
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta["model"] = self._selected_model
        self.set_status_model(self._selected_model)
        self.append_event(session_notice_event(f"Switched model to {self._selected_model}"))

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta: LLMRequestMeta = dict(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        messages = self._session_recorder.store.load_messages(self._session_id)
        self._llm_history = messages_to_llm_history(messages)


class AceAIConfiguredTUI(AceAITUI):
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
        initial_model = (
            initial_config.model
            if initial_config is not None
            else _model_from_request_meta(self._request_meta, default_model)
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
        self._agent = self._agent_factory(config)
        self._selected_model = config.model
        self._request_meta["model"] = self._selected_model
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
            event.input.value = ""
            return
        if question == "/sessions":
            self.open_session_selector()
            event.input.value = ""
            return
        if question.startswith("/resume "):
            self.switch_session(question.removeprefix("/resume "))
            event.input.value = ""
            return
        if question == "/model":
            self.open_model_selector()
            event.input.value = ""
            return
        if question.startswith("/model "):
            self.switch_model(question.removeprefix("/model "))
            event.input.value = ""
            return
        if question == "":
            return
        self.start_run(question)
        event.input.value = ""

    def start_run(self, question: str) -> None:
        if self._agent is None:
            self.query_one(CommandInput).value = question
            return
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        self.append_event(user_message_event(question))
        self._active_worker = self.run_worker(
            self._stream_agent_events(question),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )

    async def _stream_agent_events(self, question: str) -> None:
        if self._agent is None:
            raise RuntimeError("AceAI agent is not configured")
        stream = self._agent.resume(
            question,
            self._llm_history,
            trace_ctx=self._trace_ctx,
            **self._request_meta_for_run(),
        )
        try:
            async for event in stream:
                self.append_agent_event(event)
                if isinstance(event, RunCompletedEvent):
                    self._append_history_turn(question, event.final_answer)
        finally:
            await stream.aclose()

    def switch_session(self, session_id: str) -> None:
        super().switch_session(session_id)
        self._reload_llm_history()

    def _append_history_turn(self, question: str, answer: str) -> None:
        self._llm_history.append(LLMMessage.build(role="user", content=question))
        self._llm_history.append(LLMMessage.build(role="assistant", content=answer))

    def show_model(self) -> None:
        self.append_event(
            session_notice_event(
                f"Current model: {self._selected_model}\n{_model_options_text()}"
            )
        )

    def action_model_switcher(self) -> None:
        self.open_model_selector()

    def open_model_selector(self) -> None:
        self.push_screen(
            ModelSelectScreen(current_model=self._selected_model),
            self._handle_model_selection,
        )

    def _handle_model_selection(self, model: OpenAIModel | None) -> None:
        if model is None:
            return
        self.switch_model(model)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                session_notice_event("Model changes apply after the current run finishes.")
            )
            return
        if model not in SUPPORTED_OPENAI_MODELS:
            self.append_event(session_notice_event(_model_options_text()))
            return
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta["model"] = self._selected_model
        self.set_status_model(self._selected_model)
        self.append_event(session_notice_event(f"Switched model to {self._selected_model}"))

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta: LLMRequestMeta = dict(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        messages = self._session_recorder.store.load_messages(self._session_id)
        self._llm_history = messages_to_llm_history(messages)


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
