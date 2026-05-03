"""Console entry point for the AceAI TUI."""

import argparse
import os
from typing import Sequence

from aceai.agent.ace_agent import build_ace_agent
from aceai.agent.session import (
    SessionMetadata,
    SessionRecorder,
    SessionStore,
    messages_to_llm_history,
)
from aceai.core import AgentBase
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel

from .config import (
    AceAITUIConfig,
    DEFAULT_OPENAI_MODEL,
    SUPPORTED_OPENAI_MODELS,
    load_config,
)
from .cost import format_usd
from .events import TUIEvent
from .runner import run_configured_tui, run_interactive_tui

OPENAI_MODELS: tuple[OpenAIModel, ...] = SUPPORTED_OPENAI_MODELS


def build_default_agent(*, api_key: str, model: OpenAIModel) -> AgentBase:
    return build_ace_agent(api_key=api_key, model=model)


def build_agent_from_config(config: AceAITUIConfig) -> AgentBase:
    if config.provider != "openai":
        raise ValueError("Unsupported provider")
    return build_default_agent(api_key=config.api_key, model=config.model)


def resolve_model(value: str) -> OpenAIModel:
    if value not in OPENAI_MODELS:
        raise ValueError("Unsupported OpenAI model")
    return value


def resolve_initial_config(model: OpenAIModel) -> AceAITUIConfig | None:
    if "OPENAI_API_KEY" in os.environ:
        return AceAITUIConfig(
            provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            model=model,
        )
    stored = load_config()
    if stored is None:
        return None
    if "ACEAI_MODEL" in os.environ:
        return AceAITUIConfig(
            provider=stored.provider,
            api_key=stored.api_key,
            model=model,
        )
    return stored


def create_session_context(
    *,
    resume_session_id: str | None,
) -> tuple[SessionStore, SessionMetadata, list[TUIEvent], list[LLMMessage]]:
    store = SessionStore()
    if resume_session_id is None:
        metadata = store.create_session()
        return store, metadata, [], []
    metadata = store.get_session(resume_session_id)
    messages = store.load_messages(resume_session_id)
    return (
        store,
        metadata,
        store.load_tui_events(resume_session_id),
        messages_to_llm_history(messages),
    )


def latest_session_id(store: SessionStore) -> str:
    sessions = store.list_sessions()
    if not sessions:
        raise ValueError("aceai resume found no sessions")
    return sessions[0].session_id


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aceai",
        description="Launch the AceAI terminal UI.",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Optional command: resume <session_id>, export <session_id>, or cost.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=OPENAI_MODELS,
        help="OpenAI model for the default AceAI CLI agent.",
    )
    args = parser.parse_args(argv)
    command_parts = list(args.command)
    if command_parts and command_parts[0] == "export":
        if len(command_parts) != 2:
            raise ValueError("aceai export requires a session_id")
        print(SessionStore().export_text(command_parts[1]), end="")
        return
    if command_parts and command_parts[0] == "cost":
        if len(command_parts) != 1:
            raise ValueError("aceai cost does not accept arguments")
        print(format_usd(SessionStore().total_cost_usd()))
        return
    resume_session_id: str | None = None
    if command_parts and command_parts[0] == "resume":
        if len(command_parts) > 2:
            raise ValueError("aceai resume requires a session_id")
        if len(command_parts) == 2:
            resume_session_id = command_parts[1]
        else:
            resume_session_id = latest_session_id(SessionStore())
    elif command_parts:
        raise ValueError(
            "aceai only accepts no arguments, resume <session_id>, "
            "export <session_id>, or cost"
        )
    selected_model = resolve_model(
        args.model or os.environ.get("ACEAI_MODEL", DEFAULT_OPENAI_MODEL)
    )
    config = resolve_initial_config(selected_model)
    store, metadata, initial_events, initial_history = create_session_context(
        resume_session_id=resume_session_id,
    )
    recorder = SessionRecorder(store, metadata.session_id)
    if config is None:
        run_configured_tui(
            build_agent_from_config,
            initial_config=None,
            initial_question="",
            default_model=selected_model,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=metadata.session_id,
        )
        if recorder.saved:
            print(f"Session saved: {metadata.session_id}")
        return
    agent = build_agent_from_config(config)
    run_interactive_tui(
        agent,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=recorder,
        session_id=metadata.session_id,
    )
    if recorder.saved:
        print(f"Session saved: {metadata.session_id}")
