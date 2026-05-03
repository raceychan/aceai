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

from .config import AceAITUIConfig, load_config
from .events import TUIEvent
from .runner import run_agent_tui, run_configured_tui, run_interactive_tui

OPENAI_MODELS: tuple[OpenAIModel, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5o",
    "gpt-5o-mini",
    "gpt-5.1",
    "o3-large",
    "o4-mini",
)


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aceai",
        description="Launch the AceAI terminal UI.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional question to run immediately. Omit for interactive mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=OPENAI_MODELS,
        help="OpenAI model for the default AceAI CLI agent.",
    )
    args = parser.parse_args(argv)
    question_parts = list(args.question)
    if question_parts and question_parts[0] == "export":
        if len(question_parts) != 2:
            raise ValueError("aceai export requires a session_id")
        print(SessionStore().export_text(question_parts[1]), end="")
        return
    resume_session_id: str | None = None
    if question_parts and question_parts[0] == "resume":
        if len(question_parts) < 2:
            raise ValueError("aceai resume requires a session_id")
        resume_session_id = question_parts[1]
        question_parts = question_parts[2:]
    selected_model = resolve_model(args.model or os.environ.get("ACEAI_MODEL", "gpt-5.1"))
    config = resolve_initial_config(selected_model)
    question = " ".join(question_parts)
    store, metadata, initial_events, initial_history = create_session_context(
        resume_session_id=resume_session_id,
    )
    recorder = SessionRecorder(store, metadata.session_id)
    if config is None:
        run_configured_tui(
            build_agent_from_config,
            initial_config=None,
            initial_question=question,
            default_model=selected_model,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=metadata.session_id,
        )
        print(f"Session saved: {metadata.session_id}")
        return
    agent = build_agent_from_config(config)
    if question == "":
        run_interactive_tui(
            agent,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=metadata.session_id,
        )
    else:
        run_agent_tui(
            agent,
            question,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=metadata.session_id,
        )
    print(f"Session saved: {metadata.session_id}")
