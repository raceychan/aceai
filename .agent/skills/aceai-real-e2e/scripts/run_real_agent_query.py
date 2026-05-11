import argparse
import asyncio
from pathlib import Path

from aceai.agent.app import AceAgentApp
from aceai.agent.config import load_config
from aceai.agent.session import SessionEvent, SessionStore
from aceai.agent.tui.cli import build_agent
from aceai.core.events import (
    RunCompletedEvent,
    RunFailedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ToolStartedEvent,
)

WATCHED_TOOLS = {
    "cancel_subagent",
    "check_subagent",
    "collect_subagent_results",
    "delegate_to_subagent",
    "spawn_subagent",
    "wait_subagent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real AceAI query through AceAgentApp and print key e2e events."
    )
    parser.add_argument("--query", default="", help="Question to run through AceAI.")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="File containing the question to run through AceAI.",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=None,
        help="Optional session root. Defaults to AceAI's normal session store.",
    )
    return parser.parse_args()


def read_query(args: argparse.Namespace) -> str:
    if args.query_file is not None:
        return args.query_file.read_text()
    if args.query:
        return args.query
    raise RuntimeError("provide --query or --query-file")


def session_content(event: SessionEvent) -> str:
    if "content" in event.payload:
        return event.payload["content"]
    if "message" in event.payload:
        return event.payload["message"]
    return repr(event.payload)


def one_line(text: str, limit: int) -> str:
    return text.replace("\n", " ")[:limit]


async def run_query(query: str, session_root: Path | None) -> None:
    config = load_config()
    if config is None:
        raise RuntimeError("AceAI config is missing")
    agent = build_agent(config)
    model = config.model or config.default_model
    store = SessionStore(session_root) if session_root is not None else SessionStore()
    app = AceAgentApp(
        agent,
        provider_name=config.provider,
        selected_model=model,
        reasoning_level=config.reasoning_level,
        session_store=store,
    )

    seen_tools: list[str] = []
    inbox_items: list[str] = []
    delivered_items: list[str] = []
    failures: list[str] = []
    completed_threads: list[str] = []

    print(
        f"CONFIG provider={config.provider} model={model} reasoning={config.reasoning_level}",
        flush=True,
    )
    async for app_event in app.start_turn_events(query):
        event = app_event.event
        thread_id = app_event.thread_id
        if isinstance(event, SessionEvent):
            if event.kind == "agent_inbox_item":
                content = session_content(event)
                inbox_items.append(content)
                print(f"INBOX_ITEM thread={thread_id} {one_line(content, 320)}", flush=True)
            elif event.kind == "agent_inbox_delivered":
                content = session_content(event)
                delivered_items.append(content)
                print(
                    f"INBOX_DELIVERED thread={thread_id} {one_line(content, 200)}",
                    flush=True,
                )
            elif event.kind in {"run_failed", "tool_failed", "error"}:
                content = session_content(event)
                failures.append(content)
                print(
                    f"SESSION_FAILED thread={thread_id} kind={event.kind} {one_line(content, 300)}",
                    flush=True,
                )
            continue
        if isinstance(event, ToolStartedEvent):
            seen_tools.append(event.tool_name)
            if event.tool_name in WATCHED_TOOLS:
                print(f"TOOL_START thread={thread_id} {event.tool_name}", flush=True)
        elif isinstance(event, ToolCompletedEvent):
            if event.tool_name in WATCHED_TOOLS:
                print(
                    f"TOOL_DONE thread={thread_id} {event.tool_name} result={one_line(event.tool_result.output, 420)}",
                    flush=True,
                )
        elif isinstance(event, ToolFailedEvent):
            failures.append(event.error)
            print(
                f"TOOL_FAILED thread={thread_id} {event.tool_name} {one_line(event.error, 300)}",
                flush=True,
            )
        elif isinstance(event, RunFailedEvent):
            failures.append(event.error)
            print(f"RUN_FAILED thread={thread_id} {one_line(event.error, 400)}", flush=True)
        elif isinstance(event, RunCompletedEvent):
            completed_threads.append(thread_id)
            print(
                f"RUN_COMPLETED thread={thread_id} final={one_line(event.final_answer, 1000)}",
                flush=True,
            )

    print("SUMMARY_TOOLS " + ",".join(seen_tools), flush=True)
    print(f"SUMMARY_INBOX_COUNT {len(inbox_items)}", flush=True)
    print(f"SUMMARY_DELIVERED_COUNT {len(delivered_items)}", flush=True)
    print(f"SUMMARY_FAILED_COUNT {len(failures)}", flush=True)
    print("SUMMARY_COMPLETED_THREADS " + ",".join(completed_threads), flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(run_query(read_query(args), args.session_root))


if __name__ == "__main__":
    main()
