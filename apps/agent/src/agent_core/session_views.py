"""App-layer payload projections shared by UI ports."""

from datetime import datetime
from pathlib import Path
from typing import Any

from msgspec import Struct

from agent_core.app import AceAgentApp
from agent_core.memory.ideas import Idea
from agent_core.session import MAIN_THREAD_ID, SessionEvent, SessionMetadata, SessionStore
from aceai.llm.interface import is_set
from aceai.llm.models import (
    LLMMessage,
    LLMResponse,
    LLMSegment,
    LLMToolCall,
    LLMToolCallDelta,
    LLMUsage,
)


def agent_snapshot_payload(
    app: AceAgentApp,
    *,
    after_event_id: str | None,
) -> dict[str, Any]:
    session_id = app.session_id
    if session_id is None:
        raise RuntimeError("AceAI session is not active")
    store = app.session_service.store
    snapshot = app.session_service.snapshot_thread(session_id, app.active_thread_id)
    events = snapshot.event_log.events
    if after_event_id is not None:
        events = events_after(events, after_event_id)
    return {
        "session": session_metadata_payload(snapshot.metadata),
        "state": snapshot.state.as_json(),
        "runtime": agent_runtime_payload(app),
        "observability": observability_payload(events),
        "active_thread_id": app.active_thread_id,
        "threads": [
            thread_metadata_payload(thread)
            for thread in store.list_threads(session_id)
        ],
        "events": [event.as_json() for event in events],
    }


def agent_runtime_payload(app: AceAgentApp) -> dict[str, Any]:
    pending_approval = app.pending_approval_request()
    active_run = app.active_run
    return {
        "queued_questions": list(app.queued_questions),
        "queued_turns": jsonable_value(app.queued_turns),
        "pending_approval": (
            jsonable_value(pending_approval)
            if pending_approval is not None
            else None
        ),
        "is_running_suspended": app.is_running_suspended,
        "active_thread_accepts_user_turn": app.active_thread_accepts_user_turn,
        "active_run_id": active_run.run_id if active_run is not None else None,
        "active_run_status": active_run.status if active_run is not None else None,
        "provider_name": app.provider_name,
        "selected_model": app.selected_model,
        "reasoning_level": app.reasoning_level,
    }


def observability_payload(events: list[SessionEvent]) -> dict[str, Any]:
    return {
        "usage": usage_summary_payload(events),
        "event_counts": event_count_payload(events),
        "tool_calls": tool_call_stats_payload(events),
        "trajectory": trajectory_payload(events),
        "debug_events": debug_event_payload(events),
    }


def usage_summary_payload(events: list[SessionEvent]) -> dict[str, Any]:
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    latest_context_tokens: int | None = None
    latest_cache_hit_rate: float | None = None
    for event in events:
        usage = event.payload.get("usage")
        if isinstance(usage, dict):
            input_tokens += int_metric(usage, "input_tokens")
            cached_input_tokens += int_metric(usage, "cached_input_tokens")
            output_tokens += int_metric(usage, "output_tokens")
            total_tokens += int_metric(usage, "total_tokens")
            latest_context_tokens = int_metric(usage, "input_tokens")
            latest_cache_hit_rate = float_metric(usage, "input_cache_hit_rate")
        cost = event.payload.get("cost")
        if isinstance(cost, dict):
            total_cost_usd += float_metric(cost, "total_cost_usd") or 0.0
    return {
        "context_tokens": latest_context_tokens,
        "cache_hit_rate": latest_cache_hit_rate,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_usd,
    }


def event_count_payload(events: list[SessionEvent]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return [
        {"kind": kind, "count": counts[kind]}
        for kind in sorted(counts, key=lambda item: (-counts[item], item))
    ]


def tool_call_stats_payload(events: list[SessionEvent]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for event in events:
        tool_name = event.payload.get("tool_name")
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_name) is not str or type(tool_call_id) is not str:
            continue
        current = stats.get(tool_name)
        if current is None:
            current = {
                "name": tool_name,
                "call_ids": set[str](),
                "succeeded": 0,
                "failed": 0,
                "approval_requests": 0,
            }
            stats[tool_name] = current
        current["call_ids"].add(tool_call_id)
        if event.kind == "tool_completed" or event.kind == "tool_result":
            current["succeeded"] += 1
        if event.kind == "tool_failed":
            current["failed"] += 1
        if event.kind == "tool_approval_requested":
            current["approval_requests"] += 1
    return [
        {
            "name": item["name"],
            "calls": len(item["call_ids"]),
            "succeeded": item["succeeded"],
            "failed": item["failed"],
            "approval_requests": item["approval_requests"],
        }
        for item in sorted(stats.values(), key=lambda stat: stat["name"])
    ]


def trajectory_payload(events: list[SessionEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "thread_id": event.thread_id,
            "run_id": event.run_id,
            "created_at": event_created_at(event),
            "summary": event_summary(event),
        }
        for event in events[-80:]
    ]


def debug_event_payload(events: list[SessionEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "thread_id": event.thread_id,
            "run_id": event.run_id,
            "payload": event.payload,
            "created_at": event_created_at(event),
        }
        for event in events[-24:]
    ]


def event_created_at(event: SessionEvent) -> str:
    if type(event.created_at) is str:
        return event.created_at
    if type(event.created_at) is datetime:
        return event.created_at.isoformat()
    raise TypeError("SessionEvent created_at must be str or datetime")


def event_summary(event: SessionEvent) -> str:
    content = event.payload.get("content")
    if type(content) is str and content != "":
        return short_text(content)
    tool_name = event.payload.get("tool_name")
    if type(tool_name) is str:
        return tool_name
    return event.kind


def short_text(value: str) -> str:
    if len(value) <= 120:
        return value
    return value[:117] + "..."


def int_metric(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if type(value) is int:
        return value
    return 0


def float_metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if type(value) is int or type(value) is float:
        return float(value)
    return None


def events_after(events: list[SessionEvent], event_id: str) -> list[SessionEvent]:
    for index, event in enumerate(events):
        if event.event_id == event_id:
            return events[index + 1 :]
    return events


def session_metadata_payload(metadata: SessionMetadata) -> dict[str, Any]:
    return {
        "session_id": metadata.session_id,
        "project_id": metadata.project_id,
        "project_name": metadata.project_name,
        "created_at": metadata.created_at.isoformat(),
        "updated_at": metadata.updated_at.isoformat(),
        "title": metadata.title,
        "path": metadata.path,
    }


def session_list_item_payload(
    store: SessionStore,
    metadata: SessionMetadata,
) -> dict[str, Any]:
    threads = store.list_threads(metadata.session_id)
    event_count = len(store.load_event_log(metadata.session_id).events)
    active_thread = next(
        thread for thread in threads if thread.thread_id == MAIN_THREAD_ID
    )
    return {
        **session_metadata_payload(metadata),
        "event_count": event_count,
        "total_cost_usd": runtime_session_cost(store, metadata.session_id),
        "thread_count": len(threads),
        "active_thread": thread_metadata_payload(active_thread),
    }


def delete_empty_sessions(store: SessionStore) -> list[str]:
    deleted_session_ids: list[str] = []
    for metadata in store.list_sessions():
        if len(store.load_event_log(metadata.session_id).events) != 0:
            continue
        store.delete_session(metadata.session_id)
        deleted_session_ids.append(metadata.session_id)
    return deleted_session_ids


def runtime_session_cost(store: SessionStore, session_id: str) -> float:
    return store.load_event_log(session_id).total_cost_usd()


def thread_metadata_payload(thread) -> dict[str, Any]:
    return {
        "session_id": thread.session_id,
        "thread_id": thread.thread_id,
        "agent_id": thread.agent_id,
        "role": thread.role,
        "title": thread.title,
        "status": thread.status,
        "parent_thread_id": thread.parent_thread_id,
        "parent_run_id": thread.parent_run_id,
        "parent_tool_call_id": thread.parent_tool_call_id,
        "metadata": thread.metadata,
        "created_at": thread.created_at.isoformat(),
        "updated_at": thread.updated_at.isoformat(),
    }


def idea_payload(idea: Idea, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "idea_id": idea.idea_id,
        "created_at": idea.created_at.isoformat(),
        "project_id": idea.project_id,
        "project_name": idea.project_name,
        "workspace": idea.workspace,
        "content": idea.content,
        "source_session_id": idea.source_session_id,
    }


def idea_display_index(ideas: list[Idea], idea_id: str) -> int:
    for index, idea in enumerate(ideas, start=1):
        if idea.idea_id == idea_id:
            return index
    raise IndexError("Idea is not visible in display list")


def project_file_payload(root: Path, path: str) -> dict[str, Any]:
    target = project_file_path(root, path)
    stat = target.stat()
    return {
        "path": target.relative_to(root).as_posix(),
        "content": target.read_text(),
        "size": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def project_file_path(root: Path, path: str) -> Path:
    target = (root / path.removeprefix("@")).resolve()
    target.relative_to(root)
    if not target.is_file():
        raise FileNotFoundError(path)
    return target


def jsonable_value(value: Any) -> Any:
    if isinstance(value, SessionEvent):
        return value.as_json()
    if isinstance(value, LLMMessage):
        return value.asdict()
    if isinstance(value, LLMToolCall):
        return value.asdict()
    if isinstance(value, LLMToolCallDelta):
        return value.asdict()
    if isinstance(value, LLMResponse):
        return {
            "model": value.model,
            "text": value.text,
            "tool_calls": [call.asdict() for call in value.tool_calls],
            "usage": jsonable_value(value.usage) if is_set(value.usage) else None,
            "provider_meta": [
                meta.asdict() for meta in value.provider_meta
            ],
        }
    if isinstance(value, LLMUsage):
        return value.asdict()
    if isinstance(value, LLMSegment):
        return value.asdict()
    if isinstance(value, Struct):
        return {
            name: jsonable_value(getattr(value, name))
            for name in value.__struct_fields__
        }
    if isinstance(value, list):
        return [jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: jsonable_value(item)
            for key, item in value.items()
        }
    return value
