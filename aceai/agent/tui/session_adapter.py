"""Adapters from TUI events into durable AceAI session events."""

from typing import Any

from aceai.agent.session import SessionEvent

from .events import TUIEvent


def tui_event_to_session_event(event: TUIEvent) -> SessionEvent:
    return SessionEvent(
        run_id="",
        step_id=event.step_id,
        step_index=event.step_index,
        kind=event.kind,
        payload=_payload_for_event(event),
    )


def _payload_for_event(event: TUIEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {"content": event.content}
    if event.tool_name is not None:
        payload["tool_name"] = event.tool_name
    if event.tool_call_id is not None:
        payload["tool_call_id"] = event.tool_call_id
    if event.tool_call is not None:
        payload["tool_call"] = event.tool_call.asdict()
    if event.tool_calls:
        payload["tool_calls"] = [call.asdict() for call in event.tool_calls]
    if event.tool_result is not None:
        payload["tool_result"] = {
            "output": event.tool_result.output,
            "error": event.tool_result.error,
        }
    if event.error is not None:
        payload["error"] = event.error
    if event.usage is not None:
        payload["usage"] = {
            "input_tokens": event.usage.input_tokens,
            "cached_input_tokens": event.usage.cached_input_tokens,
            "cache_miss_input_tokens": event.usage.cache_miss_input_tokens,
            "input_cache_hit_rate": event.usage.input_cache_hit_rate,
            "output_tokens": event.usage.output_tokens,
            "total_tokens": event.usage.total_tokens,
        }
    if event.cost is not None:
        payload["cost"] = event.cost.asdict()
    return payload
