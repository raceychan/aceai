"""Adapters between durable AceAI sessions and TUI display events."""

from aceai.agent.cost import CostEstimate
from aceai.agent.session import SessionEvent, SessionMessage
from aceai.core.helpers.string import uuid_str
from aceai.core.models import ToolExecutionResult
from aceai.llm.models import LLMToolCall, LLMUsage

from .events import TUIEvent, user_message_event


def tui_event_to_session_event(event: TUIEvent) -> SessionEvent:
    return SessionEvent(
        kind=event.kind,
        content=event.content,
        tool_name=event.tool_name,
        tool_call_id=event.tool_call_id,
        tool_call=event.tool_call,
        tool_result=event.tool_result,
        error=event.error,
        usage=event.usage,
        cost=event.cost,
    )


def session_messages_to_tui_events(messages: list[SessionMessage]) -> list[TUIEvent]:
    events: list[TUIEvent] = []
    for message in messages:
        if message.kind == "user":
            events.append(user_message_event(message.content))
        elif message.kind == "assistant":
            events.append(_assistant_message_to_tui_event(message))
        elif message.kind == "tool":
            events.append(_tool_message_to_tui_event(message))
        elif message.kind == "error":
            events.append(_error_message_to_tui_event(message))
    return events


def _assistant_message_to_tui_event(message: SessionMessage) -> TUIEvent:
    return TUIEvent(
        kind="assistant_delta",
        step_index=-1,
        step_id=uuid_str(),
        title="assistant",
        content=message.content,
        usage=_message_usage(message),
        cost=_message_cost(message),
        raw_event=None,
    )


def _tool_message_to_tui_event(message: SessionMessage) -> TUIEvent:
    call_id = message.tool_call_id or uuid_str()
    tool_name = message.tool_name or "tool"
    tool_call = LLMToolCall(
        name=tool_name,
        arguments=message.tool_arguments,
        call_id=call_id,
    )
    return TUIEvent(
        kind="tool_failed" if message.status == "failed" else "tool_completed",
        step_index=-1,
        step_id=uuid_str(),
        title=f"tool {tool_name}",
        content=message.tool_output,
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_call=tool_call,
        tool_result=ToolExecutionResult(
            call=tool_call,
            output=message.tool_output,
            error=message.content if message.status == "failed" else None,
        ),
        error=message.content if message.status == "failed" else None,
        raw_event=None,
    )


def _error_message_to_tui_event(message: SessionMessage) -> TUIEvent:
    return TUIEvent(
        kind="run_failed",
        step_index=-1,
        step_id=uuid_str(),
        title="run failed",
        content=message.content,
        error=message.content,
        raw_event=None,
    )


def _message_usage(message: SessionMessage) -> LLMUsage | None:
    if (
        message.usage_input_tokens is None
        and message.usage_output_tokens is None
        and message.usage_total_tokens is None
    ):
        return None
    return LLMUsage(
        input_tokens=message.usage_input_tokens,
        cached_input_tokens=message.usage_cached_input_tokens,
        output_tokens=message.usage_output_tokens,
        total_tokens=message.usage_total_tokens,
    )


def _message_cost(message: SessionMessage) -> CostEstimate | None:
    if (
        message.cost_model is None
        or message.cost_input_usd is None
        or message.cost_cached_input_usd is None
        or message.cost_output_usd is None
        or message.cost_total_usd is None
        or message.cost_input_usd_per_million is None
        or message.cost_cached_input_usd_per_million is None
        or message.cost_output_usd_per_million is None
        or message.cost_pricing_source is None
    ):
        return None
    return CostEstimate(
        model=message.cost_model,
        input_cost_usd=message.cost_input_usd,
        cached_input_cost_usd=message.cost_cached_input_usd,
        output_cost_usd=message.cost_output_usd,
        total_cost_usd=message.cost_total_usd,
        input_usd_per_million=message.cost_input_usd_per_million,
        cached_input_usd_per_million=message.cost_cached_input_usd_per_million,
        output_usd_per_million=message.cost_output_usd_per_million,
        pricing_source=message.cost_pricing_source,
    )
