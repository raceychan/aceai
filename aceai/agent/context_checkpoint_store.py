import json
from pathlib import Path
from typing import Any, Literal, cast

from msgspec import Struct

from aceai.core.context_manager import estimate_message_tokens
from aceai.core.helpers.string import uuid_str
from aceai.llm.models import (
    LLMMessage,
    LLMMessagePart,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
)


CONTEXT_CHECKPOINT_VERSION = 2
ContextCheckpointReason = Literal["threshold", "context_window_retry"]


class ContextCheckpoint(Struct, frozen=True, kw_only=True):
    checkpoint_id: str
    session_id: str
    run_id: str
    step_id: str
    reason: ContextCheckpointReason
    compression_count: int
    included_event_id: str
    message_count: int
    estimated_tokens: int
    history: list[LLMMessage]
    units: list[dict[str, Any]]
    version: int = CONTEXT_CHECKPOINT_VERSION


class ContextCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def record_checkpoint(
        self,
        *,
        session_id: str,
        run_id: str,
        step_id: str,
        reason: ContextCheckpointReason,
        compression_count: int,
        included_event_id: str,
        history: list[LLMMessage],
    ) -> ContextCheckpoint:
        if included_event_id == "":
            raise ValueError("context checkpoint requires included_event_id")
        checkpoint = ContextCheckpoint(
            checkpoint_id=uuid_str(),
            session_id=session_id,
            run_id=run_id,
            step_id=step_id,
            reason=reason,
            compression_count=compression_count,
            included_event_id=included_event_id,
            message_count=len(history),
            estimated_tokens=estimate_message_tokens(history),
            history=list(history),
            units=context_units_payload_from_messages(history),
        )
        path = self._path_for(session_id)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(_checkpoint_to_payload(checkpoint), ensure_ascii=False)
            )
            stream.write("\n")
        return checkpoint

    def latest_checkpoint(self, session_id: str) -> ContextCheckpoint | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        latest: ContextCheckpoint | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "":
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("context checkpoint payload must be a mapping")
            if payload["version"] != CONTEXT_CHECKPOINT_VERSION:
                continue
            latest = _checkpoint_from_payload(payload)
        return latest

    def _path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.checkpoints.jsonl"


def llm_message_to_payload(message: LLMMessage) -> dict[str, Any]:
    if isinstance(message, LLMToolCallMessage):
        return {
            "message_type": "tool_call",
            "role": message.role,
            "content": _message_content_to_payload(message.content),
            "tool_calls": [call.asdict() for call in message.tool_calls],
            "reasoning_content": message.reasoning_content,
        }
    if isinstance(message, LLMToolUseMessage):
        return {
            "message_type": "tool_use",
            "role": message.role,
            "content": _message_content_to_payload(message.content),
            "name": message.name,
            "call_id": message.call_id,
        }
    return {
        "message_type": "message",
        "role": message.role,
        "content": _message_content_to_payload(message.content),
    }


def llm_message_from_payload(payload: dict[str, Any]) -> LLMMessage:
    message_type = payload["message_type"]
    content = _message_content_from_payload(payload["content"])
    if message_type == "message":
        return LLMMessage(role=payload["role"], content=content)
    if message_type == "tool_call":
        return LLMToolCallMessage.from_content(
            content=content,
            tool_calls=[
                LLMToolCall.from_payload(tool_call)
                for tool_call in _tool_calls_from_payload(payload["tool_calls"])
            ],
            reasoning_content=payload["reasoning_content"],
        )
    if message_type == "tool_use":
        return LLMToolUseMessage.from_content(
            content=content,
            name=payload["name"],
            call_id=payload["call_id"],
        )
    raise ValueError("Unsupported context checkpoint message_type")


def context_units_payload_from_messages(
    messages: list[LLMMessage],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "system":
            scope = _context_summary_scope(message)
            if scope == "prior_runs":
                units.append(
                    {
                        "type": "prior_run_summary",
                        "message": llm_message_to_payload(message),
                    }
                )
            elif scope == "current_run":
                units.append(
                    {
                        "type": "current_run_summary",
                        "message": llm_message_to_payload(message),
                    }
                )
            else:
                raise ValueError("checkpoint system message must be a context summary")
            index += 1
            continue
        if message.role == "user":
            units.append(
                {
                    "type": "current_user_message",
                    "message": llm_message_to_payload(message),
                }
            )
            index += 1
            continue
        unit_type = "step"
        step_messages: list[LLMMessage] = []
        if isinstance(message, LLMToolCallMessage):
            pending_call_ids = {call.call_id for call in message.tool_calls}
            step_messages.append(message)
            index += 1
            while index < len(messages):
                next_message = messages[index]
                if not isinstance(next_message, LLMToolUseMessage):
                    break
                if next_message.call_id not in pending_call_ids:
                    raise ValueError(
                        "checkpoint tool output has no matching tool call in step"
                    )
                step_messages.append(next_message)
                pending_call_ids.remove(next_message.call_id)
                index += 1
            if pending_call_ids:
                unit_type = "open_step"
        elif isinstance(message, LLMToolUseMessage):
            raise ValueError("checkpoint tool output has no matching tool call in step")
        else:
            step_messages.append(message)
            index += 1
        units.append(
            {
                "type": unit_type,
                "messages": [
                    llm_message_to_payload(step_message)
                    for step_message in step_messages
                ],
            }
        )
    return units


def messages_from_context_units_payload(
    units: list[dict[str, Any]],
) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    for unit in units:
        unit_type = unit["type"]
        if unit_type in (
            "prior_run_summary",
            "current_run_summary",
            "current_user_message",
        ):
            messages.append(llm_message_from_payload(unit["message"]))
        elif unit_type in ("step", "open_step"):
            unit_messages = unit["messages"]
            if type(unit_messages) is not list:
                raise TypeError("context checkpoint step messages must be list")
            messages.extend(
                llm_message_from_payload(message_payload)
                for message_payload in unit_messages
            )
        else:
            raise ValueError("Unsupported context checkpoint unit type")
    return messages


def _checkpoint_to_payload(checkpoint: ContextCheckpoint) -> dict[str, Any]:
    return {
        "version": checkpoint.version,
        "checkpoint_id": checkpoint.checkpoint_id,
        "session_id": checkpoint.session_id,
        "run_id": checkpoint.run_id,
        "step_id": checkpoint.step_id,
        "reason": checkpoint.reason,
        "compression_count": checkpoint.compression_count,
        "included_event_id": checkpoint.included_event_id,
        "message_count": checkpoint.message_count,
        "estimated_tokens": checkpoint.estimated_tokens,
        "units": checkpoint.units,
    }


def _checkpoint_from_payload(payload: dict[str, Any]) -> ContextCheckpoint:
    version = payload["version"]
    if version != CONTEXT_CHECKPOINT_VERSION:
        raise ValueError("Unsupported context checkpoint version")
    units_payload = payload["units"]
    if type(units_payload) is not list:
        raise TypeError("context checkpoint units must be list")
    units = _context_units_from_payload(units_payload)
    history = messages_from_context_units_payload(units)
    return ContextCheckpoint(
        version=version,
        checkpoint_id=payload["checkpoint_id"],
        session_id=payload["session_id"],
        run_id=payload["run_id"],
        step_id=payload["step_id"],
        reason=payload["reason"],
        compression_count=payload["compression_count"],
        included_event_id=payload["included_event_id"],
        message_count=payload["message_count"],
        estimated_tokens=payload["estimated_tokens"],
        history=history,
        units=units,
    )


def _message_content_to_payload(content: list[LLMMessagePart]) -> list[dict[str, Any]]:
    return [dict(part) for part in content]


def _message_content_from_payload(payload: Any) -> list[LLMMessagePart]:
    if type(payload) is not list:
        raise TypeError("context checkpoint message content must be list")
    content: list[LLMMessagePart] = []
    for part in payload:
        if type(part) is not dict:
            raise TypeError("context checkpoint message part must be mapping")
        content.append(cast(LLMMessagePart, part))
    return content


def _tool_calls_from_payload(payload: Any) -> list[dict[str, Any]]:
    if type(payload) is not list:
        raise TypeError("context checkpoint tool_calls must be list")
    tool_calls: list[dict[str, Any]] = []
    for tool_call in payload:
        if type(tool_call) is not dict:
            raise TypeError("context checkpoint tool_call must be mapping")
        tool_calls.append(tool_call)
    return tool_calls


def _context_units_from_payload(payload: list[Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for unit in payload:
        if type(unit) is not dict:
            raise TypeError("context checkpoint unit must be mapping")
        units.append(cast(dict[str, Any], unit))
    return units


def _context_summary_scope(message: LLMMessage) -> str:
    if len(message.content) != 1:
        return ""
    part = message.content[0]
    if part["type"] != "text" or "data" not in part:
        return ""
    text = part["data"]
    if text.startswith('<aceai_context_summary scope="prior_runs">'):
        return "prior_runs"
    if text.startswith('<aceai_context_summary scope="current_run">'):
        return "current_run"
    return ""
