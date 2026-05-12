"""Derived tool-call statistics for the TUI stats view."""

import json
from pathlib import Path

from msgspec import Struct

from .events import TUIEvent


class ToolCallStat(Struct, frozen=True, kw_only=True):
    name: str
    calls: int = 0
    succeeded: int = 0
    failed: int = 0


class SkillCallStat(Struct, frozen=True, kw_only=True):
    name: str
    calls: int = 0
    succeeded: int = 0
    failed: int = 0


class _MutableToolCallStat(Struct, kw_only=True):
    name: str
    call_ids: set[str]
    succeeded: int = 0
    failed: int = 0


def tool_call_stats(
    events: list[TUIEvent],
    *,
    artifact_root: Path | None = None,
) -> list[ToolCallStat]:
    stats: dict[str, _MutableToolCallStat] = {}
    for event in events:
        if event.tool_name is None:
            continue
        if event.tool_call_id is None:
            continue
        _apply_parent_tool_event(stats, event)
        _apply_child_tool_stats(stats, event, artifact_root=artifact_root)
    return [
        ToolCallStat(
            name=stat.name,
            calls=len(stat.call_ids),
            succeeded=stat.succeeded,
            failed=stat.failed,
        )
        for stat in sorted(stats.values(), key=lambda item: item.name)
    ]


def format_tool_call_stats(stats: list[ToolCallStat]) -> list[str]:
    return [
        f"{stat.name}: calls {stat.calls}  ok {stat.succeeded}  failed {stat.failed}"
        for stat in stats
    ]


def merge_tool_call_stats(stats_groups: list[list[ToolCallStat]]) -> list[ToolCallStat]:
    merged: dict[str, ToolCallStat] = {}
    for stats in stats_groups:
        for stat in stats:
            current = merged.get(stat.name)
            if current is None:
                merged[stat.name] = stat
            else:
                merged[stat.name] = ToolCallStat(
                    name=stat.name,
                    calls=current.calls + stat.calls,
                    succeeded=current.succeeded + stat.succeeded,
                    failed=current.failed + stat.failed,
                )
    return [merged[name] for name in sorted(merged)]


def skill_call_stats(
    events: list[TUIEvent],
    *,
    artifact_root: Path | None = None,
) -> list[SkillCallStat]:
    stats: dict[str, _MutableToolCallStat] = {}
    for event in events:
        _apply_parent_skill_event(stats, event)
        _apply_child_skill_stats(stats, event, artifact_root=artifact_root)
    return [
        SkillCallStat(
            name=stat.name,
            calls=len(stat.call_ids),
            succeeded=stat.succeeded,
            failed=stat.failed,
        )
        for stat in sorted(stats.values(), key=lambda item: item.name)
    ]


def format_skill_call_stats(stats: list[SkillCallStat]) -> list[str]:
    return [
        f"{stat.name}: calls {stat.calls}  ok {stat.succeeded}  failed {stat.failed}"
        for stat in stats
    ]


def merge_skill_call_stats(stats_groups: list[list[SkillCallStat]]) -> list[SkillCallStat]:
    merged: dict[str, SkillCallStat] = {}
    for stats in stats_groups:
        for stat in stats:
            current = merged.get(stat.name)
            if current is None:
                merged[stat.name] = stat
            else:
                merged[stat.name] = SkillCallStat(
                    name=stat.name,
                    calls=current.calls + stat.calls,
                    succeeded=current.succeeded + stat.succeeded,
                    failed=current.failed + stat.failed,
                )
    return [merged[name] for name in sorted(merged)]


def _apply_parent_tool_event(
    stats: dict[str, _MutableToolCallStat],
    event: TUIEvent,
) -> None:
    if event.tool_name is None:
        raise ValueError("tool event has no tool name")
    if event.tool_call_id is None:
        raise ValueError("tool event has no tool call id")
    if event.kind not in ("tool_started", "tool_completed", "tool_failed"):
        return
    stat = _tool_stat(stats, event.tool_name)
    stat.call_ids.add(event.tool_call_id)
    if event.kind == "tool_completed":
        stat.succeeded += 1
    if event.kind == "tool_failed":
        stat.failed += 1


def _apply_parent_skill_event(
    stats: dict[str, _MutableToolCallStat],
    event: TUIEvent,
) -> None:
    if event.kind not in ("tool_completed", "tool_failed"):
        return
    if event.tool_name != "skill_view":
        return
    if event.tool_call_id is None:
        return
    if event.tool_call is None:
        return
    skill_name = _skill_name_from_arguments(event.tool_call.arguments)
    stat = _tool_stat(stats, skill_name)
    stat.call_ids.add(event.tool_call_id)
    if event.kind == "tool_completed":
        stat.succeeded += 1
    if event.kind == "tool_failed":
        stat.failed += 1


def _apply_child_tool_stats(
    stats: dict[str, _MutableToolCallStat],
    event: TUIEvent,
    *,
    artifact_root: Path | None,
) -> None:
    if event.kind != "tool_completed":
        return
    if event.tool_name != "delegate_to_subagent":
        return
    payload = json.loads(event.content)
    if payload.get("type") == "subagent_audit":
        _apply_child_tool_stats_from_audit(
            stats,
            payload,
            parent_call_id=event.tool_call_id,
            artifact_root=artifact_root,
        )
        return
    _apply_child_tool_stats_from_results(
        stats,
        payload["tool_results"],
        parent_call_id=event.tool_call_id,
    )


def _apply_child_tool_stats_from_results(
    stats: dict[str, _MutableToolCallStat],
    child_tool_results: list[dict],
    *,
    parent_call_id: str | None,
) -> None:
    for result in child_tool_results:
        stat = _tool_stat(stats, result["tool_name"])
        stat.call_ids.add(_child_call_id(parent_call_id, result["call_id"]))
        if result["error"] is None:
            stat.succeeded += 1
        else:
            stat.failed += 1


def _apply_child_tool_stats_from_audit(
    stats: dict[str, _MutableToolCallStat],
    payload: dict,
    *,
    parent_call_id: str | None,
    artifact_root: Path | None,
) -> None:
    if artifact_root is None:
        return
    manifest = json.loads((artifact_root / payload["manifest_path"]).read_text())
    for result in manifest["tool_results"]:
        stat = _tool_stat(stats, result["tool_name"])
        stat.call_ids.add(_child_call_id(parent_call_id, result["tool_call_id"]))
        if result["has_error"]:
            stat.failed += 1
        else:
            stat.succeeded += 1


def _apply_child_skill_stats(
    stats: dict[str, _MutableToolCallStat],
    event: TUIEvent,
    *,
    artifact_root: Path | None,
) -> None:
    if event.kind != "tool_completed":
        return
    if event.tool_name != "delegate_to_subagent":
        return
    payload = json.loads(event.content)
    if payload.get("type") == "subagent_audit":
        _apply_child_skill_stats_from_audit(
            stats,
            payload,
            parent_call_id=event.tool_call_id,
            artifact_root=artifact_root,
        )
        return
    _apply_child_skill_stats_from_results(
        stats,
        payload["tool_results"],
        parent_call_id=event.tool_call_id,
    )


def _apply_child_skill_stats_from_results(
    stats: dict[str, _MutableToolCallStat],
    child_tool_results: list[dict],
    *,
    parent_call_id: str | None,
) -> None:
    for result in child_tool_results:
        if result["tool_name"] != "skill_view":
            continue
        stat = _tool_stat(stats, _skill_name_from_arguments(result["arguments"]))
        stat.call_ids.add(_child_call_id(parent_call_id, result["call_id"]))
        if result["error"] is None:
            stat.succeeded += 1
        else:
            stat.failed += 1


def _apply_child_skill_stats_from_audit(
    stats: dict[str, _MutableToolCallStat],
    payload: dict,
    *,
    parent_call_id: str | None,
    artifact_root: Path | None,
) -> None:
    if artifact_root is None:
        return
    manifest_path = artifact_root / payload["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    tool_results_dir = manifest_path.parent / "tool-results"
    for result in manifest["tool_results"]:
        if result["tool_name"] != "skill_view":
            continue
        arguments = (
            tool_results_dir / result["artifact_id"] / "arguments.json"
        ).read_text()
        stat = _tool_stat(stats, _skill_name_from_arguments(arguments))
        stat.call_ids.add(_child_call_id(parent_call_id, result["tool_call_id"]))
        if result["has_error"]:
            stat.failed += 1
        else:
            stat.succeeded += 1


def _tool_stat(
    stats: dict[str, _MutableToolCallStat],
    name: str,
) -> _MutableToolCallStat:
    stat = stats.get(name)
    if stat is None:
        stat = _MutableToolCallStat(name=name, call_ids=set())
        stats[name] = stat
    return stat


def _child_call_id(parent_call_id: str | None, child_call_id: str) -> str:
    if parent_call_id is None:
        return child_call_id
    return f"{parent_call_id}/{child_call_id}"


def _skill_name_from_arguments(arguments: str) -> str:
    payload = json.loads(arguments)
    name = payload["name"]
    if type(name) is not str:
        raise TypeError("skill_view name must be str")
    return name
