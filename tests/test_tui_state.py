import asyncio
import json
from datetime import datetime, timezone
from io import StringIO

import pytest

from agent_core.memory.ideas import Idea
from agent_core.session import (
    EventLog,
    MAIN_THREAD_ID,
    SessionEvent,
    SessionRecorder,
    SessionStore,
)
from aceai.core.events import AgentEventBuilder
from aceai.core.models import AgentStep, ToolExecutionResult
from agent_core.tui import app as tui_app_module
from agent_core.tui.app import AceAITUI
from agent_core.tui.app import STREAM_DELTA_REFRESH_CHARS
from agent_core.tui.app import STREAM_DELTA_REFRESH_SECONDS
from agent_core.tui.demo import static_demo_events
from agent_core.tui.events import TUIEvent
from agent_core.tui.metadata import MetadataSection, _metadata_renderables
from agent_core.tui.session_adapter import tui_event_to_session_event
from agent_core.tui.session_replay import event_log_to_tui_events
from agent_core.tui.setup import (
    SessionListWidget,
    _idea_body_text,
    _session_picker_renderables,
)
from agent_core.tui.state import (
    TUISubagentState,
    TUISubagentToolResult,
    initial_state,
    reduce_events,
    reset_cache_rate,
    select_event,
)
from agent_core.tui.tool_stats import (
    format_skill_call_stats,
    format_tool_call_stats,
    skill_call_stats,
    tool_call_stats,
)
from agent_core.tui.trajectory import TrajectoryScreen, _trajectory_renderables
from agent_core.tui.widgets import (
    CommandInput,
    DetailWidget,
    StreamWidget,
    SubagentStatusWidget,
)
from aceai.llm.models import LLMResponse, LLMToolCall, LLMToolCallDelta, LLMUsage
from rich.console import Console, Group
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.containers import VerticalScroll
from textual.widgets import RichLog, Static


async def _wait_until(pilot, predicate, timeout: float = 0.2) -> None:
    async def wait_for_match() -> None:
        while not predicate():
            await pilot.pause()

    await asyncio.wait_for(wait_for_match(), timeout=timeout)
    assert predicate()


def test_reduce_events_tracks_run_completion() -> None:
    events = static_demo_events()

    state = reduce_events(events)

    assert state.status == "completed"
    assert state.final_answer == (
        "Multi-agent review complete: core behavior, docs, and provider "
        "integration are ready."
    )
    assert state.error is None
    assert state.selected_event_id == events[-1].event_id


def test_reduce_events_tracks_context_compaction_events() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.context_compaction_started(
                reason="threshold",
                compression_count=1,
            )
        ),
        TUIEvent.from_agent_event(
            builder.context_compressed(
                reason="threshold",
                compression_count=1,
                history=[],
            )
        ),
        TUIEvent.from_agent_event(
            builder.context_compaction_failed(
                reason="context_window_retry",
                compression_count=2,
                error="summary request exceeded context window",
            )
        ),
    ]

    state = reduce_events(events)

    assert [event.kind for event in state.events] == [
        "context_compaction_started",
        "context_compressed",
        "context_compaction_failed",
    ]
    assert state.events[0].content == "Compacting context (preflight budget)..."
    assert state.events[1].content == "Context compacted. Compression #1."
    assert state.events[2].content == "summary request exceeded context window"


def test_reduce_events_tracks_step_and_tool_state() -> None:
    state = reduce_events(static_demo_events())

    assert len(state.steps) == 2
    assert [step.status for step in state.steps] == [
        "completed",
        "completed",
    ]
    assert [tool.name for step in state.steps for tool in step.tools] == [
        "delegate_to_subagent",
        "delegate_to_subagent",
        "delegate_to_subagent",
        "run_shell_command",
    ]
    assert [subagent.status for subagent in state.subagents] == [
        "completed",
        "completed",
        "completed",
    ]
    provider_subagent = state.subagents[2]
    assert provider_subagent.task == "Provider integration smoke check"
    assert provider_subagent.error is None


def test_reduce_events_tracks_threaded_delegate_to_subagent_status() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Check whether README names the version",'
            '"instructions":"report evidence","context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-subagent-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"type":"subagent_audit","thread_id":"child-thread-1",'
            '"agent_id":"child-1","run_id":"run-1","status":"completed",'
            '"summary":"README has no version","step_count":1}'
        ),
    )

    state = reduce_events(
        [
            TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
            TUIEvent.from_agent_event(
                builder.tool_completed(tool_call=call, tool_result=result)
            ),
        ]
    )

    assert len(state.subagents) == 1
    subagent = state.subagents[0]
    assert subagent.call_id == "call-subagent-1"
    assert subagent.thread_id == "child-thread-1"
    assert subagent.task == "Check whether README names the version"
    assert subagent.instructions == "report evidence"
    assert subagent.context_brief == "repo"
    assert subagent.allowed_tools == []
    assert subagent.status == "completed"
    assert subagent.agent_id == "child-1"
    assert subagent.run_id == "run-1"
    assert subagent.summary == "README has no version"
    assert subagent.final_answer == ""
    assert subagent.important_evidence == []
    assert subagent.tool_results == []
    assert subagent.step_count == 1
    assert subagent.output == result.output


def test_reduce_events_tracks_spawn_subagent_thread_after_job_created() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="spawn_subagent",
        arguments=(
            '{"task":"Inspect in background","instructions":"report evidence",'
            '"context_brief":"repo","toolset":"dev_read_only","allowed_tools":[]}'
        ),
        call_id="call-spawn-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"job_id":"job-1","thread_id":"child-thread-1",'
            '"agent_id":"child-1","run_id":"run-1","status":"running",'
            '"task":"Inspect in background"}'
        ),
    )

    state = reduce_events(
        [
            TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
            TUIEvent.from_agent_event(
                builder.tool_completed(tool_call=call, tool_result=result)
            ),
        ]
    )

    assert len(state.subagents) == 1
    subagent = state.subagents[0]
    assert subagent.call_id == "call-spawn-1"
    assert subagent.thread_id == "child-thread-1"
    assert subagent.task == "Inspect in background"
    assert subagent.instructions == "report evidence"
    assert subagent.context_brief == "repo"
    assert subagent.allowed_tools == []
    assert subagent.status == "running"
    assert subagent.agent_id == "child-1"
    assert subagent.run_id == "run-1"


def test_reduce_events_updates_spawn_subagent_status_after_collect_results() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    spawn_events: list[TUIEvent] = []
    for index in range(2):
        spawn_call = LLMToolCall(
            name="spawn_subagent",
            arguments=(
                f'{{"task":"Inspect {index + 1}",'
                '"instructions":"report evidence",'
                '"context_brief":"repo","toolset":"dev_read_only",'
                '"allowed_tools":[]}'
            ),
            call_id=f"call-spawn-{index + 1}",
        )
        spawn_result = ToolExecutionResult(
            call=spawn_call,
            output=(
                f'{{"job_id":"job-{index + 1}",'
                f'"thread_id":"child-thread-{index + 1}",'
                f'"agent_id":"child-{index + 1}",'
                f'"run_id":"run-{index + 1}",'
                '"status":"running",'
                f'"task":"Inspect {index + 1}"}}'
            ),
        )
        spawn_events.append(
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=spawn_call,
                    tool_result=spawn_result,
                )
            )
        )
    collect_call = LLMToolCall(
        name="collect_subagent_results",
        arguments='{"job_ids":["job-1","job-2"]}',
        call_id="call-collect-1",
    )
    collect_result = ToolExecutionResult(
        call=collect_call,
        output=(
            '{"jobs":['
            '{"job_id":"job-1","thread_id":"child-thread-1",'
            '"agent_id":"child-1","run_id":"run-1","status":"completed",'
            '"task":"Inspect 1","summary":"done 1",'
            '"final_answer":"answer 1","error":"","step_count":1,'
            '"tool_result_count":3},'
            '{"job_id":"job-2","thread_id":"child-thread-2",'
            '"agent_id":"child-2","run_id":"run-2","status":"completed",'
            '"task":"Inspect 2","summary":"done 2",'
            '"final_answer":"answer 2","error":"","step_count":1,'
            '"tool_result_count":4}'
            ']}'
        ),
    )

    state = reduce_events(
        [
            *spawn_events,
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=collect_call,
                    tool_result=collect_result,
                )
            ),
        ]
    )

    assert [subagent.status for subagent in state.subagents] == [
        "completed",
        "completed",
    ]
    assert [subagent.summary for subagent in state.subagents] == [
        "done 1",
        "done 2",
    ]
    assert [subagent.tool_result_count for subagent in state.subagents] == [3, 4]


def test_reduce_events_excludes_delegate_to_subagent_without_thread_id() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments='{"task":"Inspect","instructions":"","context_brief":"","allowed_tools":[]}',
        call_id="call-subagent-1",
    )
    result = ToolExecutionResult(
        call=call,
        output="subagent failed",
        error="child failed",
    )

    state = reduce_events(
        [
            TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
            TUIEvent.from_agent_event(
                builder.tool_failed(
                    tool_call=call,
                    tool_result=result,
                    error="child failed",
                )
            ),
        ]
    )

    assert state.subagents == []


def test_reduce_events_does_not_track_regular_tool_as_subagent() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"README.md"}',
        call_id="call-tool-1",
    )

    state = reduce_events(
        [TUIEvent.from_agent_event(builder.tool_started(tool_call=call))]
    )

    assert state.subagents == []


def test_reduce_events_assigns_global_step_numbers() -> None:
    events = [
        TUIEvent.from_agent_event(
            AgentEventBuilder(step_index=0, step_id="run-1-step-1").llm_started()
        ),
        TUIEvent.from_agent_event(
            AgentEventBuilder(step_index=0, step_id="run-2-step-1").llm_started()
        ),
    ]

    state = reduce_events(events)

    assert [step.step_index for step in state.steps] == [0, 1]


def test_reduce_events_does_not_add_non_step_events_to_timeline() -> None:
    state = reduce_events(
        [
            TUIEvent.user_message("What changed?"),
            TUIEvent.session_notice("Sessions"),
        ]
    )

    assert state.steps == []
    assert state.events[0].kind == "user_message"
    assert state.events[1].kind == "session_notice"


def test_idea_picker_collapses_unselected_body_to_two_lines() -> None:
    idea = _idea(
        "title\nfirst line is visible\nsecond line is visible\nthird line is hidden"
    )

    collapsed = _idea_body_text(idea, expanded=False)
    expanded = _idea_body_text(idea, expanded=True)

    assert collapsed.plain.count("\n") == 1
    assert "first line is visible" in collapsed.plain
    assert "... more" in collapsed.plain
    assert "third line is hidden" not in collapsed.plain
    assert "third line is hidden" in expanded.plain


def test_idea_picker_short_body_keeps_two_line_height() -> None:
    collapsed = _idea_body_text(_idea("title"), expanded=False)
    title_only = _idea_body_text(_idea("title"), expanded=True)
    one_line = _idea_body_text(_idea("title\none line"), expanded=True)

    assert collapsed.plain == " \n "
    assert title_only.plain == " \n "
    assert one_line.plain == "one line\n "


def test_reduce_events_does_not_add_restored_transcript_to_timeline() -> None:
    events = event_log_to_tui_events(
        EventLog(
            [
                SessionEvent(kind="user_message", payload={"content": "question"}),
                SessionEvent(kind="assistant_message", payload={"content": "answer"}),
                SessionEvent(
                    kind="tool_result",
                    payload={
                        "content": "",
                        "tool_name": "read_text_file",
                        "tool_call_id": "call-1",
                        "tool_arguments": '{"path":"README.md"}',
                        "output": "contents",
                        "truncated_output": "contents",
                        "status": "completed",
                    },
                ),
            ]
        )
    )

    state = reduce_events(events)

    assert state.steps == []
    assert state.status == "idle"


def test_reduce_events_coalesces_consecutive_assistant_deltas() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="hello ")),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="world")),
    ]

    state = reduce_events(events)

    assert len(state.events) == 1
    assert state.events[0].kind == "assistant_delta"
    assert state.events[0].content == "hello world"
    assert len(state.steps[0].events) == 1
    assert state.steps[0].events[0].content == "hello world"


def test_initial_state_is_idle() -> None:
    state = initial_state()

    assert state.status == "idle"
    assert state.steps == []
    assert state.events == []


def test_select_event_updates_selected_event_id() -> None:
    events = static_demo_events()
    state = reduce_events(events)
    tool_event = _first_event(events, "tool_completed")

    selected = select_event(state, tool_event.event_id)

    assert selected.selected_event_id == tool_event.event_id


def test_select_event_rejects_unknown_event_id() -> None:
    state = reduce_events(static_demo_events())

    with pytest.raises(ValueError, match="selected event does not exist"):
        select_event(state, "missing-event")


def test_reduce_events_tracks_usage_totals() -> None:
    first = AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="first",
                usage=LLMUsage(
                    input_tokens=100,
                    cached_input_tokens=40,
                    output_tokens=20,
                    total_tokens=120,
                ),
            )
        )
    )
    second = AgentEventBuilder(step_index=1, step_id="step-2").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="second",
                usage=LLMUsage(input_tokens=150, output_tokens=30, total_tokens=180),
            )
        )
    )

    state = reduce_events(
        [TUIEvent.from_agent_event(first), TUIEvent.from_agent_event(second)]
    )

    assert state.usage.current_context_tokens == 150
    assert state.usage.session_input_tokens == 250
    assert state.usage.session_cached_input_tokens == 40
    assert state.usage.session_output_tokens == 50
    assert state.usage.session_total_tokens == 300


def test_tool_call_stats_include_parent_and_child_tool_results(tmp_path) -> None:
    parent_call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"ls"}',
        call_id="call-parent",
    )
    delegate_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments='{"task":"inspect"}',
        call_id="call-delegate",
    )
    events = [
        TUIEvent(
            kind="tool_started",
            step_index=0,
            step_id="step-parent",
            title="tool run_shell_command",
            tool_name="run_shell_command",
            tool_call_id=parent_call.call_id,
            tool_call=parent_call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_failed",
            step_index=0,
            step_id="step-parent",
            title="tool run_shell_command failed",
            content="exit 1",
            tool_name="run_shell_command",
            tool_call_id=parent_call.call_id,
            tool_call=parent_call,
            tool_result=ToolExecutionResult(
                call=parent_call,
                output="exit 1",
                error="exit 1",
            ),
            error="exit 1",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=1,
            step_id="step-delegate",
            title="tool delegate_to_subagent completed",
            content=json.dumps(
                {
                    "agent_id": "child-1",
                    "run_id": "child-run-1",
                    "status": "completed",
                    "final_answer": "done",
                    "summary": "done",
                    "important_evidence": [],
                    "tool_results": [
                        {
                            "tool_name": "read_text_file",
                            "call_id": "call-child-ok",
                            "arguments": '{"path":"README.md"}',
                            "output": "readme",
                            "error": None,
                        },
                        {
                            "tool_name": "search_text",
                            "call_id": "call-child-failed",
                            "arguments": '{"query":"missing"}',
                            "output": "failed",
                            "error": "failed",
                        },
                    ],
                    "step_count": 2,
                }
            ),
            tool_name="delegate_to_subagent",
            tool_call_id=delegate_call.call_id,
            tool_call=delegate_call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=2,
            step_id="step-delegate-2",
            title="tool delegate_to_subagent completed",
            content=json.dumps(
                {
                    "agent_id": "child-2",
                    "run_id": "child-run-2",
                    "status": "completed",
                    "final_answer": "done",
                    "summary": "done",
                    "important_evidence": [],
                    "tool_results": [
                        {
                            "tool_name": "read_text_file",
                            "call_id": "call-child-ok",
                            "arguments": '{"path":"CHANGELOG.md"}',
                            "output": "changelog",
                            "error": None,
                        }
                    ],
                    "step_count": 2,
                }
            ),
            tool_name="delegate_to_subagent",
            tool_call_id="call-delegate-2",
            tool_call=LLMToolCall(
                name="delegate_to_subagent",
                arguments='{"task":"inspect again"}',
                call_id="call-delegate-2",
            ),
            raw_event=None,
        ),
    ]

    lines = format_tool_call_stats(tool_call_stats(events, artifact_root=tmp_path))

    assert "delegate_to_subagent: calls 2  ok 2  failed 0" in lines
    assert "run_shell_command: calls 1  ok 0  failed 1" in lines
    assert "read_text_file: calls 2  ok 2  failed 0" in lines
    assert "search_text: calls 1  ok 0  failed 1" in lines


def test_tool_call_stats_read_archived_subagent_manifest(tmp_path) -> None:
    manifest_path = "session-1/artifacts/parent-run-1/child-1/manifest.json"
    manifest_file = tmp_path / manifest_path
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text(
        json.dumps(
            {
                "type": "subagent_artifact_manifest",
                "tool_results": [
                    {
                        "tool_name": "read_text_file",
                        "tool_call_id": "call-child-ok",
                        "has_error": False,
                    },
                    {
                        "tool_name": "read_text_file",
                        "tool_call_id": "call-child-failed",
                        "has_error": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    delegate_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments='{"task":"inspect"}',
        call_id="call-delegate",
    )
    events = [
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-delegate",
            title="tool delegate_to_subagent completed",
            content=json.dumps(
                {
                    "type": "subagent_audit",
                    "agent_id": "child-1",
                    "run_id": "child-run-1",
                    "status": "completed",
                    "manifest_path": manifest_path,
                }
            ),
            tool_name="delegate_to_subagent",
            tool_call_id=delegate_call.call_id,
            tool_call=delegate_call,
            raw_event=None,
        )
    ]

    lines = format_tool_call_stats(tool_call_stats(events, artifact_root=tmp_path))

    assert "delegate_to_subagent: calls 1  ok 1  failed 0" in lines
    assert "read_text_file: calls 2  ok 1  failed 1" in lines


def test_skill_call_stats_include_parent_and_child_skill_view_calls(tmp_path) -> None:
    parent_call = LLMToolCall(
        name="skill_view",
        arguments='{"name":"release"}',
        call_id="call-parent-skill",
    )
    delegate_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments='{"task":"inspect"}',
        call_id="call-delegate",
    )
    events = [
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-parent",
            title="tool skill_view completed",
            content="release skill",
            tool_name="skill_view",
            tool_call_id=parent_call.call_id,
            tool_call=parent_call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=1,
            step_id="step-delegate",
            title="tool delegate_to_subagent completed",
            content=json.dumps(
                {
                    "agent_id": "child-1",
                    "run_id": "child-run-1",
                    "status": "completed",
                    "final_answer": "done",
                    "summary": "done",
                    "important_evidence": [],
                    "tool_results": [
                        {
                            "tool_name": "skill_view",
                            "call_id": "call-child-skill-ok",
                            "arguments": '{"name":"release","file_path":"references/api.md"}',
                            "output": "reference",
                            "error": None,
                        },
                        {
                            "tool_name": "skill_view",
                            "call_id": "call-child-skill-failed",
                            "arguments": '{"name":"debug"}',
                            "output": "failed",
                            "error": "failed",
                        },
                    ],
                    "step_count": 2,
                }
            ),
            tool_name="delegate_to_subagent",
            tool_call_id=delegate_call.call_id,
            tool_call=delegate_call,
            raw_event=None,
        ),
    ]

    lines = format_skill_call_stats(skill_call_stats(events, artifact_root=tmp_path))

    assert "debug: calls 1  ok 0  failed 1" in lines
    assert "release: calls 2  ok 2  failed 0" in lines


def test_skill_call_stats_read_archived_subagent_manifest(tmp_path) -> None:
    manifest_path = "session-1/artifacts/parent-run-1/child-1/manifest.json"
    manifest_file = tmp_path / manifest_path
    tool_artifact_dir = manifest_file.parent / "tool-results" / "artifact-skill"
    tool_artifact_dir.mkdir(parents=True)
    (tool_artifact_dir / "arguments.json").write_text(
        '{"name":"release"}',
        encoding="utf-8",
    )
    manifest_file.write_text(
        json.dumps(
            {
                "type": "subagent_artifact_manifest",
                "tool_results": [
                    {
                        "artifact_id": "artifact-skill",
                        "tool_name": "skill_view",
                        "tool_call_id": "call-child-skill",
                        "has_error": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    delegate_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments='{"task":"inspect"}',
        call_id="call-delegate",
    )
    events = [
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-delegate",
            title="tool delegate_to_subagent completed",
            content=json.dumps(
                {
                    "type": "subagent_audit",
                    "agent_id": "child-1",
                    "run_id": "child-run-1",
                    "status": "completed",
                    "manifest_path": manifest_path,
                }
            ),
            tool_name="delegate_to_subagent",
            tool_call_id=delegate_call.call_id,
            tool_call=delegate_call,
            raw_event=None,
        )
    ]

    lines = format_skill_call_stats(skill_call_stats(events, artifact_root=tmp_path))

    assert "release: calls 1  ok 1  failed 0" in lines


def test_metadata_sections_include_tool_call_stats(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tui_app_module,
        "SessionStore",
        lambda *, project: SessionStore(tmp_path, project=project),
    )
    call = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"README.md"}',
        call_id="call-read",
    )
    event = TUIEvent(
        kind="tool_completed",
        step_index=0,
        step_id="step-read",
        title="tool read_text_file completed",
        content="README",
        tool_name="read_text_file",
        tool_call_id=call.call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(call=call, output="README"),
        raw_event=None,
    )
    app = AceAITUI([event])
    app._state = reduce_events([event])

    sections = app._metadata_sections()

    section_lines = {section.title: "\n".join(section.lines) for section in sections}
    assert (
        "read_text_file: calls 1  ok 1  failed 0"
        in section_lines["Session Tool Calls"]
    )
    assert "Global Tool Calls" not in section_lines


def test_metadata_sections_include_skill_call_stats(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tui_app_module,
        "SessionStore",
        lambda *, project: SessionStore(tmp_path, project=project),
    )
    call = LLMToolCall(
        name="skill_view",
        arguments='{"name":"release"}',
        call_id="call-skill",
    )
    event = TUIEvent(
        kind="tool_completed",
        step_index=0,
        step_id="step-skill",
        title="tool skill_view completed",
        content="Release",
        tool_name="skill_view",
        tool_call_id=call.call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(call=call, output="Release"),
        raw_event=None,
    )
    app = AceAITUI([event])
    app._state = reduce_events([event])

    sections = app._metadata_sections()

    section_lines = {section.title: "\n".join(section.lines) for section in sections}
    assert "release: calls 1  ok 1  failed 0" in section_lines["Session Skill Calls"]
    assert "Global Skill Calls" not in section_lines


def test_metadata_sections_omit_empty_tool_call_stats(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        tui_app_module,
        "SessionStore",
        lambda *, project: SessionStore(tmp_path, project=project),
    )
    app = AceAITUI([])

    sections = app._metadata_sections()

    titles = [section.title for section in sections]
    assert "Session Tool Calls" not in titles
    assert "Global Tool Calls" not in titles
    assert "Session Skill Calls" not in titles
    assert "Global Skill Calls" not in titles


def test_metadata_sections_include_global_tool_stats_without_active_session(
    monkeypatch,
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    session = store.create_session()
    event = _tool_completed_event("read_text_file", "call-global")
    SessionRecorder(store, session.session_id).record(tui_event_to_session_event(event))
    monkeypatch.setattr(
        tui_app_module,
        "SessionStore",
        lambda *, project: store,
    )
    app = AceAITUI([])

    sections = app._metadata_sections()

    section_lines = {section.title: "\n".join(section.lines) for section in sections}
    assert "Session Tool Calls" not in section_lines
    assert (
        "read_text_file: calls 1  ok 1  failed 0"
        in section_lines["Global Tool Calls"]
    )


def test_metadata_sections_include_global_tool_call_stats(tmp_path) -> None:
    store = SessionStore(tmp_path)
    current = store.create_session()
    other = store.create_session()
    current_event = _tool_completed_event("read_text_file", "call-current")
    other_event = _tool_failed_event("run_shell_command", "call-other")
    SessionRecorder(store, current.session_id).record(
        tui_event_to_session_event(current_event)
    )
    SessionRecorder(store, other.session_id).record(tui_event_to_session_event(other_event))
    app = AceAITUI(
        [current_event],
        session_recorder=SessionRecorder(store, current.session_id),
        session_id=current.session_id,
    )
    app._state = reduce_events([current_event])

    sections = app._metadata_sections()

    section_lines = {section.title: "\n".join(section.lines) for section in sections}
    assert (
        "read_text_file: calls 1  ok 1  failed 0"
        in section_lines["Session Tool Calls"]
    )
    assert (
        "read_text_file: calls 1  ok 1  failed 0"
        in section_lines["Global Tool Calls"]
    )
    assert (
        "run_shell_command: calls 1  ok 0  failed 1"
        in section_lines["Global Tool Calls"]
    )
    assert "run_shell_command" not in section_lines["Session Tool Calls"]


def test_metadata_sections_include_global_skill_call_stats(tmp_path) -> None:
    store = SessionStore(tmp_path)
    current = store.create_session()
    other = store.create_session()
    current_event = _skill_completed_event("release", "call-current")
    other_event = _skill_failed_event("debug", "call-other")
    SessionRecorder(store, current.session_id).record(
        tui_event_to_session_event(current_event)
    )
    SessionRecorder(store, other.session_id).record(tui_event_to_session_event(other_event))
    app = AceAITUI(
        [current_event],
        session_recorder=SessionRecorder(store, current.session_id),
        session_id=current.session_id,
    )
    app._state = reduce_events([current_event])

    sections = app._metadata_sections()

    section_lines = {section.title: "\n".join(section.lines) for section in sections}
    assert "release: calls 1  ok 1  failed 0" in section_lines["Session Skill Calls"]
    assert "release: calls 1  ok 1  failed 0" in section_lines["Global Skill Calls"]
    assert "debug: calls 1  ok 0  failed 1" in section_lines["Global Skill Calls"]
    assert "debug" not in section_lines["Session Skill Calls"]


def test_tool_call_metadata_renders_as_stats_table() -> None:
    rendered = _render_to_text(
        Group(
            *_metadata_renderables(
                [
                    MetadataSection(
                        title="Global Tool Calls",
                        lines=[
                            "delegate_to_subagent: calls 16  ok 8  failed 8",
                            "read_text_file: calls 5  ok 5  failed 0",
                        ],
                    )
                ]
            )
        )
    )

    assert "tool" in rendered
    assert "calls" in rendered
    assert "ok" in rendered
    assert "failed" in rendered
    assert "delegate_to_subagent" in rendered
    assert "read_text_file" in rendered


def test_skill_call_metadata_renders_as_stats_table() -> None:
    rendered = _render_to_text(
        Group(
            *_metadata_renderables(
                [
                    MetadataSection(
                        title="Global Skill Calls",
                        lines=[
                            "release: calls 2  ok 2  failed 0",
                            "debug: calls 1  ok 0  failed 1",
                        ],
                    )
                ]
            )
        )
    )

    assert "skill" in rendered
    assert "calls" in rendered
    assert "ok" in rendered
    assert "failed" in rendered
    assert "release" in rendered
    assert "debug" in rendered


def test_reset_cache_rate_clears_current_cache_only() -> None:
    event = AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="first",
                usage=LLMUsage(
                    input_tokens=100,
                    cached_input_tokens=40,
                    cache_miss_input_tokens=60,
                    input_cache_hit_rate=0.4,
                    output_tokens=20,
                    total_tokens=120,
                ),
            )
        )
    )
    state = reduce_events([TUIEvent.from_agent_event(event)])

    next_state = reset_cache_rate(state)

    assert next_state.usage.current_context_tokens == 100
    assert next_state.usage.current_cached_input_tokens == 0
    assert next_state.usage.current_input_cache_hit_rate == 0.0
    assert next_state.usage.session_cached_input_tokens == 40
    assert next_state.usage.session_total_tokens == 120


def test_reduce_events_tracks_cost_totals() -> None:
    first = AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="first",
                model="gpt-5.5",
                usage=LLMUsage(
                    input_tokens=1_000,
                    cached_input_tokens=600,
                    output_tokens=100,
                    total_tokens=1_100,
                ),
            )
        )
    )
    second = AgentEventBuilder(step_index=1, step_id="step-2").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="second",
                model="gpt-5.4-mini",
                usage=LLMUsage(
                    input_tokens=1_000, output_tokens=100, total_tokens=1_100
                ),
            )
        )
    )

    state = reduce_events(
        [TUIEvent.from_agent_event(first), TUIEvent.from_agent_event(second)]
    )

    assert state.usage.current_cost_usd is not None
    assert state.usage.session_cost_usd is not None
    assert round(state.usage.current_cost_usd, 6) == 0.0012
    assert round(state.usage.session_cost_usd, 6) == 0.0065


def test_reduce_events_keeps_missing_usage_unknown() -> None:
    state = reduce_events(
        [
            TUIEvent.from_agent_event(
                AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
                    step=AgentStep(llm_response=LLMResponse(text="answer"))
                )
            )
        ]
    )

    assert state.usage.current_context_tokens is None
    assert state.usage.session_input_tokens is None
    assert state.usage.session_output_tokens is None
    assert state.usage.session_total_tokens is None


def test_session_notification_uses_native_toast() -> None:
    app = AceAITUI([])
    calls: list[tuple[str, dict[str, object]]] = []
    app.notify = lambda message, **kwargs: calls.append((message, kwargs))

    app.notify_session("Resumed session abc")

    assert calls == [
        (
            "Resumed session abc",
            {
                "title": "AceAI",
                "severity": "information",
                "timeout": 3.0,
            },
        )
    ]


def test_llm_retrying_uses_native_notification_and_stream_event() -> None:
    builder = AgentEventBuilder(
        step_index=0,
        step_id="step-1",
        run_id="12345678-1234-1234-1234-123456789abc",
    )
    app = AceAITUI([])
    calls: list[tuple[str, dict[str, object]]] = []
    app.notify = lambda message, **kwargs: calls.append((message, kwargs))
    app._refresh_widgets = lambda: None

    app.append_agent_event(
        builder.llm_retrying(
            retry_count=1,
            retry_max=2,
            retry_delay_seconds=0.5,
            error="RemoteProtocolError: peer closed",
        )
    )

    assert [event.kind for event in app._state.events] == ["llm_retrying"]
    assert app._state.events[0].run_id == "12345678-1234-1234-1234-123456789abc"
    assert app._state.events[0].retry_count == 1
    assert calls == [
        (
            "Retrying message 1/2 in 0.5s after RemoteProtocolError: peer closed",
            {
                "title": "Retrying message 1/2 · 12345678",
                "severity": "warning",
                "timeout": 3.0,
            },
        )
    ]


@pytest.mark.anyio
async def test_static_tui_loads_fixture_events() -> None:
    events = static_demo_events()
    app = AceAITUI(events)

    async with app.run_test():
        assert app._state.status == "completed"
        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)
        subagents = app.query_one(SubagentStatusWidget)
        assert stream.can_focus
        assert detail.can_focus
        assert subagents.has_class("hidden")
        app.action_show_subagents()
        assert not subagents.has_class("hidden")
        assert "subagents  3 total | 0 running | 3 done | 0 failed" in subagents.renderable
        assert not stream.debug_mode
        assert detail.has_class("collapsed")


@pytest.mark.anyio
async def test_static_tui_demo_can_start_with_subagent_panel_visible() -> None:
    from agent_core.tui.demo import static_demo_events

    class StaticDemoTUI(AceAITUI):
        def on_mount(self) -> None:
            super().on_mount()
            self.call_after_refresh(self.action_show_subagents)

    app = StaticDemoTUI(static_demo_events())

    async with app.run_test() as pilot:
        await pilot.pause()
        subagents = app.query_one(SubagentStatusWidget)
        assert not subagents.has_class("hidden")
        assert "subagents  3 total | 0 running | 3 done | 0 failed" in subagents.renderable


@pytest.mark.anyio
async def test_tui_shows_subagent_status_widget_for_delegate_tool() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect version metadata",'
            '"instructions":"report evidence","context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-subagent-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"type":"subagent_audit","thread_id":"child-thread-1",'
            '"agent_id":"child-1","run_id":"run-1","status":"running",'
            '"summary":"","step_count":0}'
        ),
    )
    app = AceAITUI(
        [
            TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
            TUIEvent.from_agent_event(
                builder.tool_completed(tool_call=call, tool_result=result)
            ),
        ],
    )

    async with app.run_test():
        subagents = app.query_one(SubagentStatusWidget)
        detail = app.query_one(DetailWidget)

        app.action_show_subagents()

        assert not subagents.has_class("hidden")
        assert detail.has_class("collapsed")
        assert "subagents  1 total | 0 running | 1 done | 0 failed\n" in subagents.renderable
        assert "                 < [1] >" in subagents.renderable
        assert "#1 [completed] Inspect version metadata" in subagents.renderable
        assert "run" in subagents.renderable
        assert "status: completed" in subagents.renderable
        assert "steps: 0" in subagents.renderable
        assert "results: 0" in subagents.renderable
        assert "tools\n     - none" in subagents.renderable
        assert "task / context\n     repo" in subagents.renderable
        assert "task / ask\n     report evidence" in subagents.renderable


@pytest.mark.anyio
async def test_tui_auto_opens_after_spawn_subagent_thread_is_created_and_closes_after_collect(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    store.create_thread(
        session_id=metadata.session_id,
        thread_id="child-thread-1",
        agent_id="child-1",
        role="subagent",
        title="Inspect in background",
        status="running",
        parent_thread_id=MAIN_THREAD_ID,
        metadata={
            "instructions": "report evidence",
            "context_brief": "repo",
            "allowed_tools": [],
        },
    )
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="spawn_subagent",
        arguments=(
            '{"task":"Inspect in background","instructions":"report evidence",'
            '"context_brief":"repo","toolset":"dev_read_only","allowed_tools":[]}'
        ),
        call_id="call-spawn-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"job_id":"job-1","thread_id":"child-thread-1",'
            '"agent_id":"child-1","run_id":"run-1","status":"running",'
            '"task":"Inspect in background"}'
        ),
    )
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        subagents = app.query_one(SubagentStatusWidget)

        app.append_event(TUIEvent.from_agent_event(builder.tool_started(tool_call=call)))

        assert subagents.has_class("hidden")

        app.append_event(
            TUIEvent.from_agent_event(
                builder.tool_completed(tool_call=call, tool_result=result)
            )
        )

        assert not subagents.has_class("hidden")
        assert "subagents  1 total | 1 running | 0 done | 0 failed" in subagents.renderable
        assert "#1 [running] Inspect in background" in subagents.renderable

        app.append_persisted_event(
            TUIEvent.from_session_event(
                SessionEvent(
                    event_id="inbox-1",
                    thread_id=MAIN_THREAD_ID,
                    agent_id="child-1",
                    kind="agent_inbox_item",
                    payload={
                        "source_thread_id": "child-thread-1",
                        "source_agent_id": "child-1",
                        "severity": "info",
                        "message": "Background subagent job job-1 completed.",
                        "status": "pending",
                    },
                )
            )
        )

        assert not subagents.has_class("hidden")

        collect_call = LLMToolCall(
            name="collect_subagent_results",
            arguments='{"job_ids":["job-1"]}',
            call_id="call-collect-1",
        )
        collect_result = ToolExecutionResult(
            call=collect_call,
            output=(
                '{"jobs":[{"job_id":"job-1","thread_id":"child-thread-1",'
                '"agent_id":"child-1","run_id":"run-1","status":"completed",'
                '"task":"Inspect in background","summary":"done",'
                '"final_answer":"done","error":"","step_count":1}]}'
            ),
        )
        app.append_event(
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=collect_call,
                    tool_result=collect_result,
                )
            )
        )

        assert subagents.has_class("hidden")


@pytest.mark.anyio
async def test_tui_closes_subagent_panel_after_collecting_multiple_background_jobs(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )
    builder = AgentEventBuilder(step_index=0, step_id="step-1")

    async with app.run_test():
        subagents = app.query_one(SubagentStatusWidget)
        for index in range(2):
            job_id = f"job-{index + 1}"
            thread_id = f"child-thread-{index + 1}"
            store.create_thread(
                session_id=metadata.session_id,
                thread_id=thread_id,
                agent_id=f"child-{index + 1}",
                role="subagent",
                title=f"Inspect {index + 1}",
                status="running",
                parent_thread_id=MAIN_THREAD_ID,
                metadata={
                    "instructions": "report evidence",
                    "context_brief": "repo",
                    "allowed_tools": [],
                },
            )
            spawn_call = LLMToolCall(
                name="spawn_subagent",
                arguments=(
                    f'{{"task":"Inspect {index + 1}",'
                    '"instructions":"report evidence",'
                    '"context_brief":"repo","toolset":"dev_read_only",'
                    '"allowed_tools":[]}'
                ),
                call_id=f"call-spawn-{index + 1}",
            )
            spawn_result = ToolExecutionResult(
                call=spawn_call,
                output=(
                    f'{{"job_id":"{job_id}","thread_id":"{thread_id}",'
                    f'"agent_id":"child-{index + 1}","run_id":"run-{index + 1}",'
                    '"status":"running",'
                    f'"task":"Inspect {index + 1}"}}'
                ),
            )
            app.append_event(
                TUIEvent.from_agent_event(
                    builder.tool_completed(
                        tool_call=spawn_call,
                        tool_result=spawn_result,
                    )
                )
            )

        assert not subagents.has_class("hidden")
        assert "subagents  2 total | 2 running | 0 done | 0 failed" in subagents.renderable

        for index in range(2):
            app.append_persisted_event(
                TUIEvent.from_session_event(
                    SessionEvent(
                        event_id=f"inbox-{index + 1}",
                        thread_id=MAIN_THREAD_ID,
                        agent_id=f"child-{index + 1}",
                        kind="agent_inbox_item",
                        payload={
                            "source_thread_id": f"child-thread-{index + 1}",
                            "source_agent_id": f"child-{index + 1}",
                            "severity": "info",
                            "message": f"Background subagent job job-{index + 1} completed.",
                            "status": "pending",
                        },
                    )
                )
            )

        assert not subagents.has_class("hidden")

        collect_call = LLMToolCall(
            name="collect_subagent_results",
            arguments='{"job_ids":["job-1","job-2"]}',
            call_id="call-collect-1",
        )
        collect_result = ToolExecutionResult(
            call=collect_call,
            output=(
                '{"jobs":['
                '{"job_id":"job-1","thread_id":"child-thread-1",'
                '"agent_id":"child-1","run_id":"run-1","status":"completed",'
                '"task":"Inspect 1","summary":"done",'
                '"final_answer":"done","error":"","step_count":1},'
                '{"job_id":"job-2","thread_id":"child-thread-2",'
                '"agent_id":"child-2","run_id":"run-2","status":"completed",'
                '"task":"Inspect 2","summary":"done",'
                '"final_answer":"done","error":"","step_count":1}'
                ']}'
            ),
        )
        app.append_event(
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=collect_call,
                    tool_result=collect_result,
                )
            )
        )

        assert subagents.has_class("hidden")


@pytest.mark.anyio
async def test_tui_excludes_completed_delegate_without_thread_id() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect version metadata",'
            '"instructions":"report evidence","context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-subagent-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"agent_id":"child-1","run_id":"run-1","status":"completed",'
            '"final_answer":"version found","summary":"version found",'
            '"important_evidence":[],"tool_results":[],"step_count":1}'
        ),
    )
    app = AceAITUI(
        [TUIEvent.from_agent_event(builder.tool_started(tool_call=call))],
    )

    async with app.run_test():
        subagents = app.query_one(SubagentStatusWidget)

        app.append_event(
            TUIEvent.from_agent_event(
                builder.tool_completed(tool_call=call, tool_result=result)
            )
        )

        assert subagents.has_class("hidden")

        app.action_show_subagents()

        assert subagents.has_class("hidden")


@pytest.mark.anyio
async def test_tui_excludes_failed_delegate_without_thread_id() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect version metadata",'
            '"instructions":"report evidence","context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-subagent-1",
    )
    result = ToolExecutionResult(
        call=call,
        output="subagent failed",
        error="child failed",
    )
    app = AceAITUI(
        [TUIEvent.from_agent_event(builder.tool_started(tool_call=call))],
    )

    async with app.run_test():
        subagents = app.query_one(SubagentStatusWidget)

        app.append_event(
            TUIEvent.from_agent_event(
                builder.tool_failed(
                    tool_call=call,
                    tool_result=result,
                    error="child failed",
                )
            )
        )

        assert subagents.has_class("hidden")


def test_subagent_status_widget_paginates_full_agent_details() -> None:
    widget = SubagentStatusWidget()
    widget.clear = lambda: None
    widget.write = lambda content, **kwargs: None
    widget.call_after_refresh = lambda callback, **kwargs: None
    widget.scroll_home = lambda **kwargs: None
    first_summary = "first summary " + ("x" * 80)
    second_brief = "second brief " + ("y" * 80)

    widget.set_subagents(
        [
            TUISubagentState(
                call_id="call-1",
                task="First delegated investigation with a long title that should not be shortened",
                instructions="Use the repo evidence and report exact files.",
                context_brief="first context",
                allowed_tools=["read_text_file"],
                status="completed",
                agent_id="child-first-full-id",
                run_id="run-first-full-id",
                summary=first_summary,
                important_evidence=["first evidence " + ("z" * 80)],
                tool_results=[
                    TUISubagentToolResult(
                        tool_name="read_text_file",
                        call_id="call-read",
                        output="tool output " + ("o" * 80),
                    )
                ],
                step_count=2,
            ),
            TUISubagentState(
                call_id="call-2",
                task="Second delegated investigation",
                instructions="Use hosted web search.",
                context_brief=second_brief,
                allowed_tools=["openai:web_search"],
                status="running",
                agent_id="child-second-full-id",
                run_id="run-second-full-id",
                step_count=1,
            ),
        ]
    )

    assert "< [1] 2 >" in widget.renderable
    assert "First delegated investigation with a long title that should not be shortened" in widget.renderable
    assert first_summary in widget.renderable
    assert "run" in widget.renderable
    assert "steps: 2" in widget.renderable
    assert "tools\n     - read_text_file" in widget.renderable
    assert "results: 1" in widget.renderable
    assert "task / context\n     first context" in widget.renderable
    assert "task / ask\n     Use the repo evidence and report exact files." in widget.renderable
    assert "task / summary\n     " + first_summary in widget.renderable
    assert "child-first-full-id" in widget.renderable
    assert "tool output " + ("o" * 80) in widget.renderable
    assert "Second delegated investigation" not in widget.renderable

    widget.next_page()

    assert "< 1 [2] >" in widget.renderable
    assert "Second delegated investigation" in widget.renderable
    assert second_brief in widget.renderable
    assert "openai:web_search" in widget.renderable
    assert "First delegated investigation" not in widget.renderable


def test_subagent_panel_states_exclude_active_child_and_style_main_entry() -> None:
    states = tui_app_module._subagent_panel_states(
        thread_options=[
            tui_app_module.SubagentThreadOption(
                thread_id=MAIN_THREAD_ID,
                label="Main",
                status="completed",
                role="main",
            ),
            tui_app_module.SubagentThreadOption(
                thread_id="child-thread-1",
                label="Child 1",
                status="running",
                role="subagent",
            ),
            tui_app_module.SubagentThreadOption(
                thread_id="child-thread-2",
                label="Child 2",
                status="completed",
                role="subagent",
            ),
        ],
        active_thread_id="child-thread-1",
    )

    assert [state.thread_id for state in states] == [
        MAIN_THREAD_ID,
        "child-thread-2",
    ]

    widget = SubagentStatusWidget()
    widget.set_state(
        subagents=states,
        thread_options=[],
        active_thread_id="child-thread-1",
    )

    assert "< [1] 2 >" in widget.renderable
    assert "#1 < main agent" in widget.renderable
    assert "role: parent conversation" in widget.renderable
    assert "action: activate returns to main" in widget.renderable
    assert "Child 1" not in widget.renderable
    assert "1 total | 0 running | 1 done | 0 failed" in widget.renderable


def test_tui_event_replays_agent_inbox_session_events() -> None:
    inbox_event = SessionEvent(
        event_id="inbox-1",
        thread_id=MAIN_THREAD_ID,
        kind="agent_inbox_item",
        payload={
            "source_thread_id": "child-thread-1",
            "source_agent_id": "child-agent-1",
            "severity": "info",
            "message": "child is done",
            "status": "pending",
        },
    )
    delivered_event = SessionEvent(
        event_id="delivered-1",
        thread_id=MAIN_THREAD_ID,
        kind="agent_inbox_delivered",
        payload={"inbox_event_id": "inbox-1"},
    )

    inbox_tui_event = TUIEvent.from_session_event(inbox_event)
    delivered_tui_event = TUIEvent.from_session_event(delivered_event)

    assert inbox_tui_event is not None
    assert inbox_tui_event.kind == "agent_inbox_item"
    assert inbox_tui_event.content == "child is done"
    assert delivered_tui_event is not None
    assert delivered_tui_event.kind == "agent_inbox_delivered"
    assert delivered_tui_event.content == "inbox-1"


def test_subagent_panel_states_include_inbox_counts() -> None:
    states = tui_app_module._subagent_panel_states(
        thread_options=[
            tui_app_module.SubagentThreadOption(
                thread_id=MAIN_THREAD_ID,
                label="Main",
                status="completed",
                role="main",
                inbox_pending_count=2,
                inbox_latest="latest main inbox",
            ),
            tui_app_module.SubagentThreadOption(
                thread_id="child-thread-1",
                label="Child 1",
                status="blocked",
                role="subagent",
                inbox_pending_count=1,
                inbox_latest="needs parent input",
            ),
        ],
        active_thread_id=MAIN_THREAD_ID,
    )

    assert len(states) == 1
    assert states[0].thread_id == "child-thread-1"
    assert states[0].status == "blocked"
    assert states[0].inbox_pending_count == 1
    assert states[0].inbox_latest == "needs parent input"

    widget = SubagentStatusWidget()
    widget.set_state(
        subagents=states,
        thread_options=[],
        active_thread_id=MAIN_THREAD_ID,
    )

    assert "inbox: 1" in widget.renderable
    assert "task / inbox\n     needs parent input" in widget.renderable


def test_subagent_panel_states_include_restored_run_stats() -> None:
    states = tui_app_module._subagent_panel_states(
        thread_options=[
            tui_app_module.SubagentThreadOption(
                thread_id=MAIN_THREAD_ID,
                label="Main",
                status="completed",
                role="main",
            ),
            tui_app_module.SubagentThreadOption(
                thread_id="child-thread-1",
                label="Child 1",
                status="completed",
                role="subagent",
                agent_id="child-agent-1",
                run_id="child-run-1",
                summary="child summary",
                final_answer="child summary",
                step_count=7,
                tool_result_count=3,
            ),
        ],
        active_thread_id=MAIN_THREAD_ID,
    )

    assert len(states) == 1
    assert states[0].run_id == "child-run-1"
    assert states[0].step_count == 7
    assert states[0].tool_result_count == 3

    widget = SubagentStatusWidget()
    widget.set_state(
        subagents=states,
        thread_options=[],
        active_thread_id=MAIN_THREAD_ID,
    )

    assert "steps: 7" in widget.renderable
    assert "results: 3" in widget.renderable


def test_run_summaries_by_thread_counts_restored_child_events() -> None:
    event_log = EventLog(
        [
            SessionEvent(
                thread_id="child-thread-1",
                run_id="child-run-1",
                kind="user_message",
                payload={"content": "inspect"},
            ),
            SessionEvent(
                thread_id="child-thread-1",
                run_id="child-run-1",
                step_id="step-1",
                kind="assistant_message",
                payload={"content": "thinking"},
            ),
            SessionEvent(
                thread_id="child-thread-1",
                run_id="child-run-1",
                step_id="step-1",
                kind="tool_result",
                payload={"tool_call_id": "call-1", "content": "ok"},
            ),
            SessionEvent(
                thread_id="child-thread-1",
                run_id="child-run-1",
                step_id="step-2",
                kind="run_completed",
                payload={"content": "done"},
            ),
        ]
    )

    summaries = tui_app_module._run_summaries_by_thread(event_log)

    summary = summaries["child-thread-1"]
    assert summary.run_id == "child-run-1"
    assert summary.final_answer == "done"
    assert summary.step_count == 2
    assert summary.tool_result_count == 1


@pytest.mark.anyio
async def test_subagent_panel_uses_thread_table_as_source(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    for index, status in enumerate(("running", "completed", "failed"), 1):
        store.create_thread(
            session_id=metadata.session_id,
            thread_id=f"child-thread-{index}",
            agent_id=f"child-agent-{index}",
            role="subagent",
            title=f"Child {index}",
            status=status,
            parent_thread_id=MAIN_THREAD_ID,
            metadata={
                "instructions": f"Ask {index}",
                "context_brief": f"Context {index}",
                "allowed_tools": ["read_text_file"],
            },
        )
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        app.action_show_subagents()
        subagents = app.query_one(SubagentStatusWidget)

        assert not subagents.has_class("hidden")
        assert "subagents  3 total | 1 running | 1 done | 1 failed" in subagents.renderable
        assert "Child 1" in subagents.renderable
        app.action_show_subagents()
        assert subagents.has_class("hidden")
        app.action_show_subagents()
        assert not subagents.has_class("hidden")
        subagents.next_page()
        assert "Child 2" in subagents.renderable
        subagents.next_page()
        assert "Child 3" in subagents.renderable


@pytest.mark.anyio
async def test_subagent_panel_uses_event_log_inbox_as_source(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    store.create_thread(
        session_id=metadata.session_id,
        thread_id="child-thread-1",
        agent_id="child-agent-1",
        role="subagent",
        title="Child 1",
        status="running",
        parent_thread_id=MAIN_THREAD_ID,
    )
    recorder = SessionRecorder(store, metadata.session_id)
    recorder.record(
        SessionEvent(
            event_id="inbox-1",
            thread_id="child-thread-1",
            agent_id="main-agent",
            kind="agent_inbox_item",
            payload={
                "source_thread_id": MAIN_THREAD_ID,
                "source_agent_id": "main-agent",
                "severity": "blocked",
                "message": "parent update",
                "status": "pending",
            },
        )
    )
    app = AceAITUI(
        [],
        session_recorder=recorder,
        session_id=metadata.session_id,
    )

    async with app.run_test():
        app.action_show_subagents()
        subagents = app.query_one(SubagentStatusWidget)

        assert "Child 1" in subagents.renderable
        assert "inbox: 1" in subagents.renderable
        assert "parent update" in subagents.renderable


@pytest.mark.anyio
async def test_subagent_thread_options_are_cached_during_stream_refresh(
    tmp_path,
    monkeypatch,
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    store.create_thread(
        session_id=metadata.session_id,
        thread_id="child-thread-1",
        agent_id="child-agent-1",
        role="subagent",
        title="Child 1",
        status="running",
        parent_thread_id=MAIN_THREAD_ID,
    )
    calls = 0
    original_list_threads = store.list_threads

    def list_threads(session_id: str):
        nonlocal calls
        calls += 1
        return original_list_threads(session_id)

    monkeypatch.setattr(store, "list_threads", list_threads)
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )

    async with app.run_test():
        app.action_show_subagents()

        assert calls == 1

        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="Hello"))
        )
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta=" world"))
        )

        assert calls == 1

        app.append_event(TUIEvent.from_agent_event(builder.tool_started(tool_call=call)))
        app.append_event(
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=call,
                    tool_result=ToolExecutionResult(call=call, output="done"),
                )
            )
        )

        assert calls == 2


@pytest.mark.anyio
async def test_subagent_status_widget_scrolls_with_mouse_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect a very long delegated report",'
            '"instructions":"report every relevant observation",'
            '"context_brief":"'
            + "\\n".join(f"context line {index}" for index in range(80))
            + '","allowed_tools":[]}'
        ),
        call_id="call-subagent-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"type":"subagent_audit","thread_id":"child-thread-1",'
            '"agent_id":"child-1","run_id":"run-1","status":"completed",'
            '"summary":"","step_count":1}'
        ),
    )
    app = AceAITUI(
        [
            TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
            TUIEvent.from_agent_event(
                builder.tool_completed(tool_call=call, tool_result=result)
            ),
        ],
    )

    async with app.run_test(size=(100, 30)) as pilot:
        app.action_show_subagents()
        subagents = app.query_one(SubagentStatusWidget)
        detail = subagents.query_one("#subagent-detail", RichLog)
        scroll_calls: list[dict[str, object]] = []

        def record_scroll_relative(
            *,
            y: int,
            animate: bool,
            force: bool,
            immediate: bool,
        ) -> None:
            scroll_calls.append(
                {
                    "y": y,
                    "animate": animate,
                    "force": force,
                    "immediate": immediate,
                }
            )

        monkeypatch.setattr(detail, "scroll_relative", record_scroll_relative)

        await subagents._dispatch_message(
            MouseScrollDown(
                subagents,
                x=1,
                y=1,
                delta_x=0,
                delta_y=1,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        await pilot.pause()

        assert scroll_calls == [
            {
                "y": 3,
                "animate": False,
                "force": True,
                "immediate": True,
            }
        ]


@pytest.mark.anyio
async def test_debug_mode_is_unavailable_while_subagents_are_visible() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect version metadata",'
            '"instructions":"report evidence","context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-subagent-1",
    )
    app = AceAITUI(
        [TUIEvent.from_agent_event(builder.tool_started(tool_call=call))],
    )

    async with app.run_test() as pilot:
        await pilot.press("d")

        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)

        assert not stream.debug_mode
        assert detail.has_class("collapsed")


@pytest.mark.anyio
async def test_debug_mode_is_unavailable_until_run_completes() -> None:
    app = AceAITUI([TUIEvent.user_message("hello")])

    async with app.run_test() as pilot:
        await pilot.press("d")

        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)

        assert not stream.debug_mode
        assert detail.has_class("collapsed")


@pytest.mark.anyio
async def test_debug_mode_stream_selection_opens_tool_result_detail() -> None:
    events = static_demo_events()
    tool_event = _first_event(events, "tool_completed")
    app = AceAITUI(events)

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)
        app.action_hide_subagents()
        stream.post_message(StreamWidget.EventSelected(tool_event.event_id))
        await pilot.pause()

        detail = app.query_one(DetailWidget)

        assert app._state.selected_event_id == tool_event.event_id
        assert not detail.has_class("collapsed")


@pytest.mark.anyio
async def test_debug_mode_reuses_message_panel_and_opens_selected_detail() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)

        app.action_hide_subagents()
        await pilot.press("d")

        assert stream.debug_mode
        assert stream.has_focus
        assert not detail.has_class("collapsed")
        assert app._state.selected_event_id is not None


@pytest.mark.anyio
async def test_debug_detail_panel_scrolls_with_page_keys() -> None:
    long_notice = "\n".join(f"line {index}" for index in range(80))
    app = AceAITUI(
        [
            TUIEvent.session_notice(long_notice),
            TUIEvent(
                kind="run_completed",
                step_index=-1,
                step_id="run",
                title="run completed",
                content="done",
                raw_event=None,
            ),
        ]
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("d")
        await pilot.pause()

        detail = app.query_one(DetailWidget)
        assert detail.max_scroll_y > 0
        assert detail.scroll_y == 0

        await pilot.press("pagedown")
        await pilot.pause()
        assert detail.scroll_y > 0

        detail.focus()
        await pilot.press("pageup")
        await pilot.pause()
        assert detail.scroll_y == 0


@pytest.mark.anyio
async def test_debug_detail_panel_scrolls_with_mouse_wheel() -> None:
    long_notice = "\n".join(f"line {index}" for index in range(80))
    app = AceAITUI(
        [
            TUIEvent.session_notice(long_notice),
            TUIEvent(
                kind="run_completed",
                step_index=-1,
                step_id="run",
                title="run completed",
                content="done",
                raw_event=None,
            ),
        ]
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("d")
        await pilot.pause()

        detail = app.query_one(DetailWidget)
        assert detail.max_scroll_y > 0
        await detail._dispatch_message(
            MouseScrollDown(
                detail,
                x=1,
                y=1,
                delta_x=0,
                delta_y=1,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        await pilot.pause()
        assert detail.scroll_y > 0

        await detail._dispatch_message(
            MouseScrollUp(
                detail,
                x=1,
                y=1,
                delta_x=0,
                delta_y=-1,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        await pilot.pause()
        assert detail.scroll_y == 0


@pytest.mark.anyio
async def test_debug_mode_can_move_selection_inside_message_panel() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)

        app.action_hide_subagents()
        await pilot.press("d")
        first_selected = app._state.selected_event_id
        await pilot.press("down")

        assert stream.debug_mode
        assert app._state.selected_event_id != first_selected


@pytest.mark.anyio
async def test_debug_mode_click_selects_message_in_main_stream() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)

        app.action_hide_subagents()
        await pilot.press("d")
        assert len(stream._debug_line_spans) >= 2
        target = stream._debug_line_spans[1]
        stream.on_click(
            Click(
                stream,
                x=1,
                y=target.start_line - stream.scroll_y,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        await pilot.pause()

        assert app._state.selected_event_id == target.event_id
        assert not app.query_one(DetailWidget).has_class("collapsed")


@pytest.mark.anyio
async def test_trajectory_screen_renders_event_trajectory() -> None:
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"letters.txt","content":"abc"}',
        call_id="call-1",
    )
    events = [
        TUIEvent.user_message("write a file"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-1",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-1",
            title="assistant",
            content="I will write it.",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_started",
            step_index=0,
            step_id="step-1",
            title="tool write_text_file",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_approval_requested",
            step_index=0,
            step_id="step-1",
            title="tool write_text_file approval",
            content="approval required",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-1",
            title="tool write_text_file completed",
            content='{"path":"letters.txt","bytes_written":3}',
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output='{"path":"letters.txt","bytes_written":3}',
            ),
            raw_event=None,
        ),
        TUIEvent(
            kind="step_completed",
            step_index=0,
            step_id="step-1",
            title="step completed",
            raw_event=None,
        ),
        TUIEvent(
            kind="run_completed",
            step_index=0,
            step_id="step-1",
            title="run completed",
            content="I will write it.",
            raw_event=None,
        ),
        TUIEvent.user_message("what happened?"),
        TUIEvent(
            kind="step_started",
            step_index=1,
            step_id="step-2",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="run_completed",
            step_index=1,
            step_id="step-2",
            title="run completed",
            content="The file was written.",
            raw_event=None,
        ),
    ]
    app = AceAITUI(events)

    async with app.run_test() as pilot:
        app.open_trajectory_screen()
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, TrajectoryScreen)

        rendered = _render_to_text(Group(*_trajectory_renderables(events)))

        assert "Trajectory" in rendered
        assert "turns" in rendered
        assert "steps" in rendered
        assert "events" in rendered
        assert "tool calls" in rendered
        assert "approvals" in rendered
        assert "failures" in rendered
        assert "T   2" not in rendered
        assert "E   11" not in rendered
        assert " 1  write a file" in rendered
        assert " 2  what happened?" in rendered
        assert "▌ 1" in rendered
        assert "▌ 2" in rendered
        assert "│ call" in rendered
        assert "└ result" in rendered
        assert "◆" in rendered
        assert "answer" in rendered
        assert "TURN" not in rendered
        assert "QUESTION" not in rendered
        assert "STEP" not in rendered
        assert "OUTCOME" not in rendered
        assert "write a file" in rendered
        assert "what happened?" in rendered
        assert "I will write it." in rendered
        assert rendered.count("I will write it.") == 1
        assert "approval required" in rendered
        assert "write_text_file" in rendered
        assert "wrote 3 bytes" in rendered
        assert "The file was written." in rendered
        assert "I will write it. - I will write it." not in rendered


def test_trajectory_summarizes_shell_tool_result_output() -> None:
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"ls","cwd":"/tmp","timeout_seconds":10}',
        call_id="call-shell",
    )
    events = [
        TUIEvent.user_message("run ls"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-shell",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_started",
            step_index=0,
            step_id="step-shell",
            title="tool run_shell_command",
            tool_name="run_shell_command",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-shell",
            title="tool run_shell_command completed",
            content='{"command":"ls","cwd":"/tmp","exit_code":0,"stdout":"a.py\\nb.py\\n","stderr":""}',
            tool_name="run_shell_command",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output='{"command":"ls","cwd":"/tmp","exit_code":0,"stdout":"a.py\\nb.py\\n","stderr":""}',
            ),
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "$ ls" in rendered
    assert "a.py" in rendered
    assert '"stdout"' not in rendered
    assert '"exit_code"' not in rendered


def test_trajectory_renders_plain_text_tool_result_output() -> None:
    call = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"tests/test_ace_agent.py"}',
        call_id="call-replace",
    )
    events = [
        TUIEvent.user_message("patch file"),
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-replace",
            title="tool replace_text_in_file completed",
            content=(
                "the tool replace_text_in_file exceeds its max calls in this run, "
                "do not call it again"
            ),
            tool_name="replace_text_in_file",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output=(
                    "the tool replace_text_in_file exceeds its max calls in this run, "
                    "do not call it again"
                ),
            ),
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "replace_text_in_file" in rendered
    assert "exceeds its max calls" in rendered


def test_trajectory_does_not_repeat_streamed_answer_on_llm_completion() -> None:
    events = [
        TUIEvent.user_message("explain it"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-stream",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-stream",
            title="assistant",
            content="streamed answer",
            raw_event=None,
        ),
        TUIEvent(
            kind="llm_completed",
            step_index=0,
            step_id="step-stream",
            title="llm completed",
            content="streamed answer",
            raw_event=None,
        ),
        TUIEvent(
            kind="step_completed",
            step_index=0,
            step_id="step-stream",
            title="step completed",
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert rendered.count("streamed answer") == 1
    assert "llm completed" in rendered


def test_trajectory_compacts_streamed_tool_call_arguments() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-arguments")
    events = [
        TUIEvent.user_message("delegate check"),
        TUIEvent.from_agent_event(builder.llm_started()),
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-delegate",
                    arguments_delta='{"task":"检查',
                )
            )
        ),
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-delegate",
                    arguments_delta="当前项目",
                )
            )
        ),
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-delegate",
                    arguments_delta=' README"}',
                )
            )
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert rendered.count("arguments") == 1
    assert '{"task":"检查当前项目 README"}' in rendered


def test_trajectory_renders_session_notices_without_running_step() -> None:
    events = [
        TUIEvent.session_notice("Resumed session abc"),
        TUIEvent.session_notice("Switched model to gpt-5.5"),
        TUIEvent.user_message("hello"),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "session" in rendered
    assert "Resumed session abc" in rendered
    assert "Switched model to gpt-5.5" in rendered
    assert "running" not in rendered
    assert "▌ -" not in rendered


def test_trajectory_marks_multiline_preview_as_truncated() -> None:
    events = [
        TUIEvent.user_message("show result"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-show",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-show",
            title="assistant",
            content="结果如下：\n\naceai\nAGENTS.md\nREADME.md",
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "结果如下：" in rendered
    assert "... (+4 lines)" in rendered
    assert "aceai\n" not in rendered
    assert "AGENTS.md" not in rendered
    assert "README.md" not in rendered


def test_trajectory_distinguishes_rejected_approval_from_tool_failure() -> None:
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"x","content":"hello"}',
        call_id="call-rejected",
    )
    events = [
        TUIEvent.user_message("write it"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-rejected",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_approval_requested",
            step_index=0,
            step_id="step-rejected",
            title="tool write_text_file approval",
            content="Tool 'write_text_file' requires approval",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_approval_resolved",
            step_index=0,
            step_id="step-rejected",
            title="tool write_text_file approval resolved",
            content="rejected: rejected by caller",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_failed",
            step_index=0,
            step_id="step-rejected",
            title="tool write_text_file failed",
            content="rejected by caller",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output="Tool execution rejected: rejected by caller",
                error="rejected by caller",
            ),
            error="rejected by caller",
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "rejected    1" in rendered
    assert "failures    0" in rendered
    assert "rejected" in rendered


def test_trajectory_marks_suspended_step_as_waiting() -> None:
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"rm x","cwd":".","timeout_seconds":10}',
        call_id="call-waiting",
    )
    events = [
        TUIEvent.user_message("delete it"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-waiting",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-waiting",
            title="assistant",
            content="I need approval.",
            raw_event=None,
        ),
        TUIEvent(
            kind="run_suspended",
            step_index=0,
            step_id="step-waiting",
            title="run suspended",
            content="waiting for approval. Choose Approve or Reject.",
            tool_name="run_shell_command",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "waiting" in rendered
    assert "running  step-w" not in rendered


@pytest.mark.anyio
async def test_detail_renders_tool_arguments_and_output() -> None:
    call = LLMToolCall(
        name="search_docs",
        arguments='{"query":"aceai tui"}',
        call_id="call-1234567890",
    )
    tool_event = TUIEvent(
        kind="tool_completed",
        step_index=0,
        step_id="step-1",
        title="tool search_docs",
        raw_event=None,
        content='{"matches":["spec/tui.md","docs/tui.md"]}',
        tool_name="search_docs",
        tool_call_id=call.call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(
            call=call,
            output='{"matches":["spec/tui.md","docs/tui.md"]}',
        ),
    )
    app = AceAITUI([tool_event])

    async with app.run_test():
        app._state = select_event(app._state, tool_event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail._render_detail())

        assert '"query": "aceai tui"' in rendered
        assert '{"matches":["spec/tui.md","docs/tui.md"]}' in rendered
        assert "TOOL CALL" in rendered
        assert "RESULT" in rendered
        assert "arguments{" not in rendered
        assert "output{" not in rendered


@pytest.mark.anyio
async def test_detail_omits_empty_raw_event_section() -> None:
    event = TUIEvent.user_message("show the readable part")
    app = AceAITUI([event])

    async with app.run_test():
        app._state = select_event(app._state, event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail._render_detail())

        assert "CONTENT" in rendered
        assert "show the readable part" in rendered
        assert "raw event" not in rendered
        assert "None" not in rendered


@pytest.mark.anyio
async def test_detail_shortens_long_ids() -> None:
    event = TUIEvent(
        kind="session_notice",
        step_index=0,
        step_id="12345678-1234-1234-1234-123456789abc",
        title="session",
        content="notice",
        raw_event=None,
    )
    app = AceAITUI([event])

    async with app.run_test():
        app._state = select_event(app._state, event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail._render_detail())

        assert "12345678...9abc" in rendered
        assert "12345678-1234-1234-1234-123456789abc" not in rendered


@pytest.mark.anyio
async def test_detail_renders_errors_as_separate_section() -> None:
    event = TUIEvent(
        kind="run_failed",
        step_index=0,
        step_id="step-1",
        title="run failed",
        content="the run failed",
        error="boom",
        raw_event=None,
    )
    app = AceAITUI([event])

    async with app.run_test():
        app._state = select_event(app._state, event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail._render_detail())

        assert "ERROR" in rendered
        assert "boom" in rendered


@pytest.mark.anyio
async def test_tui_batches_small_stream_delta_refreshes() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    app = AceAITUI([])

    async with app.run_test() as pilot:
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="hello "))
        )
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="world"))
        )

        assert refreshes == [1]
        assert len(app._state.events) == 1
        assert app._state.events[0].content == "hello "

        await _wait_until(
            pilot,
            lambda: len(refreshes) == 2,
            timeout=STREAM_DELTA_REFRESH_SECONDS * 50,
        )

        assert refreshes == [1, 1]
        assert app._state.events[0].content == "hello world"

        app.append_event(
            TUIEvent.from_agent_event(
                builder.llm_text_delta(text_delta="x" * STREAM_DELTA_REFRESH_CHARS)
            )
        )

        assert refreshes == [1, 1, 1]
        assert len(app._state.events) == 1
        assert app._state.events[0].content == "hello world" + (
            "x" * STREAM_DELTA_REFRESH_CHARS
        )


@pytest.mark.anyio
async def test_tui_refreshes_first_stream_delta_immediately() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    app = AceAITUI([])

    async with app.run_test():
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="h"))
        )

        assert refreshes == [1]
        assert app._state.events[0].content == "h"


@pytest.mark.anyio
async def test_tui_flushes_pending_stream_delta_on_completion() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    step = AgentStep(llm_response=LLMResponse(text="done"))
    app = AceAITUI([])

    async with app.run_test():
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="done"))
        )
        app.append_event(
            TUIEvent.from_agent_event(
                builder.run_completed(step=step, final_answer="done")
            )
        )

        assert refreshes == [1, 2]


@pytest.mark.anyio
async def test_stream_scrolls_to_latest_content() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [TUIEvent.user_message("Show me many lines")]
    for line_number in range(80):
        events.append(
            TUIEvent.from_agent_event(
                builder.llm_text_delta(text_delta=f"line {line_number}\n")
            )
        )
    app = AceAITUI(events)

    async with app.run_test(size=(80, 20)) as pilot:
        stream = app.query_one(StreamWidget)
        await _wait_until(
            pilot,
            lambda: stream.scroll_y == stream.max_scroll_y,
            timeout=1.0,
        )

        assert stream.max_scroll_y > 0
        assert stream.scroll_y == stream.max_scroll_y


@pytest.mark.anyio
async def test_stream_does_not_show_horizontal_scrollbar() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_text_delta(
                text_delta=(
                    "This is a long assistant response that should wrap inside "
                    "the stream instead of creating a horizontal scrollbar. " * 8
                )
            )
        )
    ]
    app = AceAITUI(events)

    async with app.run_test(size=(60, 20)) as pilot:
        await _wait_until(
            pilot,
            lambda: app.query_one(StreamWidget).max_scroll_x == 0,
        )
        stream = app.query_one(StreamWidget)

        assert not stream.show_horizontal_scrollbar
        assert stream.max_scroll_x == 0


@pytest.mark.anyio
async def test_escape_exits_input_mode_and_returns_focus_to_stream() -> None:
    app = AceAITUI([])

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        stream = app.query_one(StreamWidget)
        await pilot.pause()
        app.set_focus(command_input)
        await pilot.pause()

        assert command_input.has_focus

        await pilot.press("escape")

        assert not command_input.has_focus
        assert stream.has_focus


@pytest.mark.anyio
async def test_enter_from_main_view_focuses_input() -> None:
    app = AceAITUI([])

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        stream = app.query_one(StreamWidget)
        stream.focus()
        await pilot.pause()

        assert stream.has_focus

        await pilot.press("enter")

        assert command_input.has_focus


@pytest.mark.anyio
async def test_input_sits_at_bottom_without_footer() -> None:
    app = AceAITUI([])

    async with app.run_test(size=(80, 20)) as pilot:
        await _wait_until(
            pilot,
            lambda: app.query_one(CommandInput).region.height == 4,
        )
        command_input = app.query_one(CommandInput)

        assert command_input.region.height == 4
        assert command_input.region.y + command_input.region.height <= app.size.height


@pytest.mark.anyio
async def test_tui_header_uses_session_id(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        assert app.title == f"AceAI {metadata.project_name} {metadata.session_id}"


@pytest.mark.anyio
async def test_empty_tui_exit_does_not_create_session_store(monkeypatch) -> None:
    class FailingSessionStore:
        def __init__(self) -> None:
            raise AssertionError("empty TUI exit should not touch the session store")

    monkeypatch.setattr(tui_app_module, "SessionStore", FailingSessionStore)
    app = AceAITUI([])

    async with app.run_test():
        pass


@pytest.mark.anyio
async def test_tui_can_show_and_switch_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    _record_user_message(store, first.session_id, "first question")
    _record_user_message(store, second.session_id, "second question")
    store.create_thread(
        session_id=second.session_id,
        thread_id="child-thread-1",
        agent_id="child-agent-1",
        role="subagent",
        title="Inspect child",
        parent_thread_id=MAIN_THREAD_ID,
    )
    store.append_event(
        second.session_id,
        SessionEvent(
            kind="user_message",
            payload={"content": "child question"},
            thread_id="child-thread-1",
            agent_id="child-agent-1",
            run_id="child-run-1",
        ),
    )
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test():
        app.show_sessions()

        assert app._state.events[-1].kind == "session_notice"
        assert "Total cost: $0.000000" in app._state.events[-1].content
        assert first.session_id in app._state.events[-1].content
        assert second.session_id in app._state.events[-1].content

        app.switch_session(second.session_id)

        assert app.title == f"AceAI {second.project_name} {second.session_id}"
        assert app._state.events[0].content == "second question"
        assert [event.content for event in app._state.events] == ["second question"]


@pytest.mark.anyio
async def test_tui_replayed_incomplete_session_does_not_show_running(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create_session()
    store.append_event(
        session.session_id,
        SessionEvent(
            kind="thinking_delta",
            step_id="step-1",
            step_index=0,
            payload={"content": "thinking"},
        ),
    )
    store.append_event(
        session.session_id,
        SessionEvent(
            kind="assistant_message",
            step_id="step-1",
            step_index=0,
            payload={"content": "partial answer"},
        ),
    )
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(session.session_id)),
        session_recorder=SessionRecorder(store, session.session_id),
        session_id=session.session_id,
    )

    async with app.run_test():
        assert app._state.status == "idle"


@pytest.mark.anyio
async def test_tui_switching_to_current_session_is_noop_for_empty_session(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    current = store.create_session()
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, current.session_id),
        session_id=current.session_id,
    )

    async with app.run_test():
        app.switch_session(current.session_id)

        assert app._session_id == current.session_id
        assert store.get_session(current.session_id).session_id == current.session_id


@pytest.mark.anyio
async def test_tui_switching_to_missing_session_keeps_current_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    current = store.create_session()
    _record_user_message(store, current.session_id, "current question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(current.session_id)),
        session_recorder=SessionRecorder(store, current.session_id),
        session_id=current.session_id,
    )

    async with app.run_test():
        event_count = len(app._state.events)
        app.switch_session("missing-session")

        assert app._session_id == current.session_id
        assert len(app._state.events) == event_count


@pytest.mark.anyio
async def test_session_selector_uses_panel_list(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    store.update_session_title(first.session_id, "first question - 2026-05-04 12:13:14")
    store.update_session_title(second.session_id, "second question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test() as pilot:
        app.open_session_selector()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (
                app.screen.query_one(
                    "#session-list",
                    SessionListWidget,
                ).selected_session_id()
                == first.session_id
            ),
        )

        session_list = app.screen.query_one("#session-list", SessionListWidget)
        rendered = _render_to_text(
            Group(
                *_session_picker_renderables(
                    session_list.sessions(),
                    current_session_id=first.session_id,
                    selected_index=session_list.selected_index,
                )
            )
        )

        assert session_list.selected_session_id() == first.session_id
        assert "first question" in rendered
        assert "second question" in rendered
        assert "aceai  2" in rendered
        assert "created" in rendered
        assert first.session_id in rendered
        status = app.screen.query_one("#session-status", Static)
        assert "Press a to create" in str(status.render())
        assert "Total cost: $0.000000" in str(status.render())


@pytest.mark.anyio
async def test_session_selector_creates_and_switches_to_new_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    _record_user_message(store, first.session_id, "first question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test() as pilot:
        app.open_session_selector()
        await pilot.pause()

        await pilot.press("a")
        await _wait_until(pilot, lambda: app._session_id != first.session_id)

        assert app._session_id is not None
        assert app._state.events == []
        assert store.get_session(app._session_id).title == "New session"
        assert [session.session_id for session in store.list_sessions()] == [
            app._session_id,
            first.session_id,
        ]


@pytest.mark.anyio
async def test_session_selector_scrolls_highlighted_row_into_view(tmp_path) -> None:
    store = SessionStore(tmp_path)
    sessions = [store.create_session() for _index in range(18)]
    first = sessions[0]
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test(size=(100, 20)) as pilot:
        app.open_session_selector()
        await pilot.pause()
        session_list = app.screen.query_one("#session-list", SessionListWidget)
        scroll = app.screen.query_one("#session-list-scroll", VerticalScroll)

        for _index in range(17):
            await pilot.press("down")

        await _wait_until(pilot, lambda: session_list.selected_index == 17)
        await _wait_until(pilot, lambda: scroll.scroll_y > 0)

        selected_top = session_list._selected_item_top()
        assert scroll.scroll_y <= selected_top
        assert selected_top < scroll.scroll_y + scroll.scrollable_content_region.height


@pytest.mark.anyio
async def test_session_selector_deletes_highlighted_session_after_confirmation(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    store.update_session_title(first.session_id, "first question")
    store.update_session_title(second.session_id, "second question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test() as pilot:
        app.open_session_selector()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (
                app.screen.query_one(
                    "#session-list",
                    SessionListWidget,
                ).selected_session_id()
                == first.session_id
            ),
        )

        session_list = app.screen.query_one("#session-list", SessionListWidget)
        second_index = _session_widget_index(session_list, second.session_id)
        session_list.move_selection(second_index - session_list.selected_index)

        await pilot.press("d")
        await _wait_until(
            pilot,
            lambda: (
                second.session_id
                in str(app.screen.query_one("#delete-session-message", Static).content)
            ),
        )

        message = app.screen.query_one("#delete-session-message", Static)
        assert second.session_id in str(message.content)

        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                second.session_id
                not in [
                    session.session_id
                    for session in app.screen.query_one(
                        "#session-list",
                        SessionListWidget,
                    ).sessions()
                ]
            ),
        )

        session_list = app.screen.query_one("#session-list", SessionListWidget)
        rendered = _render_to_text(
            Group(
                *_session_picker_renderables(
                    session_list.sessions(),
                    current_session_id=first.session_id,
                    selected_index=session_list.selected_index,
                )
            )
        )
        assert second.session_id not in rendered
        assert [session.session_id for session in store.list_sessions()] == [
            first.session_id
        ]


def _session_widget_index(widget: SessionListWidget, session_id: str) -> int:
    for index, session in enumerate(widget.sessions()):
        if session.session_id == session_id:
            return index
    raise ValueError(session_id)


def _idea(content: str) -> Idea:
    return Idea(
        idea_id="idea-1",
        created_at=datetime(2026, 5, 6, 11, 13, tzinfo=timezone.utc),
        project_id="project-1",
        project_name="aceai",
        workspace="/tmp/aceai",
        content=content,
    )


def _first_event(events: list[TUIEvent], kind: str) -> TUIEvent:
    for event in events:
        if event.kind == kind:
            return event
    raise ValueError(kind)


def _render_to_text(renderable) -> str:
    console = Console(file=StringIO(), record=True, width=120)
    console.print(renderable)
    return console.export_text()


def _record_user_message(store: SessionStore, session_id: str, content: str) -> None:
    SessionRecorder(store, session_id).record(
        tui_event_to_session_event(TUIEvent.user_message(content))
    )


def _tool_completed_event(tool_name: str, call_id: str) -> TUIEvent:
    call = LLMToolCall(
        name=tool_name,
        arguments='{"path":"README.md"}',
        call_id=call_id,
    )
    return TUIEvent(
        kind="tool_completed",
        step_index=0,
        step_id=f"step-{call_id}",
        title=f"tool {tool_name} completed",
        content="ok",
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(call=call, output="ok"),
        raw_event=None,
    )


def _tool_failed_event(tool_name: str, call_id: str) -> TUIEvent:
    call = LLMToolCall(
        name=tool_name,
        arguments='{"command":"bad"}',
        call_id=call_id,
    )
    return TUIEvent(
        kind="tool_failed",
        step_index=0,
        step_id=f"step-{call_id}",
        title=f"tool {tool_name} failed",
        content="failed",
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(call=call, output="failed", error="failed"),
        error="failed",
        raw_event=None,
    )


def _skill_completed_event(skill_name: str, call_id: str) -> TUIEvent:
    call = LLMToolCall(
        name="skill_view",
        arguments=json.dumps({"name": skill_name}),
        call_id=call_id,
    )
    return TUIEvent(
        kind="tool_completed",
        step_index=0,
        step_id=f"step-{call_id}",
        title="tool skill_view completed",
        content="ok",
        tool_name="skill_view",
        tool_call_id=call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(call=call, output="ok"),
        raw_event=None,
    )


def _skill_failed_event(skill_name: str, call_id: str) -> TUIEvent:
    call = LLMToolCall(
        name="skill_view",
        arguments=json.dumps({"name": skill_name}),
        call_id=call_id,
    )
    return TUIEvent(
        kind="tool_failed",
        step_index=0,
        step_id=f"step-{call_id}",
        title="tool skill_view failed",
        content="failed",
        tool_name="skill_view",
        tool_call_id=call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(call=call, output="failed", error="failed"),
        error="failed",
        raw_event=None,
    )
