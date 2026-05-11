from io import StringIO

from rich.console import Group
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from textual.events import Click
from textual.strip import Strip

from aceai.agent.session import EventLog, SessionEvent
from aceai.agent.citations import (
    ConversationCitationOrigin,
    FileCitationOrigin,
    TurnCitation,
)
from aceai.core.events import AgentEventBuilder
from aceai.llm.models import (
    LLMMessage,
    LLMReasoningSegmentMeta,
    LLMResponse,
    LLMSegment,
    LLMToolCall,
    LLMToolCallDelta,
)
from aceai.core.models import AgentStep
from aceai.core.models import ToolApprovalRequest
from aceai.core.models import ToolExecutionResult
from aceai.agent.tui.events import TUIEvent, TUIIdeaItem
from aceai.agent.tui.session_replay import event_log_to_tui_events
from aceai.agent.tui.state import TUIRunState, reduce_events
from aceai.agent.tui.widgets.stream import (
    StreamWidget,
    _PromptBar,
    _render_events,
)


def _stream_set_state_writes(state: TUIRunState) -> list[Text]:
    stream = StreamWidget()
    writes: list[Text] = []
    _capture_stream_writes(stream, writes)
    stream.set_state(state)
    return writes


def _capture_stream_writes(stream: StreamWidget, writes: list[Text]) -> None:
    def fake_write(
        content: object,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> StreamWidget:
        if isinstance(content, Text):
            writes.append(content)
        stream.lines.append(Strip.blank(1))
        return stream

    def fake_clear() -> StreamWidget:
        writes.clear()
        stream.lines.clear()
        return stream

    def fake_call_after_refresh(
        callback: object,
        *args: object,
        animate: bool = False,
    ) -> bool:
        return True

    stream.write = fake_write
    stream.clear = fake_clear
    stream.call_after_refresh = fake_call_after_refresh


def test_consecutive_assistant_deltas_render_as_one_block() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="Hi")),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="!")),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    block = renderables[0]
    assert isinstance(block, Text)
    assert block.plain == "  Hi!"


def test_agent_inbox_item_renders_without_missing_label() -> None:
    renderables = _render_events(
        [
            TUIEvent(
                kind="agent_inbox_item",
                step_index=-1,
                step_id="step-inbox",
                title="inbox",
                raw_event=None,
                content=(
                    "Background subagent completed.\n"
                    "Task: inspect app.py\n"
                    "Summary:\nlong details"
                ),
            )
        ]
    )

    assert len(renderables) == 1
    assert isinstance(renderables[0], Text)
    assert "Background subagent completed." in renderables[0].plain
    assert "Task: inspect app.py" in renderables[0].plain
    assert "Summary:" not in renderables[0].plain


def test_agent_inbox_item_renders_in_completed_collapsed_stream() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.user_message("check background work"),
        TUIEvent(
            kind="agent_inbox_item",
            step_index=-1,
            step_id="step-inbox",
            title="inbox",
            raw_event=None,
            content="Background subagent job abc completed.",
        ),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="done")),
                final_answer="done",
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 4
    assert isinstance(renderables[2], Text)
    assert renderables[2].plain == "  ─ [+] work history · 1 inbox"


def test_agent_inbox_item_does_not_split_completed_work_history() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first_call = LLMToolCall(
        name="search_text",
        arguments='{"query":"a"}',
        call_id="call-1",
    )
    second_call = LLMToolCall(
        name="wait_subagent",
        arguments='{"job_id":"job-1"}',
        call_id="call-2",
    )
    events = [
        TUIEvent.user_message("check background work"),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first_call)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first_call,
                tool_result=ToolExecutionResult(call=first_call, output="ready"),
            )
        ),
        TUIEvent(
            kind="agent_inbox_item",
            step_index=-1,
            step_id="step-inbox",
            title="inbox",
            raw_event=None,
            content=(
                "Background subagent job abc completed.\n"
                "Task: inspect app.py\n"
                "Summary:\nlong details"
            ),
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second_call)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second_call,
                tool_result=ToolExecutionResult(call=second_call, output="done"),
            )
        ),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="done")),
                final_answer="done",
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    work_history = [
        renderable
        for renderable in renderables
        if isinstance(renderable, Text) and "work history" in renderable.plain
    ]
    assert len(work_history) == 1
    assert work_history[0].plain == "  ─ [+] work history · 2 tool calls · 1 inbox"


def test_agent_inbox_delivered_is_hidden_from_stream() -> None:
    assert _render_events(
        [
            TUIEvent(
                kind="agent_inbox_delivered",
                step_index=-1,
                step_id="step-inbox",
                title="inbox",
                raw_event=None,
                content="inbox-1",
            )
        ]
    ) == []


def test_context_compaction_events_render_progress_and_summary() -> None:
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
                history=[
                    LLMMessage.build(
                        role="system",
                        content=(
                            '<aceai_context_summary scope="prior_runs">\n'
                            "Earlier decisions and tool results.\n"
                            "</aceai_context_summary>"
                        ),
                    )
                ],
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

    renderables = _render_events(events)
    texts = [renderable.plain for renderable in renderables if isinstance(renderable, Text)]

    assert "  ● compact  Compacting context (preflight budget)..." in texts
    assert any("Summary is available in details." in text for text in texts)
    assert not any("Earlier decisions and tool results." in text for text in texts)
    assert any("summary request exceeded context window" in text for text in texts)


def test_llm_retry_progress_renders_in_stream() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_retrying(
                retry_count=1,
                retry_max=2,
                retry_delay_seconds=0.5,
                error="RemoteProtocolError: peer closed",
            )
        )
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert (
        text.plain
        == "  ● retry  Retrying message 1/2 in 0.5s after RemoteProtocolError: peer closed"
    )


def test_main_stream_renders_question_before_answer() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.user_message("How do I search?"),
        TUIEvent.from_agent_event(builder.llm_started()),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="Use rg.")),
        TUIEvent.from_agent_event(
            builder.step_completed(step=AgentStep(llm_response=LLMResponse(text="Use rg.")))
        ),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="Use rg.")),
                final_answer="Use rg.",
            )
        ),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 2
    question = renderables[0]
    answer = renderables[1]
    assert isinstance(question, _PromptBar)
    assert isinstance(answer, Text)
    _assert_prompt_bar_contains(question, "▌ How do I search?")
    assert answer.plain == "  Use rg."


def test_user_messages_render_as_taller_prompt_bar() -> None:
    renderables = _render_events([TUIEvent.user_message("Where am I?")])

    message = renderables[0]
    assert isinstance(message, _PromptBar)
    lines = _render_prompt_bar_plain_lines(message, width=24)

    assert lines == [
        "▌                       ",
        "▌ Where am I?           ",
        "▌                       ",
    ]


def test_wrapped_user_message_keeps_continuous_left_prompt_bar() -> None:
    renderables = [
        TUIEvent.user_message(
            "Use @spec/multi-agent/agent_inbox.md and explain why the file is cited.",
        )
    ]
    message = _render_events(renderables)[0]
    assert isinstance(message, _PromptBar)

    lines = _render_prompt_bar_plain_lines(message, width=30)

    assert len(lines) > 3
    assert all(line.startswith("▌ ") for line in lines)


def test_user_message_citations_render_as_separate_source_block() -> None:
    renderables = _render_events(
        [
            TUIEvent.user_message(
                "Explain it",
                citations=(
                    TurnCitation(
                        quote="The job is pending.",
                        origin=ConversationCitationOrigin(
                            kind="conversation",
                            event_id="event-1",
                            role="assistant",
                            span_start=0,
                            span_end=19,
                        ),
                    ),
                ),
            )
        ]
    )

    question = renderables[0]
    assert isinstance(question, _PromptBar)
    assert len(question.citations) == 1
    assert question.citations[0].quote == "The job is pending."
    assert question.citations[0].origin.kind == "conversation"
    _assert_prompt_bar_contains(question, "▌ Explain it")


def test_file_citation_with_empty_content_renders_path_only() -> None:
    path = "/Users/raceychan/mylab/aceai/spec/multi-agent/agent_inbox.md"
    renderables = _render_events(
        [
            TUIEvent.user_message(
                "@spec/multi-agent/agent_inbox.md discuss this",
                citations=(
                    TurnCitation(
                        quote=path,
                        origin=FileCitationOrigin(kind="file", path=path),
                    ),
                ),
            )
        ]
    )

    question = renderables[0]
    assert isinstance(question, _PromptBar)
    assert len(question.citations) == 1
    assert question.citations[0].quote == path
    assert question.citations[0].origin.kind == "file"


def test_user_messages_after_answers_get_turn_spacing() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    renderables = _render_events(
        [
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="answer")),
            TUIEvent.user_message("next question"),
        ]
    )

    assert len(renderables) == 3
    spacer = renderables[1]
    assert isinstance(spacer, Text)
    assert spacer.plain == ""
    question = renderables[2]
    assert isinstance(question, _PromptBar)
    _assert_prompt_bar_contains(question, "▌ next question")


def test_idea_list_renders_as_separate_items_with_title_and_body() -> None:
    renderables = _render_events(
        [
            TUIEvent.idea_list(
                [
                    TUIIdeaItem(
                        index=1,
                        project_name="ioa",
                        created_at="2026-05-06 11:13",
                        title="Add a Learn button",
                        body="Select failed trajectories by default.",
                    ),
                    TUIIdeaItem(
                        index=2,
                        project_name="aceai",
                        created_at="2026-05-06 11:39",
                        title="Use obsidian as external knowledge base",
                    ),
                ]
            )
        ]
    )

    assert len(renderables) == 1
    group = renderables[0]
    assert isinstance(group, Group)
    assert len(group.renderables) == 3
    first_item = group.renderables[1]
    assert isinstance(first_item, Panel)
    assert isinstance(first_item.title, Text)
    assert "Add a Learn button" in first_item.title.plain
    assert "Select failed trajectories by default." in first_item.renderable.plain


def test_stream_writes_user_message_rows_expanded() -> None:
    stream = StreamWidget()
    writes: list[tuple[object, bool]] = []

    def fake_write(
        content: object,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> StreamWidget:
        writes.append((content, expand))
        return stream

    def fake_clear() -> StreamWidget:
        return stream

    def fake_call_after_refresh(
        callback: object,
        *args: object,
        animate: bool = False,
    ) -> bool:
        return True

    stream.write = fake_write
    stream.clear = fake_clear
    stream.call_after_refresh = fake_call_after_refresh

    stream.set_state(reduce_events([TUIEvent.user_message("Where am I?")]))

    assert len(writes) == 1
    content, expand = writes[0]
    assert isinstance(content, _PromptBar)
    assert expand


def _render_prompt_bar_plain_lines(message: _PromptBar, *, width: int) -> list[str]:
    console = Console(
        file=StringIO(),
        width=width,
        record=True,
        color_system=None,
    )
    console.print(message)
    return console.export_text().splitlines()


def _assert_prompt_bar_contains(message: _PromptBar, expected_line: str) -> None:
    lines = _render_prompt_bar_plain_lines(message, width=80)
    assert expected_line in [line.rstrip() for line in lines]


def test_stream_writes_assistant_messages_as_plain_text() -> None:
    stream = StreamWidget()
    writes: list[tuple[object, int | None]] = []

    def fake_write(
        content: object,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> StreamWidget:
        writes.append((content, width))
        return stream

    def fake_clear() -> StreamWidget:
        return stream

    def fake_call_after_refresh(
        callback: object,
        *args: object,
        animate: bool = False,
    ) -> bool:
        return True

    stream.write = fake_write
    stream.clear = fake_clear
    stream.call_after_refresh = fake_call_after_refresh

    event = TUIEvent.from_agent_event(
        AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
            text_delta="x" * 300
        )
    )
    stream.set_state(reduce_events([event]))

    assert len(writes) == 1
    content, width = writes[0]
    assert isinstance(content, Text)
    assert width == 1


def test_stream_collapses_tool_activity_only_after_completion() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output='{"content":"b"}'),
            )
        ),
    ]
    running_writes = _stream_set_state_writes(reduce_events(events))
    completed_writes = _stream_set_state_writes(
        reduce_events(
            [
                *events,
                TUIEvent.from_agent_event(
                    builder.run_completed(
                        step=AgentStep(llm_response=LLMResponse(text="done")),
                        final_answer="done",
                    )
                ),
            ]
        )
    )

    assert [text.plain for text in running_writes] == [
        '  ● read_text_file("a.py")  result ready',
        '  ● read_text_file("b.py")  result ready',
    ]
    assert [text.plain for text in completed_writes] == [
        "  ─ [+] work history · 2 tool calls",
        "  done",
    ]


def test_clicking_collapsed_tool_activity_expands_and_collapses() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    state = reduce_events(
        [
            TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=first,
                    tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
                )
            ),
            TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=second,
                    tool_result=ToolExecutionResult(call=second, output='{"content":"b"}'),
                )
            ),
            TUIEvent.from_agent_event(
                builder.run_completed(
                    step=AgentStep(llm_response=LLMResponse(text="done")),
                    final_answer="done",
                )
            ),
        ]
    )
    stream = StreamWidget()
    writes: list[Text] = []
    _capture_stream_writes(stream, writes)
    stream.set_state(state)

    stream.on_click(
        Click(
            stream,
            0,
            0,
            0,
            0,
            1,
            False,
            False,
            False,
            style=Style(meta={"tool_activity_id": "call-1|call-2"}),
        )
    )

    assert [text.plain for text in writes] == [
        "  ─ [-] work history · 2 tool calls",
        '    ● read_text_file("a.py")  result ready',
        '    ● read_text_file("b.py")  result ready',
        "  done",
    ]

    stream.on_click(
        Click(
            stream,
            0,
            0,
            0,
            0,
            1,
            False,
            False,
            False,
            style=Style(meta={"tool_activity_id": "call-1|call-2"}),
        )
    )

    assert [text.plain for text in writes] == [
        "  ─ [+] work history · 2 tool calls",
        "  done",
    ]


def test_expanded_working_history_preserves_reasoning_tool_order() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="search_text",
        arguments='{"query":"needle"}',
        call_id="call-2",
    )
    state = reduce_events(
        [
            TUIEvent.from_agent_event(
                builder.llm_reasoning(
                    segment=LLMSegment(
                        type="reasoning",
                        content="think first",
                        meta=LLMReasoningSegmentMeta(
                            item_id="reasoning-1",
                            kind="content",
                            index=0,
                            is_delta=True,
                        ),
                    )
                )
            ),
            TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=first,
                    tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
                )
            ),
            TUIEvent.from_agent_event(
                builder.llm_reasoning(
                    segment=LLMSegment(
                        type="reasoning",
                        content="think second",
                        meta=LLMReasoningSegmentMeta(
                            item_id="reasoning-2",
                            kind="content",
                            index=0,
                            is_delta=True,
                        ),
                    )
                )
            ),
            TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
            TUIEvent.from_agent_event(
                builder.tool_completed(
                    tool_call=second,
                    tool_result=ToolExecutionResult(call=second, output='{"matches":[]}'),
                )
            ),
            TUIEvent.from_agent_event(
                builder.run_completed(
                    step=AgentStep(llm_response=LLMResponse(text="done")),
                    final_answer="done",
                )
            ),
        ]
    )
    stream = StreamWidget()
    writes: list[Text] = []
    _capture_stream_writes(stream, writes)
    stream.set_state(state)

    stream.on_click(
        Click(
            stream,
            0,
            0,
            0,
            0,
            1,
            False,
            False,
            False,
            style=Style(meta={"tool_activity_id": "call-1|call-2"}),
        )
    )

    assert [text.plain for text in writes] == [
        "  ─ [-] work history · 2 tool calls",
        "    * think first",
        '    ● read_text_file("a.py")  result ready',
        "    * think second",
        '    ● search_text("needle")  search finished',
        "  done",
    ]


def test_completed_working_history_renders_between_question_and_answer() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    events = [
        TUIEvent.user_message("what changed?"),
        TUIEvent.from_agent_event(
            builder.llm_reasoning(
                segment=LLMSegment(
                    type="reasoning",
                    content="inspect first",
                    meta=LLMReasoningSegmentMeta(
                        item_id="reasoning-1",
                        kind="content",
                        index=0,
                        is_delta=True,
                    ),
                )
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=call,
                tool_result=ToolExecutionResult(call=call, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="answer")),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="answer")),
                final_answer="answer",
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 4
    assert isinstance(renderables[0], _PromptBar)
    _assert_prompt_bar_contains(renderables[0], "▌ what changed?")
    assert isinstance(renderables[1], Text)
    assert renderables[1].plain == ""
    assert isinstance(renderables[2], Text)
    assert renderables[2].plain == "  ─ [+] work history · 1 tool call"
    assert isinstance(renderables[3], Text)
    assert renderables[3].plain == "  answer"


def test_completed_working_history_stays_before_answer_when_tool_replays_late() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    events = [
        TUIEvent.user_message("what changed?"),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="answer")),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=call,
                tool_result=ToolExecutionResult(call=call, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="answer")),
                final_answer="answer",
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 4
    assert isinstance(renderables[0], _PromptBar)
    assert isinstance(renderables[1], Text)
    assert renderables[1].plain == ""
    assert isinstance(renderables[2], Text)
    assert renderables[2].plain == "  ─ [+] work history · 1 tool call"
    assert isinstance(renderables[3], Text)
    assert renderables[3].plain == "  answer"


def test_completed_working_history_stays_before_answer_when_next_question_flushes() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    events = [
        TUIEvent.user_message("first?"),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="answer")),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=call,
                tool_result=ToolExecutionResult(call=call, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="answer")),
                final_answer="answer",
            )
        ),
        TUIEvent.user_message("second?"),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 6
    assert isinstance(renderables[0], _PromptBar)
    assert isinstance(renderables[1], Text)
    assert renderables[1].plain == ""
    assert isinstance(renderables[2], Text)
    assert renderables[2].plain == "  ─ [+] work history · 1 tool call"
    assert isinstance(renderables[3], Text)
    assert renderables[3].plain == "  answer"
    assert isinstance(renderables[4], Text)
    assert renderables[4].plain == ""
    assert isinstance(renderables[5], _PromptBar)


def test_completed_retry_does_not_split_or_render_working_history() -> None:
    first_builder = AgentEventBuilder(step_index=0, step_id="step-1")
    second_builder = AgentEventBuilder(step_index=1, step_id="step-2")
    first = LLMToolCall(
        name="search_text",
        arguments='{"query":"version"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"aceai/__init__.py"}',
        call_id="call-2",
    )
    events = [
        TUIEvent.user_message("version?"),
        TUIEvent.from_agent_event(
            first_builder.llm_reasoning(
                segment=LLMSegment(
                    type="reasoning",
                    content="check version",
                    meta=LLMReasoningSegmentMeta(
                        item_id="reasoning-1",
                        kind="content",
                        index=0,
                        is_delta=True,
                    ),
                )
            )
        ),
        TUIEvent.from_agent_event(first_builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            first_builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output="{}"),
            )
        ),
        TUIEvent.from_agent_event(second_builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            second_builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output="{}"),
            )
        ),
        TUIEvent.from_agent_event(
            second_builder.llm_retrying(
                retry_count=1,
                retry_max=5,
                retry_delay_seconds=0.5,
                error="TimeoutError:",
            )
        ),
        TUIEvent.from_agent_event(
            second_builder.llm_reasoning(
                segment=LLMSegment(
                    type="reasoning",
                    content="use cached result",
                    meta=LLMReasoningSegmentMeta(
                        item_id="reasoning-2",
                        kind="content",
                        index=0,
                        is_delta=True,
                    ),
                )
            )
        ),
        TUIEvent.from_agent_event(second_builder.llm_text_delta(text_delta="answer")),
        TUIEvent.from_agent_event(
            second_builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="answer")),
                final_answer="answer",
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)
    texts = [renderable.plain for renderable in renderables if isinstance(renderable, Text)]

    assert texts == [
        "",
        "  ─ [+] work history · 2 tool calls",
        "  answer",
    ]


def test_completed_replayed_approval_cycles_collapse() -> None:
    call = {
        "type": "function_call",
        "name": "replace_text_in_file",
        "arguments": '{"path":"a.py"}',
        "call_id": "call-1",
    }
    events = event_log_to_tui_events(
        EventLog(
            [
                SessionEvent(
                    kind="tool_approval_requested",
                    payload={
                        "content": "Tool requires approval",
                        "tool_name": "replace_text_in_file",
                        "tool_call_id": "call-1",
                        "tool_arguments": '{"path":"a.py"}',
                        "tool_call": call,
                    },
                    step_id="step-1",
                    step_index=0,
                ),
                SessionEvent(
                    kind="run_suspended",
                    payload={
                        "content": "waiting for approval. Choose Approve or Reject.",
                    },
                    step_id="step-1",
                    step_index=0,
                ),
                SessionEvent(
                    kind="tool_result",
                        payload={
                            "content": "",
                            "tool_name": "replace_text_in_file",
                            "tool_call_id": "call-1",
                            "tool_arguments": '{"path":"a.py"}',
                            "output": '{"ok":true}',
                            "truncated_output": '{"ok":true}',
                            "status": "completed",
                        },
                    step_id="step-1",
                    step_index=0,
                ),
                SessionEvent(
                    kind="run_completed",
                    payload={"content": "done"},
                    step_id="step-1",
                    step_index=0,
                ),
            ]
        )
    )

    writes = _stream_set_state_writes(reduce_events(events))

    assert [text.plain for text in writes] == [
        "  ─ [+] work history · 1 tool call",
        "  done",
    ]


def test_completed_tool_activity_ignores_invisible_control_events() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    first_request = ToolApprovalRequest(
        call=first,
        tool_name="replace_text_in_file",
        reason="requires approval",
        policy="filesystem_write",
    )
    second_request = ToolApprovalRequest(
        call=second,
        tool_name="replace_text_in_file",
        reason="requires approval",
        policy="filesystem_write",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(builder.tool_approval_requested(request=first_request)),
        TUIEvent.from_agent_event(builder.run_suspended(request=first_request)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"ok":true}'),
            )
        ),
        TUIEvent.from_agent_event(
            builder.step_completed(step=AgentStep(llm_response=LLMResponse(text="")))
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(builder.tool_approval_requested(request=second_request)),
        TUIEvent.from_agent_event(builder.run_suspended(request=second_request)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output='{"ok":true}'),
            )
        ),
        TUIEvent.from_agent_event(
            builder.run_completed(
                step=AgentStep(llm_response=LLMResponse(text="done")),
                final_answer="done",
            )
        ),
    ]

    writes = _stream_set_state_writes(reduce_events(events))

    assert [text.plain for text in writes] == [
        "  ─ [+] work history · 2 tool calls",
        "  done",
    ]


def test_assistant_markdown_renders_as_markdown() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_text_delta(text_delta="## Steps\n\n- Use `rg`\n- Run tests")
        ),
    ]

    renderables = _render_events(events)

    block = renderables[0]
    assert isinstance(block, Group)
    panel_renderable = block.renderables[1]
    assert isinstance(panel_renderable, Markdown)
    assert panel_renderable.markup == "## Steps\n\n- Use `rg`\n- Run tests"


def test_reasoning_summary_renders_before_completed_answer() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(builder.llm_started()),
        TUIEvent.from_agent_event(
            builder.llm_reasoning(
                segment=LLMSegment(type="reasoning", content="think first")
            )
        ),
        TUIEvent.from_agent_event(
            builder.llm_completed(
                step=AgentStep(llm_response=LLMResponse(text="answer"))
            )
        ),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 2
    reasoning = renderables[0]
    answer = renderables[1]
    assert isinstance(reasoning, Text)
    reasoning_renderable = reasoning
    assert isinstance(reasoning_renderable, Text)
    assert reasoning_renderable.plain == "  * think first"
    assert isinstance(answer, Text)
    assert answer.plain == "  answer"


def test_reasoning_summary_renders_inline_emphasis() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_reasoning(
                segment=LLMSegment(
                    type="reasoning",
                    content="**Search for sources**",
                )
            )
        ),
    ]

    renderables = _render_events(events)

    reasoning = renderables[0]
    assert isinstance(reasoning, Text)
    assert reasoning.plain == "  * Search for sources"
    assert len(reasoning.spans) == 2
    assert reasoning.spans[1].start == len("  * ")
    assert reasoning.spans[1].end == len("  * Search for sources")


def test_streaming_reasoning_renders_before_later_answer_delta() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_reasoning(
                segment=LLMSegment(
                    type="reasoning",
                    content="think first",
                    meta=LLMReasoningSegmentMeta(
                        item_id="reasoning",
                        kind="content",
                        index=0,
                        is_delta=True,
                    ),
                )
            )
        ),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="answer")),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 2
    reasoning = renderables[0]
    answer = renderables[1]
    assert isinstance(reasoning, Text)
    reasoning_renderable = reasoning
    assert isinstance(reasoning_renderable, Text)
    assert reasoning_renderable.plain == "  * think first"
    assert isinstance(answer, Text)
    assert answer.plain == "  answer"


def test_tool_call_step_does_not_render_assistant_scratchpad() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="lookup",
        arguments="{}",
        call_id="call-1",
    )
    result = ToolExecutionResult(call=call, output='{"ok":true}')
    events = [
        TUIEvent.from_agent_event(
            builder.llm_text_delta(text_delta="Need to inspect files.")
        ),
        TUIEvent.from_agent_event(
            builder.llm_completed(
                step=AgentStep(
                    llm_response=LLMResponse(
                        text="Need to inspect files.",
                        tool_calls=[call],
                    )
                )
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(
            builder.tool_completed(tool_call=call, tool_result=result)
        ),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    tool = renderables[0]
    assert isinstance(tool, Text)
    assert "lookup" in tool.plain
    assert "Need to inspect files" not in tool.plain


def test_main_stream_omits_lifecycle_events() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    step = AgentStep(llm_response=LLMResponse(text="done"))
    events = [
        TUIEvent.from_agent_event(builder.llm_started()),
        TUIEvent.from_agent_event(builder.step_completed(step=step)),
        TUIEvent.from_agent_event(builder.run_completed(step=step, final_answer="done")),
    ]

    renderables = _render_events(events)

    assert renderables == []


def test_tool_call_deltas_render_as_one_collapsed_tool_message() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"binary_search.py","content":"print(1)\\n"}',
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output='{"path":"binary_search.py","bytes_written":9}',
    )
    events = [
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='{"path":"binary',
                )
            )
        ),
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='_search.py","content":"print(1)\\n"}',
                )
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert (
        text.plain
        == '  ● write_text_file(path: "binary_search.py", content: "print(1)\\n")  file written'
    )


def test_unknown_in_progress_tool_call_deltas_do_not_render() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='{"q":',
                )
            )
        ),
        TUIEvent.from_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='"aceai"}',
                )
            )
        ),
    ]

    renderables = _render_events(events)

    assert renderables == []


def test_directory_tool_result_summarizes_entry_count_without_details() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="list_directory",
        arguments='{"path":"."}',
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"path":".","entries":['
            '{"name":"aceai","path":"aceai","kind":"directory"},'
            '{"name":"tests","path":"tests","kind":"directory"}'
            "]}"
        ),
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == '  ● list_directory(".")  2 entries'


def test_short_shell_command_arguments_render_inline() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"bash ls"}',
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output='{"exit_code":0,"stdout":"","stderr":""}',
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == '  ● run_shell_command("bash ls")  succeeded'


def test_shell_command_result_summarizes_success_without_stdout_details() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"bash ls"}',
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output='{"exit_code":0,"stdout":"a.py\\nb.py\\n","stderr":""}',
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == '  ● run_shell_command("bash ls")  succeeded'


def test_search_result_with_exit_code_summarizes_as_search() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="search_text",
        arguments='{"query":"delegate_to_subagent","path":"/Users/raceychan/mylab/aceai"}',
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output=(
            '{"query":"delegate_to_subagent",'
            '"path":"/Users/raceychan/mylab/aceai",'
            '"exit_code":0,'
            '"matches":"/Users/raceychan/mylab/aceai/aceai/agent/ace_agent.py",'
            '"errors":""}'
        ),
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == '  ● search_text(query: "delegate_to_subagent", path: "/Users/raceychan/mylab/aceai")  search finished'


def test_run_failed_event_renders_error() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    step = AgentStep(
        step_id="step-1",
        llm_response=LLMResponse(text=""),
    )
    events = [
        TUIEvent.from_agent_event(
            builder.run_failed(
                step=step,
                error="LLM request failed after retries. Please try again later.",
            )
        )
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert (
        text.plain
        == "  ● failed  LLM request failed after retries. Please try again later."
    )


def test_long_and_complex_tool_arguments_render_collapsed_preview() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="replace_text_in_file",
        arguments=(
            '{"path":"a.py","old_text":"line 1\\nline 2\\nline 3",'
            '"metadata":{"source":"test"},"new_text":"'
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
            '"}'
        ),
        call_id="call-1",
    )
    result = ToolExecutionResult(
        call=call,
        output='{"path":"a.py","replacements":1}',
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert (
        text.plain
        == '  ● replace_text_in_file(path: "a.py", old_text: "line 1\\nline 2\\nline 3", metadata: ..., new_text: "abcdefghijklmnopq...)  result ready'
    )


def test_consecutive_completed_tools_compact_by_tool_name() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output='{"content":"b"}'),
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == "  ─ [+] work history · 2 tool calls"


def test_consecutive_completed_tool_activity_compacts_across_tool_names() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="search_text",
        arguments='{"query":"needle"}',
        call_id="call-2",
    )
    third = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"a.py"}',
        call_id="call-3",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output='{"matches":[]}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=third)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=third,
                tool_result=ToolExecutionResult(call=third, output='{"ok":true}'),
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == "  ─ [+] work history · 3 tool calls"


def test_failed_tools_do_not_compact_into_completed_group() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    failed_result = ToolExecutionResult(
        call=second,
        output="failed",
        error="missing",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_failed(
                tool_call=second,
                tool_result=failed_result,
                error="missing",
            )
        ),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 2
    completed = renderables[0]
    failed = renderables[1]
    assert isinstance(completed, Text)
    assert isinstance(failed, Text)
    assert completed.plain == '  ● read_text_file("a.py")  result ready'
    assert failed.plain == '  ● read_text_file("b.py")  failed'


def test_repeated_approval_cycles_compact_by_tool_name() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_approval_requested(
                request=ToolApprovalRequest(
                    call=first,
                    tool_name="replace_text_in_file",
                    reason="requires approval",
                    policy="filesystem_write",
                )
            )
        ),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"ok":true}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_approval_requested(
                request=ToolApprovalRequest(
                    call=second,
                    tool_name="replace_text_in_file",
                    reason="requires approval",
                    policy="filesystem_write",
                )
            )
        ),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output='{"ok":true}'),
            )
        ),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == "  ─ [+] work history · 2 tool calls"


def test_pending_approval_group_stays_visible() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    request = ToolApprovalRequest(
        call=second,
        tool_name="replace_text_in_file",
        reason="requires approval",
        policy="filesystem_write",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"ok":true}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_approval_requested(
                request=request
            )
        ),
        TUIEvent.from_agent_event(builder.run_suspended(request=request)),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == (
        "  ─ [+] work history · 2 tool calls"
    )


def test_run_suspended_does_not_render_separate_approval_line() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    call = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    request = ToolApprovalRequest(
        call=call,
        tool_name="replace_text_in_file",
        reason="requires approval",
        policy="filesystem_write",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=call)),
        TUIEvent.from_agent_event(builder.tool_approval_requested(request=request)),
        TUIEvent.from_agent_event(builder.run_suspended(request=request)),
    ]

    renderables = _render_events(events, collapse_tool_activity=True)

    assert len(renderables) == 1
    text = renderables[0]
    assert isinstance(text, Text)
    assert text.plain == (
        "  ─ [+] work history · 1 tool call"
    )


def test_running_tool_activity_does_not_collapse() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    first = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"a.py"}',
        call_id="call-1",
    )
    second = LLMToolCall(
        name="read_text_file",
        arguments='{"path":"b.py"}',
        call_id="call-2",
    )
    events = [
        TUIEvent.from_agent_event(builder.tool_started(tool_call=first)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=first,
                tool_result=ToolExecutionResult(call=first, output='{"content":"a"}'),
            )
        ),
        TUIEvent.from_agent_event(builder.tool_started(tool_call=second)),
        TUIEvent.from_agent_event(
            builder.tool_completed(
                tool_call=second,
                tool_result=ToolExecutionResult(call=second, output='{"content":"b"}'),
            )
        ),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 2
    first_rendered = renderables[0]
    second_rendered = renderables[1]
    assert isinstance(first_rendered, Text)
    assert isinstance(second_rendered, Text)
    assert first_rendered.plain == '  ● read_text_file("a.py")  result ready'
    assert second_rendered.plain == '  ● read_text_file("b.py")  result ready'
