from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from aceai.core.events import AgentEventBuilder
from aceai.llm.models import (
    LLMReasoningSegmentMeta,
    LLMResponse,
    LLMSegment,
    LLMToolCall,
    LLMToolCallDelta,
)
from aceai.core.models import AgentStep
from aceai.core.models import ToolExecutionResult
from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.state import reduce_events
from aceai.agent.tui.widgets.stream import (
    StreamWidget,
    _render_events,
)


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
    assert isinstance(question, Table)
    assert question.expand
    assert isinstance(answer, Text)
    question_renderable = question.columns[0]._cells[0]
    assert isinstance(question_renderable, Text)
    assert question_renderable.plain == "▌ How do I search?"
    assert answer.plain == "  Use rg."


def test_user_messages_render_right_aligned() -> None:
    renderables = _render_events([TUIEvent.user_message("Where am I?")])

    message = renderables[0]
    assert isinstance(message, Table)
    assert message.expand
    assert len(message.columns) == 1
    assert message.columns[0].ratio == 1
    assert message.columns[0].style == "bold #eceff4 on #3b4252"
    text = message.columns[0]._cells[0]
    assert isinstance(text, Text)
    assert text.plain == "▌ Where am I?"


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
    assert isinstance(question, Table)
    question_text = question.columns[0]._cells[0]
    assert isinstance(question_text, Text)
    assert question_text.plain == "▌ next question"


def test_stream_writes_user_message_rows_expanded() -> None:
    stream = StreamWidget()
    writes: list[tuple[object, bool]] = []

    def fake_write(
        content: object,
        *,
        expand: bool = False,
        **kwargs: object,
    ) -> StreamWidget:
        writes.append((content, expand))
        return stream

    stream.write = fake_write
    stream.clear = lambda: None
    stream.call_after_refresh = lambda *args, **kwargs: None

    stream.set_state(reduce_events([TUIEvent.user_message("Where am I?")]))

    assert len(writes) == 1
    content, expand = writes[0]
    assert isinstance(content, Table)
    assert expand


def test_stream_writes_assistant_messages_as_plain_text() -> None:
    stream = StreamWidget()
    writes: list[tuple[object, int | None]] = []

    def fake_write(
        content: object,
        *,
        width: int | None = None,
        **kwargs: object,
    ) -> StreamWidget:
        writes.append((content, width))
        return stream

    stream.write = fake_write
    stream.clear = lambda: None
    stream.call_after_refresh = lambda *args, **kwargs: None

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
    assert reasoning_renderable.plain == "  * reasoning  think first"
    assert isinstance(answer, Text)
    assert answer.plain == "  answer"


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
    assert reasoning_renderable.plain == "  * reasoning  think first"
    assert isinstance(answer, Text)
    assert answer.plain == "  answer"


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
    assert text.plain == "  ● write_text_file  completed - file written"


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
    assert text.plain == "  ● list_directory  completed - 2 entries"
