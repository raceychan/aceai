from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from aceai.core.events import AgentEventBuilder
from aceai.llm.models import LLMResponse, LLMToolCall, LLMToolCallDelta
from aceai.core.models import AgentStep
from aceai.core.models import ToolExecutionResult
from aceai.agent.tui.events import adapt_agent_event, user_message_event
from aceai.agent.tui.state import reduce_events
from aceai.agent.tui.widgets.stream import (
    StreamWidget,
    _chat_panel_width,
    _render_events,
)


def test_consecutive_assistant_deltas_render_as_one_block() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        adapt_agent_event(builder.llm_text_delta(text_delta="Hi")),
        adapt_agent_event(builder.llm_text_delta(text_delta="!")),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    panel = renderables[0]
    assert isinstance(panel, Panel)
    panel_renderable = panel.renderable
    assert isinstance(panel_renderable, Markdown)
    assert panel_renderable.markup == "Hi!"


def test_main_stream_renders_question_before_answer() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        user_message_event("How do I search?"),
        adapt_agent_event(builder.llm_started()),
        adapt_agent_event(builder.llm_text_delta(text_delta="Use rg.")),
        adapt_agent_event(
            builder.step_completed(step=AgentStep(llm_response=LLMResponse(text="Use rg.")))
        ),
        adapt_agent_event(
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
    assert isinstance(answer, Panel)
    question_panel = question.columns[1]._cells[0]
    assert isinstance(question_panel, Panel)
    assert question_panel.title is None
    assert answer.title is None
    assert not answer.expand
    question_renderable = question_panel.renderable
    answer_renderable = answer.renderable
    assert isinstance(question_renderable, Text)
    assert isinstance(answer_renderable, Markdown)
    assert question_renderable.plain == "How do I search?"
    assert answer_renderable.markup == "Use rg."


def test_user_messages_render_right_aligned() -> None:
    renderables = _render_events([user_message_event("Where am I?")])

    message = renderables[0]
    assert isinstance(message, Table)
    assert message.expand
    assert len(message.columns) == 2
    assert message.columns[0].ratio == 1
    panel = message.columns[1]._cells[0]
    assert isinstance(panel, Panel)
    assert not panel.expand
    assert panel.title is None


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

    stream.set_state(reduce_events([user_message_event("Where am I?")]))

    assert len(writes) == 1
    content, expand = writes[0]
    assert isinstance(content, Table)
    assert expand


def test_stream_writes_assistant_messages_with_capped_width() -> None:
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

    event = adapt_agent_event(
        AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
            text_delta="x" * 300
        )
    )
    stream.set_state(reduce_events([event]))

    assert len(writes) == 1
    content, width = writes[0]
    assert isinstance(content, Panel)
    assert width == 100


def test_short_assistant_messages_keep_natural_width() -> None:
    panel = _render_events(
        [
            adapt_agent_event(
                AgentEventBuilder(step_index=0, step_id="step-1").llm_text_delta(
                    text_delta="Use rg."
                )
            )
        ]
    )[0]

    assert isinstance(panel, Panel)
    assert _chat_panel_width(panel, 120) == 11


def test_assistant_markdown_renders_as_markdown() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        adapt_agent_event(
            builder.llm_text_delta(text_delta="## Steps\n\n- Use `rg`\n- Run tests")
        ),
    ]

    renderables = _render_events(events)

    panel = renderables[0]
    assert isinstance(panel, Panel)
    assert not panel.expand
    assert panel.title is None
    panel_renderable = panel.renderable
    assert isinstance(panel_renderable, Markdown)
    assert panel_renderable.markup == "## Steps\n\n- Use `rg`\n- Run tests"


def test_main_stream_omits_lifecycle_events() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    step = AgentStep(llm_response=LLMResponse(text="done"))
    events = [
        adapt_agent_event(builder.llm_started()),
        adapt_agent_event(builder.step_completed(step=step)),
        adapt_agent_event(builder.run_completed(step=step, final_answer="done")),
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
        adapt_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='{"path":"binary',
                )
            )
        ),
        adapt_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='_search.py","content":"print(1)\\n"}',
                )
            )
        ),
        adapt_agent_event(builder.tool_started(tool_call=call)),
        adapt_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    panel = renderables[0]
    assert isinstance(panel, Panel)
    assert panel.title == "tool: write_text_file"
    panel_renderable = panel.renderable
    assert isinstance(panel_renderable, Text)
    assert panel_renderable.plain == "completed - file written"


def test_unknown_in_progress_tool_call_deltas_do_not_render() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        adapt_agent_event(
            builder.llm_tool_call_delta(
                tool_call_delta=LLMToolCallDelta(
                    id="call-1",
                    arguments_delta='{"q":',
                )
            )
        ),
        adapt_agent_event(
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
        adapt_agent_event(builder.tool_started(tool_call=call)),
        adapt_agent_event(builder.tool_completed(tool_call=call, tool_result=result)),
    ]

    renderables = _render_events(events)

    assert len(renderables) == 1
    panel = renderables[0]
    assert isinstance(panel, Panel)
    assert panel.title == "tool: list_directory"
    panel_renderable = panel.renderable
    assert isinstance(panel_renderable, Text)
    assert panel_renderable.plain == "completed - 2 entries"
