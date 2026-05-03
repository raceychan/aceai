from rich.panel import Panel
from rich.text import Text

from aceai.core.events import AgentEventBuilder
from aceai.llm.models import LLMResponse, LLMToolCall, LLMToolCallDelta
from aceai.core.models import AgentStep
from aceai.core.models import ToolExecutionResult
from aceai.agent.tui.events import adapt_agent_event, user_message_event
from aceai.agent.tui.widgets.stream import _render_events


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
    assert isinstance(panel_renderable, Text)
    assert panel_renderable.plain == "Hi!"


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
    assert isinstance(question, Panel)
    assert isinstance(answer, Panel)
    assert question.title == "you"
    assert answer.title == "assistant"
    question_renderable = question.renderable
    answer_renderable = answer.renderable
    assert isinstance(question_renderable, Text)
    assert isinstance(answer_renderable, Text)
    assert question_renderable.plain == "How do I search?"
    assert answer_renderable.plain == "Use rg."


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
