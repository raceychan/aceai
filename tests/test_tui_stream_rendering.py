from rich.panel import Panel
from rich.text import Text

from aceai.agent.events import AgentEventBuilder
from aceai.llm.models import LLMResponse
from aceai.models import AgentStep
from aceai.tui.events import adapt_agent_event, user_message_event
from aceai.tui.widgets.stream import _render_events


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
    assert isinstance(panel.renderable, Text)
    assert panel.renderable.plain == "Hi!"


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
    assert question.renderable.plain == "How do I search?"
    assert answer.renderable.plain == "Use rg."


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
