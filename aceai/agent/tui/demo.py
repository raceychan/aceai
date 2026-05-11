"""Static event fixtures for the AceAI TUI."""

from aceai.core.events import AgentEvent, AgentEventBuilder
from aceai.core.models import AgentStep, ToolExecutionResult
from aceai.llm.models import LLMResponse, LLMSegment, LLMToolCall, LLMToolCallDelta
from aceai.agent.citations import AdHocCitationOrigin, TurnCitation

from .events import TUIEvent


def run_static_demo() -> None:
    from .app import AceAITUI

    class StaticDemoTUI(AceAITUI):
        def on_mount(self) -> None:
            super().on_mount()
            self.call_after_refresh(self.action_show_subagents)

    StaticDemoTUI(static_demo_events(), model="gpt-5.5", reasoning_level="auto").run()


def run_static_trajectory_demo() -> None:
    from .app import AceAITUI

    class StaticTrajectoryDemoTUI(AceAITUI):
        def on_mount(self) -> None:
            super().on_mount()
            self.call_after_refresh(self.open_trajectory_screen)

    StaticTrajectoryDemoTUI(
        static_demo_events(),
        model="gpt-5.5",
        reasoning_level="auto",
    ).run()


def static_demo_events() -> list[TUIEvent]:
    dispatch_builder = AgentEventBuilder(step_index=0, step_id="demo-step-1")
    code_audit_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Core agent behavior audit",'
            '"instructions":"inspect run-loop and executor boundaries; report '
            'regression risks with file evidence",'
            '"context_brief":"AceAI multi-agent release readiness",'
            '"allowed_tools":["search_text","read_text_file","run_shell_command"]}'
        ),
        call_id="call-subagent-core",
    )
    docs_audit_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Documentation and README audit",'
            '"instructions":"verify user-facing docs describe the TUI and skills '
            'accurately; suggest concise wording",'
            '"context_brief":"AceAI docs and README",'
            '"allowed_tools":["read_text_file","search_text"]}'
        ),
        call_id="call-subagent-docs",
    )
    integration_call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Provider integration smoke check",'
            '"instructions":"check provider adapters and streaming tool-call deltas; '
            'surface any blocker separately from docs issues",'
            '"context_brief":"LLM provider boundary",'
            '"allowed_tools":["search_text","read_text_file","run_shell_command"]}'
        ),
        call_id="call-subagent-integration",
    )
    dispatch_response = LLMResponse(
        text=(
            "I will split the release readiness review into focused child agents "
            "and merge their findings before answering."
        ),
        model="gpt-5.5",
        tool_calls=[code_audit_call, docs_audit_call, integration_call],
        segments=[
            LLMSegment(
                type="reasoning",
                content=(
                    "Use subagents for independent evidence gathering: core behavior, "
                    "documentation accuracy, and provider integration can run in parallel."
                ),
            )
        ],
    )
    dispatch_step = AgentStep(llm_response=dispatch_response, tool_results=[])
    code_audit_result = ToolExecutionResult(
        call=code_audit_call,
        output=(
            '{"type":"subagent_audit","thread_id":"child-thread-core-audit",'
            '"agent_id":"agent-core-audit","run_id":"run-core-audit",'
            '"status":"completed","final_answer":"Core loop is stable.",'
            '"summary":"Executor remains the capability surface; run-loop events '
            'preserve approval and tool-result ordering.",'
            '"important_evidence":["aceai/core/run_loop.py","aceai/core/executor.py"],'
            '"tool_results":[{"tool_name":"search_text","output":"Executor owns '
            'skill_registry, hosted tools, and local tool selection."},'
            '{"tool_name":"run_shell_command","output":"focused behavior tests passed"}],'
            '"step_count":4}'
        ),
    )
    docs_audit_result = ToolExecutionResult(
        call=docs_audit_call,
        output=(
            '{"type":"subagent_audit","thread_id":"child-thread-docs-audit",'
            '"agent_id":"agent-docs-audit","run_id":"run-docs-audit",'
            '"status":"completed","final_answer":"README needs only a compact '
            'TUI screenshot caption.",'
            '"summary":"Docs are accurate; the screenshot should emphasize '
            'skills, subagents, approvals, citations, and queued turns.",'
            '"important_evidence":["README.md","docs/tui.md"],'
            '"tool_results":[{"tool_name":"read_text_file","output":"Terminal UI '
            'section explains provider setup and per-tool permissions."}],'
            '"step_count":3}'
        ),
    )
    integration_result = ToolExecutionResult(
        call=integration_call,
        output=(
            '{"type":"subagent_audit","thread_id":"child-thread-provider-check",'
            '"agent_id":"agent-provider-check","run_id":"run-provider-check",'
            '"status":"completed","final_answer":"Provider smoke check passed.",'
            '"summary":"Streaming deltas, provider adapters, and retry progress '
            'all render through the TUI event stream.",'
            '"important_evidence":["aceai/llm/openai.py","tests/test_tui_stream_rendering.py"],'
            '"tool_results":[{"tool_name":"search_text","output":"retry event '
            'is surfaced to TUI stream"},'
            '{"tool_name":"run_shell_command","output":"provider smoke passed"}],'
            '"step_count":5}'
        ),
    )
    dispatch_step.tool_results.extend(
        [code_audit_result, docs_audit_result, integration_result]
    )

    synth_builder = AgentEventBuilder(step_index=1, step_id="demo-step-2")
    test_call = LLMToolCall(
        name="run_shell_command",
        arguments=(
            '{"command":"UV_CACHE_DIR=/tmp/uv-cache uv run pytest '
            'tests/test_tui_state.py tests/test_tui_stream_rendering.py"}'
        ),
        call_id="call-merge-tests",
    )
    synth_response = LLMResponse(
        text=(
            "All child agents completed. I am merging their evidence into a release "
            "readiness summary."
        ),
        model="gpt-5.5",
        tool_calls=[test_call],
    )
    synth_step = AgentStep(llm_response=synth_response, tool_results=[])
    test_result = ToolExecutionResult(
        call=test_call,
        output="116 passed; all child-agent checks are green",
    )
    synth_step.tool_results.append(test_result)

    agent_events: list[AgentEvent] = [
        dispatch_builder.llm_started(),
        dispatch_builder.llm_text_delta(
            text_delta="I will coordinate three child agents: core, docs, and provider checks."
        ),
        dispatch_builder.llm_tool_call_delta(
            tool_call_delta=LLMToolCallDelta(
                id=code_audit_call.call_id,
                arguments_delta=(
                    '{"task":"Core agent behavior audit","allowed_tools":'
                    '["search_text","read_text_file","run_shell_command"]}'
                ),
            )
        ),
        dispatch_builder.llm_tool_call_delta(
            tool_call_delta=LLMToolCallDelta(
                id=docs_audit_call.call_id,
                arguments_delta=(
                    '{"task":"Documentation and README audit","allowed_tools":'
                    '["read_text_file","search_text"]}'
                ),
            )
        ),
        dispatch_builder.llm_tool_call_delta(
            tool_call_delta=LLMToolCallDelta(
                id=integration_call.call_id,
                arguments_delta=(
                    '{"task":"Provider integration smoke check","allowed_tools":'
                    '["search_text","read_text_file","run_shell_command"]}'
                ),
            )
        ),
        dispatch_builder.llm_reasoning(segment=dispatch_response.segments[0]),
        dispatch_builder.llm_completed(step=dispatch_step),
        dispatch_builder.tool_started(tool_call=code_audit_call),
        dispatch_builder.tool_completed(
            tool_call=code_audit_call,
            tool_result=code_audit_result,
        ),
        dispatch_builder.tool_started(tool_call=docs_audit_call),
        dispatch_builder.tool_completed(
            tool_call=docs_audit_call,
            tool_result=docs_audit_result,
        ),
        dispatch_builder.tool_started(tool_call=integration_call),
        dispatch_builder.tool_completed(
            tool_call=integration_call,
            tool_result=integration_result,
        ),
        dispatch_builder.step_completed(step=dispatch_step),
        synth_builder.llm_started(),
        synth_builder.llm_text_delta(
            text_delta=(
                "All three child agents are complete. I will merge the evidence into "
                "one concise readiness summary."
            ),
        ),
        synth_builder.llm_completed(step=synth_step),
        synth_builder.tool_started(tool_call=test_call),
        synth_builder.tool_completed(tool_call=test_call, tool_result=test_result),
        synth_builder.step_completed(step=synth_step),
        synth_builder.run_completed(
            step=synth_step,
            final_answer=(
                "Multi-agent review complete: core behavior, docs, and provider "
                "integration are ready."
            ),
        ),
    ]
    citation = TurnCitation(
        quote=(
            "Show AceAI coordinating child agents: independent task assignment, "
            "per-agent status, evidence, tool results, and merged release summary."
        ),
        origin=AdHocCitationOrigin(kind="ad_hoc", label="multi-agent demo brief"),
    )
    return [
        TUIEvent.session_notice("AceAI delegated three release-readiness checks."),
        TUIEvent.user_message(
            "Coordinate a multi-agent release review and show me the result.",
            citations=(citation,),
        ),
        *[TUIEvent.from_agent_event(event) for event in agent_events],
    ]


def main() -> None:
    run_static_demo()


def trajectory_main() -> None:
    run_static_trajectory_demo()


if __name__ == "__main__":
    main()
