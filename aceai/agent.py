from typing import Any, AsyncIterator
from uuid import uuid4

from .errors import AceAIRuntimeError
from .executor import ToolExecutor
from .llm import LLMMessage, LLMResponse, LLMService
from .llm.models import LLMToolCall, LLMToolCallMessage, LLMToolUseMessage
from .models import AgentStep, AgentStepEvent, ToolExecutionResult


class AgentStepEventBuilder:
    """Utility helper that stamps shared step metadata onto events."""

    __slots__ = ("step_index", "step_id")

    def __init__(self, *, step_index: int, step_id: str):
        self.step_index = step_index
        self.step_id = step_id

    def _event(self, *, event_type: str, **payload: Any) -> AgentStepEvent:
        return AgentStepEvent(
            event_type=event_type,
            step_index=self.step_index,
            step_id=self.step_id,
            **payload,
        )

    def llm_started(self) -> AgentStepEvent:
        return self._event(event_type="agent.llm.started")

    def llm_completed(self, *, step: AgentStep) -> AgentStepEvent:
        return self._event(event_type="agent.llm.completed", step=step)

    def tool_started(self, *, tool_call: LLMToolCall) -> AgentStepEvent:
        return self._event(event_type="agent.tool.started", tool_call=tool_call)

    def tool_completed(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolExecutionResult,
    ) -> AgentStepEvent:
        return self._event(
            event_type="agent.tool.completed",
            tool_call=tool_call,
            tool_result=tool_result,
        )

    def tool_failed(
        self,
        *,
        tool_call: LLMToolCall,
        tool_result: ToolExecutionResult,
        error: str,
    ) -> AgentStepEvent:
        return self._event(
            event_type="agent.tool.failed",
            tool_call=tool_call,
            tool_result=tool_result,
            error=error,
        )

    def step_completed(self, *, step: AgentStep) -> AgentStepEvent:
        return self._event(event_type="agent.step.completed", step=step)

    def step_failed(self, *, step: AgentStep, error: str) -> AgentStepEvent:
        return self._event(
            event_type="agent.step.failed",
            step=step,
            error=error,
        )

    def run_completed(
        self,
        *,
        step: AgentStep,
        annotations: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgentStepEvent:
        payload: dict[str, Any] = {"step": step}
        if annotations is not None:
            payload["annotations"] = annotations
        if error is not None:
            payload["error"] = error
        return self._event(event_type="agent.run.completed", **payload)


class AgentBase:
    """Base class for agents using an LLM provider."""

    def __init__(
        self,
        *,
        prompt: str,
        default_model: str,
        llm_service: LLMService,
        executor: ToolExecutor,
        max_turns: int = 5,
    ):
        self.prompt = prompt
        self.default_model = default_model
        self.llm_service = llm_service
        self.executor = executor
        self.max_turns = max_turns

    async def run(
        self,
        question: str,
        *,
        model: str | None = None,
    ) -> AsyncIterator[AgentStepEvent]:
        """Yield AgentStepEvent entries as the agent reasons."""

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.prompt),
            LLMMessage(role="user", content=question),
        ]
        selected_model = model or self.default_model
        steps: list[AgentStep] = []

        for _ in range(self.max_turns):
            step_index = len(steps)
            step_id = str(uuid4())
            event_builder = AgentStepEventBuilder(
                step_index=step_index,
                step_id=step_id,
            )
            yield event_builder.llm_started()

            response: LLMResponse = await self.llm_service.complete(
                messages=messages,
                tools=self.executor.tool_schemas,
                metadata={"model": selected_model},
            )
            step = AgentStep(step_id=step_id, llm_response=response)
            steps.append(step)

            yield event_builder.llm_completed(step=step)

            if response.tool_calls:
                assistant_msg = LLMToolCallMessage(
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)
                final_answer_output: str | None = None

                for call in response.tool_calls:
                    yield event_builder.tool_started(tool_call=call)
                    try:
                        tool_output = await self.executor.execute_tool(call)
                    except Exception as exc:
                        error_msg = str(exc)
                        failed_result = ToolExecutionResult(call=call, error=error_msg)
                        step.tool_results.append(failed_result)
                        yield event_builder.tool_failed(
                            tool_call=call,
                            tool_result=failed_result,
                            error=error_msg,
                        )
                        yield event_builder.step_failed(step=step, error=error_msg)
                        yield event_builder.run_completed(step=step, error=error_msg)
                        raise

                    tool_result = ToolExecutionResult(call=call, output=tool_output)
                    step.tool_results.append(tool_result)
                    yield event_builder.tool_completed(
                        tool_call=call,
                        tool_result=tool_result,
                    )

                    if call.name == "final_answer":
                        final_answer_output = tool_result.output
                        break

                    messages.append(
                        LLMToolUseMessage(
                            name=call.name,
                            call_id=call.call_id,
                            content=tool_result.output,
                        )
                    )

                yield event_builder.step_completed(step=step)

                if final_answer_output is not None:
                    yield event_builder.run_completed(
                        step=step,
                        annotations={"final_output": final_answer_output},
                    )
                    return

                continue

            final_answer = response.text

            yield event_builder.step_completed(step=step)

            if final_answer:
                yield event_builder.run_completed(
                    step=step,
                    annotations={"final_output": final_answer},
                )
                return

            messages.append(LLMMessage(role="assistant", content=""))

        error_msg = "Agent exceeded maximum reasoning turns without answering"
        if steps:
            last_step = steps[-1]
            last_index = len(steps) - 1
            event_builder = AgentStepEventBuilder(
                step_index=last_index,
                step_id=last_step.step_id,
            )
            yield event_builder.step_failed(step=last_step, error=error_msg)
            yield event_builder.run_completed(step=last_step, error=error_msg)
        else:
            yield AgentStepEvent(
                event_type="agent.run.completed",
                step_index=0,
                step_id=str(uuid4()),
                error=error_msg,
            )
        raise AceAIRuntimeError(error_msg)
