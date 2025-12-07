from typing import AsyncIterator
from uuid import uuid4

from .errors import AceAIConfigurationError, AceAIRuntimeError
from .events import AgentEvent, AgentEventBuilder, RunCompletedEvent
from .executor import ToolExecutor
from .llm import LLMMessage, LLMResponse, LLMService
from .llm.models import LLMToolCall, LLMToolCallMessage, LLMToolUseMessage
from .models import AgentStep, ToolExecutionResult


class ToolExecutionFailure(AceAIRuntimeError):
    """Raised when a tool invocation fails during a reasoning step."""

    def __init__(self, *, tool_call: LLMToolCall, error: Exception):
        message = str(error)
        super().__init__(message)
        self.tool_call = tool_call
        self.original_error = error


class AgentBase:
    """Base class for agents using an LLM provider."""

    def __init__(
        self,
        *,
        prompt: str,
        default_model: str,
        llm_service: LLMService,
        executor: ToolExecutor,
        max_steps: int = 5,
    ):
        if max_steps < 1:
            raise AceAIConfigurationError("max_steps must be at least 1")
        self.prompt = prompt
        self.default_model = default_model
        self.llm_service = llm_service
        self.executor = executor
        self.max_steps = max_steps

    async def _run_step(
        self,
        *,
        messages: list[LLMMessage],
        steps: list[AgentStep],
        selected_model: str,
    ) -> AsyncIterator[AgentEvent]:
        step_index = len(steps)
        step_id = str(uuid4())
        event_builder = AgentEventBuilder(
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

        if not response.tool_calls:
            yield event_builder.step_completed(step=step)

            if final_answer := response.text:
                yield event_builder.run_completed(
                    step=step,
                    final_answer=final_answer,
                )
        else:
            assistant_msg = LLMToolCallMessage(
                content=response.text,
                tool_calls=response.tool_calls,
            )
            messages.append(assistant_msg)

            for call in response.tool_calls:
                yield event_builder.tool_started(tool_call=call)
                try:
                    tool_output = await self.executor.execute_tool(call)
                except Exception as exc:
                    raise ToolExecutionFailure(tool_call=call, error=exc) from exc

                tool_result = ToolExecutionResult(call=call, output=tool_output)
                step.tool_results.append(tool_result)
                yield event_builder.tool_completed(
                    tool_call=call,
                    tool_result=tool_result,
                )

                if call.name == "final_answer":
                    yield event_builder.run_completed(
                        step=step,
                        final_answer=tool_result.output,
                    )
                    return

                messages.append(
                    LLMToolUseMessage(
                        name=call.name,
                        call_id=call.call_id,
                        content=tool_result.output,
                    )
                )

            yield event_builder.step_completed(step=step)

    async def run(
        self,
        question: str,
        *,
        model: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvent entries as the agent reasons."""
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.prompt),
            LLMMessage(role="user", content=question),
        ]
        selected_model = model or self.default_model
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            try:
                async for event in self._run_step(
                    messages=messages,
                    steps=steps,
                    selected_model=selected_model,
                ):
                    yield event
                    if isinstance(event, RunCompletedEvent):
                        return
            except Exception as exc:
                if not steps:
                    raise
                error_msg = str(exc)
                last_step = steps[-1]
                last_index = len(steps) - 1
                event_builder = AgentEventBuilder(
                    step_index=last_index,
                    step_id=last_step.step_id,
                )
                if isinstance(exc, ToolExecutionFailure):
                    tool_call = exc.tool_call
                    error_msg = str(exc.original_error)
                    failed_result = ToolExecutionResult(
                        call=tool_call,
                        error=error_msg,
                    )
                    last_step.tool_results.append(failed_result)
                    yield event_builder.tool_failed(
                        tool_call=tool_call,
                        tool_result=failed_result,
                        error=error_msg,
                    )
                yield event_builder.step_failed(step=last_step, error=error_msg)
                yield event_builder.run_failed(step=last_step, error=error_msg)
                raise
        else:
            error_msg = "Agent exceeded maximum reasoning turns without answering"
            last_step = steps[-1]
            last_index = len(steps) - 1
            event_builder = AgentEventBuilder(
                step_index=last_index,
                step_id=last_step.step_id,
            )
            yield event_builder.step_failed(step=last_step, error=error_msg)
            yield event_builder.run_failed(step=last_step, error=error_msg)
            raise AceAIRuntimeError(error_msg)
