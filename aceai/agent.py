from typing import AsyncIterator, Unpack
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from .errors import AceAIConfigurationError, AceAIRuntimeError
from .events import (
    AgentEvent,
    AgentEventBuilder,
    LLMCompletedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    StepFailedEvent,
    ToolFailedEvent,
)
from .executor import ToolExecutor
from .helpers.delta_buffer import LLMDeltaChunker, ReasoningLogBuffer
from .interface import is_set
from .llm import ILLMService, LLMMessage, LLMResponse
from .llm.models import (
    LLMRequestMeta,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
)
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
        sys_prompt: str = "",
        *,
        default_model: str,
        llm_service: ILLMService,
        executor: ToolExecutor | None = None,
        max_steps: int = 5,
        delta_chunk_size: int = 0,
        reasoning_log_max_chars: int | None = None,
        tracer: trace.Tracer | None = None,
    ):
        if max_steps < 1:
            raise AceAIConfigurationError("max_steps must be at least 1")
        if delta_chunk_size < 0:
            raise AceAIConfigurationError("delta_chunk_size must be non-negative")
        if reasoning_log_max_chars is not None and reasoning_log_max_chars < 0:
            raise AceAIConfigurationError(
                "reasoning_log_max_chars must be non-negative or None"
            )
        self.sys_prompt = sys_prompt
        self.default_model = default_model
        self.llm_service = llm_service
        self._executor = executor
        self.max_steps = max_steps
        self.delta_chunk_size = delta_chunk_size
        self.reasoning_log_max_chars = reasoning_log_max_chars
        self._tracer = tracer or trace.get_tracer("aceai.agent")

    async def _run_step(
        self,
        *,
        messages: list[LLMMessage],
        event_builder: AgentEventBuilder,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncIterator[AgentEvent]:
        executor = self._executor
        with self._tracer.start_as_current_span(
            "agent.step",
            kind=SpanKind.INTERNAL,
            record_exception=True,
            set_status_on_exception=True,
            attributes={
                "agent.step_id": event_builder.step_id,
                "agent.max_steps": self.max_steps,
            },
        ):
            yield event_builder.llm_started()

            log_buffer = ReasoningLogBuffer(max_chars=self.reasoning_log_max_chars)
            chunker = LLMDeltaChunker(self.delta_chunk_size)

            response: LLMResponse | None = None
            if executor:
                stream = self.llm_service.stream(
                    messages=messages,
                    tools=executor.tool_specs,
                    metadata=request_meta,
                )
            else:
                stream = self.llm_service.stream(
                    messages=messages, metadata=request_meta
                )

            try:
                async for stream_event in stream:
                    if stream_event.event_type == "response.output_text.delta":
                        if is_set(stream_event.text_delta):
                            for chunk in chunker.push(stream_event.text_delta):
                                log_buffer.append(chunk)
                                yield event_builder.llm_text_delta(text_delta=chunk)
                        continue

                    if (
                        stream_event.event_type
                        == "response.function_call_arguments.delta"
                    ):
                        # TODO: emit tool call argument deltas once planner wiring lands.
                        continue

                    if stream_event.event_type == "response.error":
                        if is_set(stream_event.error) and stream_event.error:
                            raise AceAIRuntimeError(stream_event.error)
                        raise AceAIRuntimeError("LLM streaming error")

                    if stream_event.event_type == "response.completed":
                        if is_set(stream_event.response):
                            response = stream_event.response
                            for chunk in chunker.flush():
                                log_buffer.append(chunk)
                                yield event_builder.llm_text_delta(text_delta=chunk)
                            break
                        raise AceAIRuntimeError(
                            "LLM stream completed without a response payload"
                        )
                if response is None:
                    for chunk in chunker.flush():
                        log_buffer.append(chunk)
                        if chunk:
                            yield event_builder.llm_text_delta(text_delta=chunk)
            except Exception:
                for chunk in chunker.flush():
                    log_buffer.append(chunk)
                    if chunk:
                        yield event_builder.llm_text_delta(text_delta=chunk)
                raise
            finally:
                reasoning_log = log_buffer.snapshot()
                if response is not None and not reasoning_log:
                    reasoning_chunks = [
                        segment.content
                        for segment in response.segments
                        if segment.type == "reasoning" and segment.content
                    ]
                    if reasoning_chunks:
                        reasoning_log = "\n\n".join(reasoning_chunks)
                final_response = response or LLMResponse(text=reasoning_log)
                current_step = AgentStep(
                    step_id=event_builder.step_id,
                    llm_response=final_response,
                    reasoning_log=reasoning_log,
                    reasoning_log_truncated=log_buffer.truncated,
                )

        if response is None:
            raise AceAIRuntimeError("LLM stream ended without completion")

        yield event_builder.llm_completed(step=current_step)

        if not current_step.llm_response.tool_calls:
            yield event_builder.step_completed(step=current_step)

            if final_answer := current_step.llm_response.text:
                yield event_builder.run_completed(
                    step=current_step,
                    final_answer=final_answer,
                )
        else:
            assistant_msg = LLMToolCallMessage.build(
                content=current_step.llm_response.text,
                tool_calls=current_step.llm_response.tool_calls,
            )
            messages.append(assistant_msg)

            for call in current_step.llm_response.tool_calls:
                yield event_builder.tool_started(tool_call=call)
                if executor is None:
                    raise AceAIConfigurationError(
                        "executor must be provided when tool calls are enabled"
                    )
                try:
                    tool_output = await executor.execute_tool(call)
                except Exception as exc:
                    raise ToolExecutionFailure(tool_call=call, error=exc) from exc

                tool_result = ToolExecutionResult(call=call, output=tool_output)
                current_step.tool_results.append(tool_result)
                yield event_builder.tool_completed(
                    tool_call=call,
                    tool_result=tool_result,
                )

                if call.name == "final_answer":
                    yield event_builder.run_completed(
                        step=current_step,
                        final_answer=tool_result.output,
                    )
                    return

                messages.append(
                    LLMToolUseMessage.build(
                        name=call.name,
                        call_id=call.call_id,
                        content=tool_result.output,
                    )
                )

            yield event_builder.step_completed(step=current_step)

    async def ask(self, question: str, **request_meta: Unpack[LLMRequestMeta]) -> str:
        """Run the agent to completion and return the final answer in plain text."""
        async for event in self.run(
            question,
            **request_meta,
        ):
            if isinstance(event, RunCompletedEvent):
                return event.final_answer

        raise AceAIRuntimeError("Agent run did not complete successfully")

    async def run(
        self,
        question: str,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvent entries as the agent reasons."""
        messages: list[LLMMessage] = [
            LLMMessage.build(role="system", content=self.sys_prompt),
            LLMMessage.build(role="user", content=question),
        ]
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            step_id = str(uuid4())
            event_builder = AgentEventBuilder(
                step_index=len(steps),
                step_id=step_id,
            )
            try:
                async for event in self._run_step(
                    messages=messages,
                    event_builder=event_builder,
                    **request_meta,
                ):
                    yield event
                    if isinstance(event, LLMCompletedEvent):
                        steps.append(event.step)
                    if isinstance(event, RunCompletedEvent):
                        return
            except Exception as exc:
                if not steps:
                    raise
                error_msg = str(exc)
                last_step = steps[-1]
                last_index = len(steps) - 1
                if isinstance(exc, ToolExecutionFailure):
                    tool_call = exc.tool_call
                    error_msg = str(exc.original_error)
                    failed_result = ToolExecutionResult(
                        call=tool_call,
                        error=error_msg,
                    )
                    last_step.tool_results.append(failed_result)
                    yield ToolFailedEvent(
                        step_index=last_index,
                        step_id=last_step.step_id,
                        tool_call=tool_call,
                        tool_name=tool_call.name,
                        tool_result=failed_result,
                        error=error_msg,
                    )
                yield StepFailedEvent(
                    step_index=last_index,
                    step_id=last_step.step_id,
                    step=last_step,
                    error=error_msg,
                )
                yield RunFailedEvent(
                    step_index=last_index,
                    step_id=last_step.step_id,
                    step=last_step,
                    error=error_msg,
                )
                raise

        error_msg = "Agent exceeded maximum reasoning turns without answering"
        last_step = steps[-1]
        last_index = len(steps) - 1
        yield StepFailedEvent(
            step_index=last_index,
            step_id=last_step.step_id,
            step=last_step,
            error=error_msg,
        )
        yield RunFailedEvent(
            step_index=last_index,
            step_id=last_step.step_id,
            step=last_step,
            error=error_msg,
        )
        raise AceAIRuntimeError(error_msg)
