from typing import AsyncGenerator, Unpack
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, set_span_in_context

from .errors import AceAIConfigurationError, AceAIRuntimeError
from .events import (
    AgentEvent,
    AgentEventBuilder,
    LLMCompletedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    StepFailedEvent,
    ToolCompletedEvent,
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
        prompt: str = "",
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
        self._instructions: list[str] = [prompt]
        self.default_model = default_model
        self.llm_service = llm_service
        self._executor = executor
        self._max_steps = max_steps
        self.delta_chunk_size = delta_chunk_size
        self.reasoning_log_max_chars = reasoning_log_max_chars
        self._tracer = tracer or trace.get_tracer("aceai.agent")

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def add_instruction(self, instruction: str) -> None:
        if not instruction:
            raise ValueError(f"Empty Instruction")
        self._instructions.append(instruction)

    @property
    def instructions(self) -> list[str]:
        return self._instructions

    @property
    def system_message(self) -> LLMMessage:
        content = "".join(c for c in self._instructions if c)
        return LLMMessage.build(role="system", content=content)

    async def _call_llm(
        self,
        messages: list[LLMMessage],
        request_meta: LLMRequestMeta,
        event_builder: AgentEventBuilder,
        step_context: Context,
    ):
        executor = self._executor
        log_buffer = ReasoningLogBuffer(max_chars=self.reasoning_log_max_chars)
        chunker = LLMDeltaChunker(self.delta_chunk_size)
        completed = False

        if executor:
            stream = self.llm_service.stream(
                messages=messages,
                tools=executor.tool_specs,
                metadata=request_meta,
                trace_ctx=step_context,
            )
        else:
            stream = self.llm_service.stream(
                messages=messages,
                metadata=request_meta,
                trace_ctx=step_context,
            )

        try:
            async for stream_event in stream:
                if stream_event.event_type == "response.output_text.delta":
                    if is_set(stream_event.text_delta):
                        for chunk in chunker.push(stream_event.text_delta):
                            log_buffer.append(chunk)
                            yield event_builder.llm_text_delta(text_delta=chunk)
                    continue

                if stream_event.event_type == "response.function_call_arguments.delta":
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
                        reasoning_log = log_buffer.snapshot()
                        if not reasoning_log:
                            reasoning_chunks: list[str] = []
                            for segment in response.segments:
                                if segment.type == "reasoning" and segment.content:
                                    reasoning_chunks.append(segment.content)

                            if reasoning_chunks:
                                reasoning_log = "\n\n".join(reasoning_chunks)
                        final_response = response or LLMResponse(text=reasoning_log)
                        yield AgentStep(
                            step_id=event_builder.step_id,
                            llm_response=final_response,
                            reasoning_log=reasoning_log,
                            reasoning_log_truncated=log_buffer.truncated,
                        )
                        completed = True
                        break

                    raise AceAIRuntimeError(
                        "LLM stream completed without a response payload"
                    )
        except Exception:
            for chunk in chunker.flush():
                log_buffer.append(chunk)
                if chunk:
                    yield event_builder.llm_text_delta(text_delta=chunk)
            raise
        finally:
            await stream.aclose()

        if not completed:
            for chunk in chunker.flush():
                log_buffer.append(chunk)
                if chunk:
                    yield event_builder.llm_text_delta(text_delta=chunk)
            raise AceAIRuntimeError("LLM stream ended without completion")

    async def _make_toolcalls(
        self,
        current_step: AgentStep,
        event_builder: AgentEventBuilder,
        trace_ctx: Context,
    ):
        executor = self._executor

        for call in current_step.llm_response.tool_calls:
            yield event_builder.tool_started(tool_call=call)
            if executor is None:
                raise AceAIConfigurationError(
                    "executor must be provided when tool calls are enabled"
                )
            try:
                tool_output = await executor.execute_tool(call, trace_ctx=trace_ctx)
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

    async def _run_step(
        self,
        *,
        messages: list[LLMMessage],
        event_builder: AgentEventBuilder,
        trace_ctx: Context,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncGenerator[AgentEvent, None]:
        step_span = self._tracer.start_span(
            "agent.step",
            kind=SpanKind.INTERNAL,
            context=trace_ctx,
            attributes={
                "agent.step_id": event_builder.step_id,
                "agent.max_steps": self._max_steps,
            },
        )
        step_context = set_span_in_context(step_span, trace_ctx)

        try:
            yield event_builder.llm_started()
            llm_gen = self._call_llm(messages, request_meta, event_builder, step_context)
            try:
                async for event in llm_gen:
                    if not isinstance(event, AgentStep):
                        yield event
                        continue

                    current_step = event
                    yield event_builder.llm_completed(step=current_step)

                    if not current_step.llm_response.tool_calls:
                        yield event_builder.step_completed(step=current_step)
                        if final_answer := current_step.llm_response.text:
                            yield event_builder.run_completed(
                                step=current_step,
                                final_answer=final_answer,
                            )
                        return

                    assistant_msg = LLMToolCallMessage.from_content(
                        content=current_step.llm_response.text,
                        tool_calls=current_step.llm_response.tool_calls,
                    )
                    messages.append(assistant_msg)

                    tool_gen = self._make_toolcalls(
                        current_step, event_builder, step_context
                    )
                    try:
                        async for tool_event in tool_gen:
                            if isinstance(tool_event, ToolCompletedEvent):
                                call = tool_event.tool_call
                                res = tool_event.tool_result.output
                                tool_use_msg = LLMToolUseMessage.from_content(
                                    name=call.name,
                                    call_id=call.call_id,
                                    content=res,
                                )
                                messages.append(tool_use_msg)
                            yield tool_event
                            if isinstance(tool_event, RunCompletedEvent):
                                return
                    finally:
                        await tool_gen.aclose()
            finally:
                await llm_gen.aclose()

        finally:
            if step_span.is_recording():
                step_span.end()

    def _handle_failed_step(
        self,
        step: AgentStep,
        step_index: int,
        exc: Exception | None = None,
        error_msg: str | None = None,
    ):
        step_id = step.step_id
        error_msg = error_msg or str(exc)

        if isinstance(exc, ToolExecutionFailure):
            tool_call = exc.tool_call
            error_msg = str(exc.original_error)
            failed_result = ToolExecutionResult(
                call=tool_call,
                error=error_msg,
            )
            step.tool_results.append(failed_result)
            yield ToolFailedEvent(
                step_index=step_index,
                step_id=step_id,
                tool_call=tool_call,
                tool_name=tool_call.name,
                tool_result=failed_result,
                error=error_msg,
            )

        yield StepFailedEvent(
            step_index=step_index,
            step_id=step_id,
            step=step,
            error=error_msg,
        )
        yield RunFailedEvent(
            step_index=step_index,
            step_id=step_id,
            step=step,
            error=error_msg,
        )

    async def run(
        self,
        question: str,
        trace_ctx: Context | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvent entries as the agent reasons."""
        run_span = self._tracer.start_span(
            "agent.run",
            kind=SpanKind.INTERNAL,
            context=trace_ctx,
            attributes={
                "agent.max_steps": self._max_steps,
                "langfuse.trace.name": "aceai.run",
                "langfuse.trace.input": question,
            },
        )
        run_context = set_span_in_context(run_span)
        messages: list[LLMMessage] = [
            self.system_message,
            LLMMessage.build(role="user", content=question),
        ]
        steps: list[AgentStep] = []
        error_msg: str | None = None
        try:
            for _ in range(self._max_steps):
                step_id = str(uuid4())
                event_builder = AgentEventBuilder(
                    step_index=len(steps),
                    step_id=step_id,
                )

                step_gen = self._run_step(
                    messages=messages,
                    event_builder=event_builder,
                    trace_ctx=run_context,
                    **request_meta,
                )
                try:
                    async for event in step_gen:
                        if isinstance(event, LLMCompletedEvent):
                            steps.append(event.step)
                        if isinstance(event, RunCompletedEvent):
                            run_span.set_attribute(
                                "langfuse.trace.output", event.final_answer
                            )
                            yield event
                            return
                        yield event
                except Exception as exc:
                    if not steps:
                        raise
                    error_msg = str(exc)
                    last_step = steps[-1]
                    last_index = len(steps) - 1
                    for event in self._handle_failed_step(
                        last_step, last_index, exc=exc
                    ):
                        yield event
                    raise
                finally:
                    await step_gen.aclose()
            else:
                error_msg = (
                    f"Agent exceeded maximum steps: {self._max_steps} without answering"
                )
                last_step = steps[-1]
                last_index = len(steps) - 1
                for event in self._handle_failed_step(
                    last_step, last_index, error_msg=error_msg
                ):
                    yield event
                raise AceAIRuntimeError(error_msg)
        finally:
            if error_msg:
                run_span.set_attribute("langfuse.trace.output", error_msg)
            if run_span.is_recording():
                run_span.end()

    async def ask(
        self,
        question: str,
        trace_ctx: Context | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> str:
        """Run the agent to completion and return the final answer in plain text."""
        async for event in self.run(
            question,
            trace_ctx=trace_ctx,
            **request_meta,
        ):
            if isinstance(event, RunCompletedEvent):
                return event.final_answer

        raise AceAIRuntimeError("Agent run did not complete successfully")
