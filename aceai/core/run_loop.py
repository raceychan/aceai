import asyncio
from itertools import count
from typing import AsyncGenerator
from uuid import uuid4

from msgspec import Struct, field
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, set_span_in_context

from aceai.llm import ILLMService, LLMResponse
from aceai.llm.errors import (
    AceAIRuntimeError,
    LLMContextWindowExceededError,
    LLMProviderError,
)
from aceai.llm.interface import Unset, is_set
from aceai.llm.models import LLMRequestMeta, LLMToolCallDelta, LLMToolSpec
from aceai.llm.tracing import get_trace_ctx, set_trace_ctx

from .context_manager import ContextManager
from .events import (
    AgentEvent,
    AgentEventBuilder,
    LLMCompletedEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunSuspendedEvent,
    StepFailedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
)
from .executor import IExecutor
from .executor import ToolExecutionError
from .models import (
    AgentStep,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolExecutionOutput,
    ToolExecutionResult,
)
from .output_truncation import truncate_output
from .run_state import AgentRunState, AgentRunStatus
from .run_state import ToolInvocation


class AgentRunContext(Struct, kw_only=True):
    """Mutable state for one run of an Agent definition."""

    agent_id: str
    run_id: str
    question: str
    context: ContextManager
    max_steps_label: str | int
    trace_ctx: Context | None
    request_meta: LLMRequestMeta
    steps: list[AgentStep] = field(default_factory=list[AgentStep])
    run_state: AgentRunState = field(default_factory=AgentRunState)

    @property
    def status(self) -> AgentRunStatus:
        return self.run_state.status


async def execute_agent_run(
    *,
    llm_service: ILLMService,
    executor: IExecutor,
    tracer: trace.Tracer,
    max_steps: Unset[int],
    run_context: AgentRunContext,
) -> AsyncGenerator[AgentEvent, None]:
    if run_context.run_state.pending_approval is not None:
        raise AceAIRuntimeError(
            "agent run is suspended; call resume_approval() with a decision"
        )
    run_context.run_state.status = "running"
    run_span = tracer.start_span(
        "agent.run",
        kind=SpanKind.INTERNAL,
        context=run_context.trace_ctx,
        attributes={
            "agent.max_steps": run_context.max_steps_label,
            "agent.run.input": run_context.question,
            "agent.run.output": "",
            "langfuse.trace.name": "aceai.run",
            "langfuse.trace.input": run_context.question,
            "langfuse.trace.output": "",
        },
    )
    trace_context = set_span_in_context(run_span, run_context.trace_ctx or Context())
    set_trace_ctx(trace_context)

    error_msg: str | None = None
    try:
        if is_set(max_steps):
            step_iterator = range(len(run_context.steps), max_steps)
        else:
            step_iterator = count(len(run_context.steps))
        for _ in step_iterator:
            step_id = str(uuid4())
            event_builder = AgentEventBuilder(
                run_id=run_context.run_id,
                step_index=len(run_context.steps),
                step_id=step_id,
            )
            step_gen = _run_step(
                llm_service=llm_service,
                executor=executor,
                tracer=tracer,
                run_context=run_context,
                event_builder=event_builder,
            )
            try:
                async for event in step_gen:
                    if isinstance(event, LLMCompletedEvent):
                        run_context.steps.append(event.step)
                    elif isinstance(event, RunCompletedEvent):
                        run_context.run_state.status = "completed"
                        run_span.set_attribute(
                            "agent.run.output", event.final_answer
                        )
                        run_span.set_attribute(
                            "langfuse.trace.output", event.final_answer
                        )
                        yield event
                        return
                    elif isinstance(event, RunSuspendedEvent):
                        yield event
                        return
                    yield event
            except LLMProviderError as exc:
                run_context.run_state.status = "failed"
                error_msg = str(exc)
                failed_step = AgentStep(
                    step_id=step_id,
                    llm_response=LLMResponse(text="", status="failed"),
                )
                run_context.steps.append(failed_step)
                for event in _handle_failed_step(
                    run_context,
                    failed_step,
                    len(run_context.steps) - 1,
                    error_msg=error_msg,
                ):
                    yield event
                return
            except Exception as exc:
                run_context.run_state.status = "failed"
                if not run_context.steps:
                    raise
                error_msg = str(exc)
                last_step = run_context.steps[-1]
                last_index = len(run_context.steps) - 1
                for event in _handle_failed_step(
                    run_context,
                    last_step,
                    last_index,
                    exc=exc,
                ):
                    yield event
                raise
            finally:
                await step_gen.aclose()
        else:
            error_msg = (
                f"Agent exceeded maximum steps: {max_steps} without answering"
            )
            last_step = run_context.steps[-1]
            last_index = len(run_context.steps) - 1
            for event in _handle_failed_step(
                run_context,
                last_step,
                last_index,
                error_msg=error_msg,
            ):
                yield event
            raise AceAIRuntimeError(error_msg)
    finally:
        if error_msg is not None:
            run_span.set_attribute("agent.run.output", error_msg)
            run_span.set_attribute("langfuse.trace.output", error_msg)
        set_trace_ctx(None)
        if run_span.is_recording():
            run_span.end()


async def resume_agent_approval(
    *,
    llm_service: ILLMService,
    executor: IExecutor,
    tracer: trace.Tracer,
    max_steps: Unset[int],
    run_context: AgentRunContext,
    decision: ToolApprovalDecision,
) -> AsyncGenerator[AgentEvent, None]:
    try:
        pending = run_context.run_state.resume_from_approval()
    except ValueError:
        raise AceAIRuntimeError("agent run is not suspended for tool approval")
    request = pending.request
    if decision.call_id != request.call.call_id:
        raise AceAIRuntimeError("approval decision does not match pending tool call")

    event_builder = AgentEventBuilder(
        run_id=pending.run_id,
        step_index=pending.step_index,
        step_id=pending.step_id,
    )

    yield event_builder.tool_approval_resolved(
        request=request,
        decision=decision,
    )
    if decision.approved:
        run_context.run_state.tools.approved_tool_names.add(
            pending.invocation.tool.name
        )
        async for event in _execute_invocation(
            executor=executor,
            run_context=run_context,
            current_step=pending.step,
            event_builder=event_builder,
            invocation=pending.invocation,
        ):
            if isinstance(event, ToolCompletedEvent | ToolFailedEvent):
                run_context.context.add_tool_use(event)
            yield event
    else:
        async for event in _reject_invocation(
            pending.step,
            event_builder,
            request,
            decision,
        ):
            if isinstance(event, ToolCompletedEvent | ToolFailedEvent):
                run_context.context.add_tool_use(event)
            yield event

    tool_gen = _make_toolcalls(
        executor=executor,
        run_context=run_context,
        current_step=pending.step,
        event_builder=event_builder,
        start_index=pending.tool_index + 1,
    )
    try:
        async for event in tool_gen:
            if isinstance(event, ToolCompletedEvent | ToolFailedEvent):
                run_context.context.add_tool_use(event)
            yield event
            if isinstance(event, RunSuspendedEvent):
                return
    finally:
        await tool_gen.aclose()
    if run_context.run_state.status == "suspended":
        return
    yield event_builder.step_completed(step=pending.step)
    async for event in execute_agent_run(
        llm_service=llm_service,
        executor=executor,
        tracer=tracer,
        max_steps=max_steps,
        run_context=run_context,
    ):
        yield event


async def _call_llm(
    *,
    llm_service: ILLMService,
    executor: IExecutor,
    run_context: AgentRunContext,
    event_builder: AgentEventBuilder,
):
    tools: list[LLMToolSpec] = []
    tools.extend(executor.select_tools())
    tools.extend(executor.hosted_tools)

    compressed_after_context_window_error = False
    while True:
        compression_count_before_prepare = run_context.context.compression_count
        preflight_compaction_started = False
        if run_context.context.needs_compression(
            tools=tools
        ) and run_context.context.has_compressible_context():
            preflight_compaction_started = True
            yield event_builder.context_compaction_started(
                reason="threshold",
                compression_count=compression_count_before_prepare + 1,
            )
        try:
            messages = await run_context.context.prepare_for_llm(
                llm_service=llm_service,
                tools=tools,
            )
        except LLMProviderError as exc:
            if preflight_compaction_started:
                yield event_builder.context_compaction_failed(
                    reason="threshold",
                    compression_count=compression_count_before_prepare + 1,
                    error=str(exc),
                )
            raise
        if run_context.context.compression_count > compression_count_before_prepare:
            yield event_builder.context_compressed(
                reason="threshold",
                compression_count=run_context.context.compression_count,
                history=list(run_context.context.context[1:]),
            )

        if tools:
            stream = llm_service.stream(
                messages=messages,
                tools=tools,
                metadata=run_context.request_meta,
            )
        else:
            stream = llm_service.stream(
                messages=messages,
                metadata=run_context.request_meta,
            )

        try:
            reasoning_streamed = False
            async for stream_event in stream:
                match stream_event.event_type:
                    case "response.output_text.delta":
                        chunk = stream_event.text_delta
                        if isinstance(chunk, str):
                            yield event_builder.llm_text_delta(text_delta=chunk)
                    case "response.reasoning.delta":
                        reasoning_streamed = True
                        for segment in stream_event.segments:
                            if segment.type == "reasoning":
                                yield event_builder.llm_reasoning(segment=segment)
                    case "response.media":
                        yield event_builder.llm_media(segments=stream_event.segments)
                    case "response.function_call_arguments.delta":
                        tool_call_delta = stream_event.tool_call_delta
                        if isinstance(tool_call_delta, LLMToolCallDelta):
                            yield event_builder.llm_tool_call_delta(
                                tool_call_delta=tool_call_delta,
                            )
                    case "response.error":
                        if isinstance(stream_event.error, str):
                            raise LLMProviderError(stream_event.error)
                        raise LLMProviderError("LLM streaming error")
                    case "response.retrying":
                        if not isinstance(stream_event.error, str):
                            raise AceAIRuntimeError("LLM retry event missing error")
                        yield event_builder.llm_retrying(
                            retry_count=stream_event.retry_count,
                            retry_max=stream_event.retry_max,
                            retry_delay_seconds=stream_event.retry_delay_seconds,
                            error=stream_event.error,
                        )
                    case "response.completed":
                        response = stream_event.response
                        if not isinstance(response, LLMResponse):
                            raise AceAIRuntimeError(
                                "LLM stream completed without a response payload"
                            )
                        if response.status == "failed":
                            raise LLMProviderError(response.text)
                        for segment in response.segments:
                            if segment.type == "reasoning" and not reasoning_streamed:
                                yield event_builder.llm_reasoning(segment=segment)
                        yield AgentStep(
                            step_id=event_builder.step_id,
                            llm_response=response,
                        )
                        return
                    case _:
                        raise AceAIRuntimeError(
                            f"Unsupported LLM stream event: {stream_event.event_type}"
                        )
        except LLMContextWindowExceededError:
            if compressed_after_context_window_error:
                raise
            compression_started = False
            if run_context.context.has_compressible_context(
                force_current_run_steps=True,
            ):
                compression_started = True
                yield event_builder.context_compaction_started(
                    reason="context_window_retry",
                    compression_count=run_context.context.compression_count + 1,
                )
            try:
                compressed = await run_context.context.compress(
                    llm_service=llm_service,
                    force_current_run_steps=True,
                )
            except LLMProviderError as exc:
                if compression_started:
                    yield event_builder.context_compaction_failed(
                        reason="context_window_retry",
                        compression_count=run_context.context.compression_count + 1,
                        error=str(exc),
                    )
                raise
            if not compressed:
                raise LLMProviderError(
                    "Context compaction could not reduce this context-window retry "
                    "because there are no completed prior runs or completed "
                    "current-run steps available to summarize. The oversized "
                    "content is in required context such as the current user "
                    "message, open tool exchange, system instructions, tool "
                    "schemas, or attached context."
                )
            yield event_builder.context_compressed(
                reason="context_window_retry",
                compression_count=run_context.context.compression_count,
                history=list(run_context.context.context[1:]),
            )
            compressed_after_context_window_error = True
        finally:
            await stream.aclose()


async def _make_toolcalls(
    *,
    executor: IExecutor,
    run_context: AgentRunContext,
    current_step: AgentStep,
    event_builder: AgentEventBuilder,
    start_index: int = 0,
):
    tool_calls = current_step.llm_response.tool_calls

    if not tool_calls:
        return

    index = start_index
    while index < len(tool_calls):
        invocations: list[ToolInvocation] = []
        while index < len(tool_calls):
            call = tool_calls[index]
            invocation = executor.resolve_invocation(call)
            if (
                invocation.approval_required
                and invocation.tool.name
                not in run_context.run_state.tools.approved_tool_names
            ):
                break
            yield event_builder.tool_started(tool_call=call)
            invocations.append(invocation)
            index += 1

        async for event in _execute_invocations(
            executor=executor,
            run_context=run_context,
            current_step=current_step,
            event_builder=event_builder,
            invocations=invocations,
        ):
            yield event

        if index == len(tool_calls):
            return

        call = tool_calls[index]
        invocation = executor.resolve_invocation(call)
        yield event_builder.tool_started(tool_call=call)
        request = ToolApprovalRequest(
            call=call,
            tool_name=invocation.tool.name,
            reason=f"Tool {invocation.tool.name!r} requires approval",
            policy=invocation.tool.metadata.approval_policy,
        )
        run_context.run_state.suspend_for_approval(
            step=current_step,
            invocation=invocation,
            request=request,
            run_id=event_builder.run_id,
            step_index=event_builder.step_index,
            step_id=event_builder.step_id,
            tool_index=index,
        )
        yield event_builder.tool_approval_requested(request=request)
        yield event_builder.run_suspended(request=request)
        return


async def _execute_invocations(
    *,
    executor: IExecutor,
    run_context: AgentRunContext,
    current_step: AgentStep,
    event_builder: AgentEventBuilder,
    invocations: list[ToolInvocation],
):
    if len(invocations) == 1:
        yield await _execute_invocation_event(
            executor=executor,
            run_context=run_context,
            current_step=current_step,
            event_builder=event_builder,
            invocation=invocations[0],
        )
        return
    tasks: list[asyncio.Task[ToolCompletedEvent | ToolFailedEvent]] = []
    async with asyncio.TaskGroup() as task_group:
        for invocation in invocations:
            tasks.append(
                task_group.create_task(
                    _execute_invocation_event(
                        executor=executor,
                        run_context=run_context,
                        current_step=current_step,
                        event_builder=event_builder,
                        invocation=invocation,
                    )
                )
            )
    for task in tasks:
        yield task.result()


async def _execute_invocation(
    *,
    executor: IExecutor,
    run_context: AgentRunContext,
    current_step: AgentStep,
    event_builder: AgentEventBuilder,
    invocation: ToolInvocation,
):
    yield await _execute_invocation_event(
        executor=executor,
        run_context=run_context,
        current_step=current_step,
        event_builder=event_builder,
        invocation=invocation,
    )


async def _execute_invocation_event(
    *,
    executor: IExecutor,
    run_context: AgentRunContext,
    current_step: AgentStep,
    event_builder: AgentEventBuilder,
    invocation: ToolInvocation,
) -> ToolCompletedEvent | ToolFailedEvent:
    call = invocation.call
    try:
        tool_output = await _execute_tool_invocation(
            executor=executor,
            run_context=run_context,
            invocation=invocation,
        )
    except ToolExecutionError as exc:
        error_msg = str(exc)
        tool_result = ToolExecutionResult(
            call=call,
            output=f"Tool execution failed: {error_msg}",
            truncated_output=f"Tool execution failed: {error_msg}",
            error=error_msg,
        )
        current_step.tool_results.append(tool_result)
        return event_builder.tool_failed(
            tool_call=call,
            tool_result=tool_result,
            error=error_msg,
        )

    if type(tool_output) is str:
        tool_output = ToolExecutionOutput(
            output=tool_output,
            truncated_output=truncate_output(tool_output),
        )
    tool_result = ToolExecutionResult(
        call=call,
        output=tool_output.output,
        truncated_output=truncate_output(tool_output.truncated_output),
    )
    current_step.tool_results.append(tool_result)
    return event_builder.tool_completed(
        tool_call=call,
        tool_result=tool_result,
    )


async def _execute_tool_invocation(
    *,
    executor: IExecutor,
    run_context: AgentRunContext,
    invocation: ToolInvocation,
) -> ToolExecutionOutput:
    return await executor.execute(
        invocation,
        tool_state=run_context.run_state.tools,
    )


async def _reject_invocation(
    current_step: AgentStep,
    event_builder: AgentEventBuilder,
    request: ToolApprovalRequest,
    decision: ToolApprovalDecision,
):
    error_msg = decision.reason or "Tool execution rejected by caller"
    tool_result = ToolExecutionResult(
        call=request.call,
        output=f"Tool execution rejected: {error_msg}",
        truncated_output=f"Tool execution rejected: {error_msg}",
        error=error_msg,
    )
    current_step.tool_results.append(tool_result)
    yield event_builder.tool_failed(
        tool_call=request.call,
        tool_result=tool_result,
        error=error_msg,
    )


async def _run_step(
    *,
    llm_service: ILLMService,
    executor: IExecutor,
    tracer: trace.Tracer,
    run_context: AgentRunContext,
    event_builder: AgentEventBuilder,
) -> AsyncGenerator[AgentEvent, None]:
    trace_context = get_trace_ctx()
    if trace_context is None:
        raise AceAIRuntimeError("trace context is not set for agent step")
    step_span = tracer.start_span(
        "agent.step",
        kind=SpanKind.INTERNAL,
        context=trace_context,
        attributes={
            "agent.step_id": event_builder.step_id,
            "agent.max_steps": run_context.max_steps_label,
        },
    )
    step_context = set_span_in_context(step_span, trace_context)
    set_trace_ctx(step_context)

    try:
        yield event_builder.llm_started()
        llm_gen = _call_llm(
            llm_service=llm_service,
            executor=executor,
            run_context=run_context,
            event_builder=event_builder,
        )
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
                run_context.context.add_tool_call(current_step.llm_response)
                tool_gen = _make_toolcalls(
                    executor=executor,
                    run_context=run_context,
                    current_step=current_step,
                    event_builder=event_builder,
                )
                try:
                    async for tool_event in tool_gen:
                        if isinstance(tool_event, ToolCompletedEvent | ToolFailedEvent):
                            run_context.context.add_tool_use(tool_event)
                        yield tool_event
                        if isinstance(tool_event, RunSuspendedEvent):
                            return
                finally:
                    await tool_gen.aclose()
                if run_context.run_state.status == "suspended":
                    return
                yield event_builder.step_completed(step=current_step)
        finally:
            await llm_gen.aclose()

    finally:
        set_trace_ctx(trace_context)
        if step_span.is_recording():
            step_span.end()


def _handle_failed_step(
    run_context: AgentRunContext,
    step: AgentStep,
    step_index: int,
    exc: Exception | None = None,
    error_msg: str | None = None,
):
    step_id = step.step_id
    error_msg = error_msg or str(exc)

    yield StepFailedEvent(
        step_index=step_index,
        step_id=step_id,
        step=step,
        error=error_msg,
    )
    yield RunFailedEvent(
        run_id=run_context.run_id,
        step_index=step_index,
        step_id=step_id,
        step=step,
        error=error_msg,
    )
