from itertools import count
from pathlib import Path
from typing import AsyncGenerator, Callable, Literal, Unpack
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, set_span_in_context

from ..llm import ILLMService, LLMResponse
from ..llm.errors import AceAIConfigurationError, AceAIRuntimeError
from ..llm.interface import UNSET, Unset, is_set
from ..llm.models import (
    LLMHostedToolSpec,
    LLMMessage,
    LLMRequestMeta,
    LLMToolCallDelta,
    LLMToolSpec,
)
from .models import (
    AgentStep,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolExecutionResult,
)
from .skills import (
    SkillLoader,
    SkillRegistry,
    format_skills_for_prompt,
)
from aceai.llm.tracing import get_trace_ctx, set_trace_ctx
from .context_manager import CompressThreshold, ContextCompressionPolicy, ContextManager
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
from .executor import IExecutor, ToolExecutionError, ToolExecutor
from .run_state import AgentRuntimeState, AgentRuntimeStatus, ToolInvocation


class AgentRuntime:
    def __init__(
        self,
        *,
        question: str,
        prompt: str,
        messages: list[LLMMessage],
        llm_service: ILLMService,
        executor: IExecutor | None,
        hosted_tools: list[LLMHostedToolSpec],
        max_steps: Unset[int] = UNSET,
        tracer: trace.Tracer | None = None,
        max_steps_label: str | int = "unlimited",
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta,
        compression_policy: ContextCompressionPolicy | None = None,
    ) -> None:
        self.question = question
        self.context = ContextManager(
            prompt,
            compression_policy=compression_policy,
        )
        self.context.init_context(messages)
        self.llm_service = llm_service
        self.executor = executor
        self.hosted_tools = hosted_tools
        self.max_steps = max_steps
        self.max_steps_label = max_steps_label
        self.tracer = tracer or trace.get_tracer("aceai.core")
        self.trace_ctx = trace_ctx
        self.request_meta = request_meta
        self.steps: list[AgentStep] = []
        self.run_state = AgentRuntimeState()
        self.run_id = str(uuid4())

    @property
    def status(self) -> AgentRuntimeStatus:
        return self.run_state.status

    @classmethod
    def from_question(
        cls,
        *,
        question: str,
        prompt: str,
        llm_service: ILLMService,
        executor: IExecutor | None,
        hosted_tools: list[LLMHostedToolSpec],
        max_steps: Unset[int],
        max_steps_label: str | int,
        tracer: trace.Tracer,
        trace_ctx: Context | None,
        request_meta: LLMRequestMeta,
        compression_policy: ContextCompressionPolicy | None,
    ) -> "AgentRuntime":
        return cls(
            question=question,
            prompt=prompt,
            messages=[LLMMessage.build(role="user", content=question)],
            llm_service=llm_service,
            executor=executor,
            hosted_tools=hosted_tools,
            max_steps=max_steps,
            max_steps_label=max_steps_label,
            tracer=tracer,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            compression_policy=compression_policy,
        )

    @classmethod
    def from_history(
        cls,
        *,
        question: str,
        history: list[LLMMessage],
        prompt: str,
        llm_service: ILLMService,
        executor: IExecutor | None,
        hosted_tools: list[LLMHostedToolSpec],
        max_steps: Unset[int],
        max_steps_label: str | int,
        tracer: trace.Tracer,
        trace_ctx: Context | None,
        request_meta: LLMRequestMeta,
        compression_policy: ContextCompressionPolicy | None,
    ) -> "AgentRuntime":
        return cls(
            question=question,
            prompt=prompt,
            messages=list(history) + [LLMMessage.build(role="user", content=question)],
            llm_service=llm_service,
            executor=executor,
            hosted_tools=hosted_tools,
            max_steps=max_steps,
            max_steps_label=max_steps_label,
            tracer=tracer,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            compression_policy=compression_policy,
        )

    async def _call_llm(
        self,
        event_builder: AgentEventBuilder,
    ):
        executor = self.executor

        tools: list[LLMToolSpec] = []
        if executor:
            tools.extend(executor.select_tools())
        tools.extend(self.hosted_tools)

        messages = await self.context.prepare_for_llm(llm_service=self.llm_service)

        if tools:
            stream = self.llm_service.stream(
                messages=messages,
                tools=tools,
                metadata=self.request_meta,
            )
        else:
            stream = self.llm_service.stream(
                messages=messages,
                metadata=self.request_meta,
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
                            raise AceAIRuntimeError(stream_event.error)
                        raise AceAIRuntimeError("LLM streaming error")
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
        finally:
            await stream.aclose()

    async def _make_toolcalls(
        self,
        current_step: AgentStep,
        event_builder: AgentEventBuilder,
        start_index: int = 0,
    ):
        executor = self.executor
        tool_calls = current_step.llm_response.tool_calls

        if not tool_calls:
            return

        if executor is None:
            raise AceAIConfigurationError(
                "executor must be provided when tool calls are enabled"
            )

        for index, call in enumerate(tool_calls[start_index:], start=start_index):
            invocation = executor.resolve_invocation(call)
            yield event_builder.tool_started(tool_call=call)
            if (
                invocation.approval_required
                and invocation.tool.name not in self.run_state.tools.approved_tool_names
            ):
                request = ToolApprovalRequest(
                    call=call,
                    tool_name=invocation.tool.name,
                    reason=f"Tool {invocation.tool.name!r} requires approval",
                    policy=invocation.tool.metadata.approval_policy,
                )
                self.run_state.suspend_for_approval(
                    step=current_step,
                    invocation=invocation,
                    request=request,
                    event_builder=event_builder,
                    tool_index=index,
                )
                yield event_builder.tool_approval_requested(request=request)
                yield event_builder.run_suspended(request=request)
                return
            async for event in self._execute_invocation(
                current_step,
                event_builder,
                invocation,
            ):
                yield event

    async def _execute_invocation(
        self,
        current_step: AgentStep,
        event_builder: AgentEventBuilder,
        invocation: ToolInvocation,
    ):
        call = invocation.call
        try:
            tool_output = await self._execute_tool_invocation(invocation)
        except ToolExecutionError as exc:
            error_msg = str(exc)
            tool_result = ToolExecutionResult(
                call=call,
                output=f"Tool execution failed: {error_msg}",
                error=error_msg,
            )
            current_step.tool_results.append(tool_result)
            yield event_builder.tool_failed(
                tool_call=call,
                tool_result=tool_result,
                error=error_msg,
            )
            return

        tool_result = ToolExecutionResult(call=call, output=tool_output)
        current_step.tool_results.append(tool_result)
        yield event_builder.tool_completed(
            tool_call=call,
            tool_result=tool_result,
        )

    async def _execute_tool_invocation(self, invocation: ToolInvocation) -> str:
        executor = self.executor
        if executor is None:
            raise AceAIConfigurationError(
                "executor must be provided when tool calls are enabled"
            )
        return await executor.execute(
            invocation,
            tool_state=self.run_state.tools,
        )

    async def _reject_invocation(
        self,
        current_step: AgentStep,
        event_builder: AgentEventBuilder,
        request: ToolApprovalRequest,
        decision: ToolApprovalDecision,
    ):
        error_msg = decision.reason or "Tool execution rejected by caller"
        tool_result = ToolExecutionResult(
            call=request.call,
            output=f"Tool execution rejected: {error_msg}",
            error=error_msg,
        )
        current_step.tool_results.append(tool_result)
        yield event_builder.tool_failed(
            tool_call=request.call,
            tool_result=tool_result,
            error=error_msg,
        )

    async def _run_step(
        self,
        *,
        event_builder: AgentEventBuilder,
    ) -> AsyncGenerator[AgentEvent, None]:
        run_context = get_trace_ctx()
        if run_context is None:
            raise AceAIRuntimeError("trace context is not set for agent step")
        step_span = self.tracer.start_span(
            "agent.step",
            kind=SpanKind.INTERNAL,
            context=run_context,
            attributes={
                "agent.step_id": event_builder.step_id,
                "agent.max_steps": self.max_steps_label,
            },
        )
        step_context = set_span_in_context(step_span, run_context)
        set_trace_ctx(step_context)

        try:
            yield event_builder.llm_started()
            llm_gen = self._call_llm(event_builder=event_builder)
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
                    self.context.add_tool_call(current_step.llm_response)
                    tool_gen = self._make_toolcalls(
                        current_step,
                        event_builder,
                    )
                    try:
                        async for tool_event in tool_gen:
                            if isinstance(
                                tool_event, ToolCompletedEvent | ToolFailedEvent
                            ):
                                self.context.add_tool_use(tool_event)
                            yield tool_event
                            if isinstance(tool_event, RunSuspendedEvent):
                                return
                    finally:
                        await tool_gen.aclose()
                    if self.run_state.status == "suspended":
                        return
                    yield event_builder.step_completed(step=current_step)
            finally:
                await llm_gen.aclose()

        finally:
            set_trace_ctx(run_context)
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

        yield StepFailedEvent(
            step_index=step_index,
            step_id=step_id,
            step=step,
            error=error_msg,
        )
        yield RunFailedEvent(
            run_id=self.run_id,
            step_index=step_index,
            step_id=step_id,
            step=step,
            error=error_msg,
        )

    async def execute(self) -> AsyncGenerator[AgentEvent, None]:
        if self.run_state.pending_approval is not None:
            raise AceAIRuntimeError(
                "agent run is suspended; call resume_approval() with a decision"
            )
        self.run_state.status = "running"
        run_span = self.tracer.start_span(
            "agent.run",
            kind=SpanKind.INTERNAL,
            context=self.trace_ctx,
            attributes={
                "agent.max_steps": self.max_steps_label,
                "agent.run.input": self.question,
                "agent.run.output": "",
                "langfuse.trace.name": "aceai.run",
                "langfuse.trace.input": self.question,
                "langfuse.trace.output": "",
            },
        )
        run_context = set_span_in_context(run_span, self.trace_ctx or Context())
        set_trace_ctx(run_context)

        error_msg: str | None = None
        try:
            if is_set(self.max_steps):
                step_iterator = range(len(self.steps), self.max_steps)
            else:
                step_iterator = count(len(self.steps))
            for _ in step_iterator:
                step_id = str(uuid4())
                event_builder = AgentEventBuilder(
                    run_id=self.run_id,
                    step_index=len(self.steps),
                    step_id=step_id,
                )
                step_gen = self._run_step(
                    event_builder=event_builder,
                )
                try:
                    async for event in step_gen:
                        if isinstance(event, LLMCompletedEvent):
                            self.steps.append(event.step)
                        elif isinstance(event, RunCompletedEvent):
                            self.run_state.status = "completed"
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
                except Exception as exc:
                    self.run_state.status = "failed"
                    if not self.steps:
                        raise
                    error_msg = str(exc)
                    last_step = self.steps[-1]
                    last_index = len(self.steps) - 1
                    for event in self._handle_failed_step(
                        last_step, last_index, exc=exc
                    ):
                        yield event
                    raise
                finally:
                    await step_gen.aclose()
            else:
                error_msg = (
                    f"Agent exceeded maximum steps: {self.max_steps} without answering"
                )
                last_step = self.steps[-1]
                last_index = len(self.steps) - 1
                for event in self._handle_failed_step(
                    last_step, last_index, error_msg=error_msg
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

    async def resume_approval(
        self,
        decision: ToolApprovalDecision,
    ) -> AsyncGenerator[AgentEvent, None]:
        try:
            pending = self.run_state.resume_from_approval()
        except ValueError as exc:
            raise AceAIRuntimeError("agent run is not suspended for tool approval")
        request = pending.request
        if decision.call_id != request.call.call_id:
            raise AceAIRuntimeError("approval decision does not match pending tool call")

        yield pending.event_builder.tool_approval_resolved(
            request=request,
            decision=decision,
        )
        if decision.approved:
            self.run_state.tools.approved_tool_names.add(pending.invocation.tool.name)
            async for event in self._execute_invocation(
                pending.step,
                pending.event_builder,
                pending.invocation,
            ):
                if isinstance(event, ToolCompletedEvent | ToolFailedEvent):
                    self.context.add_tool_use(event)
                yield event
        else:
            async for event in self._reject_invocation(
                pending.step,
                pending.event_builder,
                request,
                decision,
            ):
                if isinstance(event, ToolCompletedEvent | ToolFailedEvent):
                    self.context.add_tool_use(event)
                yield event

        tool_gen = self._make_toolcalls(
            pending.step,
            pending.event_builder,
            start_index=pending.tool_index + 1,
        )
        try:
            async for event in tool_gen:
                if isinstance(event, ToolCompletedEvent | ToolFailedEvent):
                    self.context.add_tool_use(event)
                yield event
                if isinstance(event, RunSuspendedEvent):
                    return
        finally:
            await tool_gen.aclose()
        if self.run_state.status == "suspended":
            return
        yield pending.event_builder.step_completed(step=pending.step)
        async for event in self.execute():
            yield event


class AgentBase:
    """Base class for agents using an LLM provider."""

    def __init__(
        self,
        prompt: str = "",
        *,
        default_model: str,
        llm_service: ILLMService,
        max_steps: Unset[int] = UNSET,
        executor: IExecutor | None = None,
        tracer: trace.Tracer | None = None,
        skill_path: str | Path | Literal["auto", "disable"] = "auto",
        enabled_skill_names: Unset[tuple[str, ...]] = UNSET,
        skill_loader_factory: Callable[[str], SkillLoader] = SkillLoader,
        hosted_tools: list[LLMHostedToolSpec] | None = None,
        compress_threshold: CompressThreshold = "100%",
        context_window_tokens: int = 128000,
    ):
        if is_set(max_steps) and max_steps < 1:
            raise AceAIConfigurationError("max_steps must be positive or UNSET")
        self._skill_registry = SkillLoader.load_registry(
            skill_path,
            loader_factory=skill_loader_factory,
        )
        if is_set(enabled_skill_names):
            self._skill_registry = self._skill_registry.select(enabled_skill_names)
        skill_prompt = format_skills_for_prompt(self._skill_registry)
        self._default_model = default_model
        self._llm_service = llm_service
        self._prompt = prompt
        self._ctx_mgr: ContextManager = ContextManager(prompt + skill_prompt)
        self._executor = executor
        self._hosted_tools = hosted_tools if hosted_tools is not None else []
        self._compression_policy = ContextCompressionPolicy(
            compress_threshold,
            context_window_tokens=context_window_tokens,
        )
        if isinstance(executor, ToolExecutor) and self._skill_registry.get_skills():
            executor.register_tools(*self._skill_registry.as_tools())
        self._max_steps = max_steps
        if is_set(max_steps):
            self._max_steps_label = max_steps
        else:
            self._max_steps_label = "unlimited"
        self._tracer = tracer or trace.get_tracer("aceai.core")

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def llm_service(self) -> ILLMService:
        return self._llm_service

    @property
    def executor(self) -> IExecutor | None:
        return self._executor

    @property
    def hosted_tools(self) -> list[LLMHostedToolSpec]:
        return self._hosted_tools

    @property
    def max_steps(self) -> Unset[int]:
        return self._max_steps

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def system_message(self) -> LLMMessage:
        return self._ctx_mgr.system_message

    def add_instruction(self, instruction: str) -> None:
        """Add an instruction into the agent's context manager."""
        if instruction == "":
            raise ValueError("Empty Instruction")
        self._ctx_mgr.add_instruction(instruction)

    def create_run(
        self,
        question: str,
        trace_ctx: Context | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AgentRuntime:
        return AgentRuntime.from_question(
            question=question,
            prompt=self._ctx_mgr.instructions_text,
            llm_service=self._llm_service,
            executor=self._executor,
            hosted_tools=self._hosted_tools,
            max_steps=self._max_steps,
            max_steps_label=self._max_steps_label,
            tracer=self._tracer,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            compression_policy=self._compression_policy,
        )

    def create_resume_run(
        self,
        question: str,
        history: list[LLMMessage],
        trace_ctx: Context | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AgentRuntime:
        return AgentRuntime.from_history(
            question=question,
            history=history,
            prompt=self._ctx_mgr.instructions_text,
            llm_service=self._llm_service,
            executor=self._executor,
            hosted_tools=self._hosted_tools,
            max_steps=self._max_steps,
            max_steps_label=self._max_steps_label,
            tracer=self._tracer,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            compression_policy=self._compression_policy,
        )

    async def run(
        self,
        question: str,
        trace_ctx: Context | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvent entries as the agent reasons."""
        run = self.create_run(question, trace_ctx=trace_ctx, **request_meta)
        async for event in run.execute():
            yield event

    async def resume(
        self,
        question: str,
        history: list[LLMMessage],
        trace_ctx: Context | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvent entries with existing conversation history."""
        run = self.create_resume_run(
            question,
            history,
            trace_ctx=trace_ctx,
            **request_meta,
        )
        async for event in run.execute():
            yield event

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
