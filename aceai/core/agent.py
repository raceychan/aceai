from typing import AsyncGenerator, Generic, TypeVar, Unpack
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.context import Context

from ..llm import ILLMService
from ..llm.errors import AceAIConfigurationError, AceAIRuntimeError
from ..llm.interface import UNSET, Unset, is_set
from ..llm.models import (
    LLMHostedToolSpec,
    LLMMessage,
    LLMMessagePart,
    LLMRequestMeta,
    SupportedValueType,
)
from .context_manager import (
    CompressThreshold,
    ContextCompressionPolicy,
    ContextManager,
    PromptBlock,
)
from .events import AgentEvent, RunCompletedEvent
from .executor import DummyExecutor, IExecutor
from .hooks import (
    HookPlan,
    HookRegistry,
    PreparedModelRequest,
)
from .models import ToolApprovalDecision
from .run_loop import (
    AgentRunContext,
    execute_agent_run,
    prepare_agent_model_request,
    resume_agent_approval,
)
from .skills import SkillRegistry


TContext = TypeVar("TContext")


class Agent(Generic[TContext]):
    """Agent definition using an LLM provider."""

    def __init__(
        self,
        prompt: str = "",
        *,
        prompt_blocks: tuple[PromptBlock, ...] = (),
        default_model: str,
        llm_service: ILLMService,
        max_steps: Unset[int] = UNSET,
        executor: IExecutor = DummyExecutor(),
        tracer: trace.Tracer | None = None,
        compress_threshold: CompressThreshold = "100%",
        context_window_tokens: int = 128000,
        agent_id: str = "default",
        hook_registry: HookRegistry[TContext] | None = None,
    ):
        if is_set(max_steps) and max_steps < 1:
            raise AceAIConfigurationError("max_steps must be positive or UNSET")
        self._agent_id = agent_id
        self._default_model = default_model
        self._llm_service = llm_service
        self._prompt = prompt
        if executor is None:
            raise TypeError("executor must be IExecutor")
        self._executor = executor
        self._ctx_mgr: ContextManager = ContextManager(
            _agent_prompt_blocks(
                prompt=prompt,
                extra_blocks=prompt_blocks,
                executor_blocks=executor.prompt_blocks,
            )
        )
        self._compression_policy = ContextCompressionPolicy(
            compress_threshold,
            context_window_tokens=context_window_tokens,
        )
        self._max_steps = max_steps
        if is_set(max_steps):
            self._max_steps_label = max_steps
        else:
            self._max_steps_label = "unlimited"
        self._tracer = tracer or trace.get_tracer("aceai.core")
        if hook_registry is None:
            self._hook_plan = HookPlan.empty()
        else:
            self._hook_plan = hook_registry.build_plan()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def llm_service(self) -> ILLMService:
        return self._llm_service

    @property
    def executor(self) -> IExecutor:
        return self._executor

    @property
    def hosted_tools(self) -> list[LLMHostedToolSpec]:
        return self._executor.hosted_tools

    @property
    def max_steps(self) -> Unset[int]:
        return self._max_steps

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._executor.skill_registry

    @property
    def system_message(self) -> LLMMessage:
        return self._ctx_mgr.system_message

    @property
    def system_prompt_blocks(self) -> tuple[PromptBlock, ...]:
        return self._ctx_mgr.instruction_blocks

    def add_instruction(self, block: PromptBlock) -> None:
        """Add an instruction into the agent's context manager."""
        if block.content == "":
            raise ValueError("Empty Instruction")
        self._ctx_mgr.add_instruction(block)

    def create_run(
        self,
        question: SupportedValueType,
        trace_ctx: Context | None = None,
        *,
        hook_context: TContext | None = None,
        hook_registry: HookRegistry[TContext] | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AgentRunContext:
        context = ContextManager(
            self._ctx_mgr.instruction_blocks,
            compression_policy=self._compression_policy,
        )
        context.init_context([LLMMessage.build(role="user", content=question)])
        return self._create_run_context(
            agent_id=self._agent_id,
            question=_question_preview(question),
            context=context,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            hook_context=hook_context,
            hook_registry=hook_registry,
        )

    def create_resume_run(
        self,
        question: SupportedValueType,
        history: list[LLMMessage],
        trace_ctx: Context | None = None,
        *,
        hook_context: TContext | None = None,
        hook_registry: HookRegistry[TContext] | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AgentRunContext:
        context = ContextManager(
            self._ctx_mgr.instruction_blocks,
            compression_policy=self._compression_policy,
        )
        context.init_context(
            list(history) + [LLMMessage.build(role="user", content=question)]
        )
        return self._create_run_context(
            agent_id=self._agent_id,
            question=_question_preview(question),
            context=context,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            hook_context=hook_context,
            hook_registry=hook_registry,
        )

    def _create_run_context(
        self,
        *,
        agent_id: str,
        question: str,
        context: ContextManager,
        trace_ctx: Context | None,
        request_meta: LLMRequestMeta,
        hook_context: TContext | None,
        hook_registry: HookRegistry[TContext] | None,
    ) -> AgentRunContext:
        if hook_registry is None:
            hook_plan = self._hook_plan
        else:
            hook_plan = self._hook_plan.combine(hook_registry.build_plan())
        return AgentRunContext(
            agent_id=agent_id,
            run_id=str(uuid4()),
            question=question,
            context=context,
            max_steps_label=self._max_steps_label,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
            hook_plan=hook_plan,
            hook_context=hook_context,
        )

    async def execute(
        self,
        run_context: AgentRunContext,
    ) -> AsyncGenerator[AgentEvent, None]:
        self._ensure_run_context_owner(run_context)
        async for event in execute_agent_run(
            llm_service=self._llm_service,
            executor=self._executor,
            tracer=self._tracer,
            max_steps=self._max_steps,
            run_context=run_context,
        ):
            yield event

    async def resume_approval(
        self,
        run_context: AgentRunContext,
        decision: ToolApprovalDecision,
    ) -> AsyncGenerator[AgentEvent, None]:
        self._ensure_run_context_owner(run_context)
        async for event in resume_agent_approval(
            llm_service=self._llm_service,
            executor=self._executor,
            tracer=self._tracer,
            max_steps=self._max_steps,
            run_context=run_context,
            decision=decision,
        ):
            yield event

    async def prepare_model_request(
        self,
        run_context: AgentRunContext,
        *,
        step_id: str | None = None,
        step_index: int | None = None,
    ) -> PreparedModelRequest:
        self._ensure_run_context_owner(run_context)
        return await prepare_agent_model_request(
            llm_service=self._llm_service,
            executor=self._executor,
            run_context=run_context,
            step_id=step_id,
            step_index=step_index,
        )

    def _ensure_run_context_owner(self, run_context: AgentRunContext) -> None:
        if run_context.agent_id != self._agent_id:
            raise AceAIRuntimeError("agent run context belongs to a different agent")

    async def run(
        self,
        question: SupportedValueType,
        trace_ctx: Context | None = None,
        *,
        hook_context: TContext | None = None,
        hook_registry: HookRegistry[TContext] | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvent entries as the agent reasons."""
        run_context = self.create_run(
            question,
            trace_ctx=trace_ctx,
            hook_context=hook_context,
            hook_registry=hook_registry,
            **request_meta,
        )
        async for event in self.execute(run_context):
            yield event

    async def resume(
        self,
        question: SupportedValueType,
        history: list[LLMMessage],
        trace_ctx: Context | None = None,
        *,
        hook_context: TContext | None = None,
        hook_registry: HookRegistry[TContext] | None = None,
        **request_meta: Unpack[LLMRequestMeta],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvent entries with existing conversation history."""
        run_context = self.create_resume_run(
            question,
            history,
            trace_ctx=trace_ctx,
            hook_context=hook_context,
            hook_registry=hook_registry,
            **request_meta,
        )
        async for event in self.execute(run_context):
            yield event

    async def ask(
        self,
        question: SupportedValueType,
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


def _question_preview(content: SupportedValueType) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts: list[LLMMessagePart] = [content]
    else:
        parts = content
    previews: list[str] = []
    image_count = 0
    for part in parts:
        if part["type"] == "text":
            previews.append(str(part.get("data", "")))
        elif part["type"] == "image":
            image_count += 1
    if image_count > 0:
        suffix = "" if image_count == 1 else "s"
        previews.append(f"[{image_count} image{suffix}]")
    return "\n".join(previews)


def _agent_prompt_blocks(
    *,
    prompt: str,
    extra_blocks: tuple[PromptBlock, ...],
    executor_blocks: tuple[PromptBlock, ...],
) -> tuple[PromptBlock, ...]:
    blocks: list[PromptBlock] = []
    if prompt:
        blocks.append(
            PromptBlock(
                slot="agent_instructions",
                source="agent.prompt",
                content=prompt,
                detail="base agent instructions",
            )
        )
    blocks.extend(extra_blocks)
    blocks.extend(executor_blocks)
    return tuple(blocks)
