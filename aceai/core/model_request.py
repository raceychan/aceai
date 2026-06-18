from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from aceai.llm import ILLMService
from aceai.llm.models import (
    LLMHostedToolSpec,
    LLMMessage,
    LLMRequestMeta,
    LLMToolSpec,
)

from .context_manager import ContextManager
from .executor import IExecutor
from .hooks import (
    AppliedHookPatch,
    HookContext,
    HookMode,
    HookPlan,
    ModelRequestDraft,
    ModelRequestPatch,
    PreparedModelRequest,
)
from .hook_execution import (
    call_before_model_request_hook,
    call_model_request_committed_hook,
)


@dataclass(frozen=True)
class ModelRequestState:
    agent_id: str
    run_id: str
    step_id: str
    step_index: int
    request_meta: LLMRequestMeta
    hook_plan: HookPlan[Any]
    hook_context: Any | None = None


@dataclass(frozen=True)
class ModelRequestAssembly:
    request_id: str
    hook_context: HookContext[Any]
    tools: tuple[LLMToolSpec, ...]
    metadata: LLMRequestMeta
    patches: tuple[AppliedHookPatch, ...]


async def prepare_model_request(
    *,
    llm_service: ILLMService,
    executor: IExecutor,
    context: ContextManager,
    state: ModelRequestState,
    mode: HookMode,
) -> PreparedModelRequest:
    assembly = await assemble_model_request(
        executor=executor,
        context=context,
        state=state,
        mode=mode,
    )
    messages = await context.prepare_for_llm(
        llm_service=llm_service,
        tools=list(assembly.tools),
    )
    return prepared_model_request(
        assembly=assembly,
        messages=messages,
        state=state,
    )


async def assemble_model_request(
    *,
    executor: IExecutor,
    context: ContextManager,
    state: ModelRequestState,
    mode: HookMode,
) -> ModelRequestAssembly:
    tools: list[LLMToolSpec] = []
    tools.extend(executor.select_tools())
    tools.extend(executor.hosted_tools)
    metadata = dict(state.request_meta)
    hook_context = HookContext(
        mode=mode,
        run_id=state.run_id,
        step_id=state.step_id,
        step_index=state.step_index,
        agent_id=state.agent_id,
        data=state.hook_context,
    )
    draft = ModelRequestDraft(
        messages=tuple(context.context),
        tools=tuple(tools),
        metadata=cast(LLMRequestMeta, metadata),
    )
    patches = await collect_before_model_request_patches(
        plan=state.hook_plan,
        ctx=hook_context,
        draft=draft,
    )
    merge = merge_model_request_patches(patches)
    if merge.prepended_messages:
        context.context[1:1] = merge.prepended_messages
    if merge.appended_messages:
        context.context.extend(merge.appended_messages)
    tools = filter_model_request_tools(
        tools,
        allowlist=merge.tool_allowlist,
        denylist=merge.tool_denylist,
    )
    return ModelRequestAssembly(
        request_id=str(uuid4()),
        hook_context=hook_context,
        tools=tuple(tools),
        metadata=cast(LLMRequestMeta, {**metadata, **merge.metadata}),
        patches=merge.applied_patches,
    )


async def collect_before_model_request_patches(
    *,
    plan: HookPlan[Any],
    ctx: HookContext[Any],
    draft: ModelRequestDraft,
) -> tuple[tuple[ModelRequestPatch, AppliedHookPatch], ...]:
    patches: list[tuple[ModelRequestPatch, AppliedHookPatch]] = []
    for spec in plan.before_model_request:
        patch = await call_before_model_request_hook(spec, ctx, draft)
        patches.append(
            (
                patch,
                AppliedHookPatch(
                    hook_name=spec.name,
                    point="before_model_request",
                    prepended_message_count=len(patch.prepend_messages),
                    appended_message_count=len(patch.append_messages),
                    metadata_keys=tuple(patch.metadata),
                    tool_allowlist=patch.tool_allowlist,
                    tool_denylist=patch.tool_denylist,
                    trace=patch.trace,
                    effects=patch.effects,
                ),
            )
        )
    return tuple(patches)


@dataclass(frozen=True)
class ModelRequestPatchMerge:
    prepended_messages: list[LLMMessage]
    appended_messages: list[LLMMessage]
    metadata: dict[str, object]
    tool_allowlist: set[str] | None
    tool_denylist: set[str]
    applied_patches: tuple[AppliedHookPatch, ...]


def merge_model_request_patches(
    patches: tuple[tuple[ModelRequestPatch, AppliedHookPatch], ...],
) -> ModelRequestPatchMerge:
    prepended_messages: list[LLMMessage] = []
    appended_messages: list[LLMMessage] = []
    metadata: dict[str, object] = {}
    tool_allowlist: set[str] | None = None
    tool_denylist: set[str] = set()
    applied: list[AppliedHookPatch] = []
    for patch, applied_patch in patches:
        prepended_messages.extend(patch.prepend_messages)
        appended_messages.extend(patch.append_messages)
        metadata.update(patch.metadata)
        if patch.tool_allowlist is not None:
            allowlist = set(patch.tool_allowlist)
            tool_allowlist = (
                allowlist
                if tool_allowlist is None
                else tool_allowlist.intersection(allowlist)
            )
        tool_denylist.update(patch.tool_denylist)
        applied.append(applied_patch)
    return ModelRequestPatchMerge(
        prepended_messages=prepended_messages,
        appended_messages=appended_messages,
        metadata=metadata,
        tool_allowlist=tool_allowlist,
        tool_denylist=tool_denylist,
        applied_patches=tuple(applied),
    )


def prepared_model_request(
    *,
    assembly: ModelRequestAssembly,
    messages: list[LLMMessage],
    state: ModelRequestState,
) -> PreparedModelRequest:
    return PreparedModelRequest(
        request_id=assembly.request_id,
        attempt_id=str(uuid4()),
        run_id=state.run_id,
        step_id=state.step_id,
        step_index=state.step_index,
        messages=tuple(messages),
        tools=assembly.tools,
        metadata=assembly.metadata,
        patches=assembly.patches,
    )


async def run_model_request_committed_hooks(
    *,
    plan: HookPlan[Any],
    ctx: HookContext[Any],
    request: PreparedModelRequest,
) -> None:
    for spec in plan.on_model_request_committed:
        await call_model_request_committed_hook(spec, ctx, request)


def filter_model_request_tools(
    tools: list[LLMToolSpec],
    *,
    allowlist: set[str] | None,
    denylist: set[str],
) -> list[LLMToolSpec]:
    if allowlist is None and not denylist:
        return tools
    filtered: list[LLMToolSpec] = []
    for tool in tools:
        name = tool_spec_name(tool)
        if name in denylist:
            continue
        if allowlist is not None and name not in allowlist:
            continue
        filtered.append(tool)
    return filtered


def tool_spec_name(tool: LLMToolSpec) -> str:
    if isinstance(tool, LLMHostedToolSpec):
        return tool.native_name
    return tool.name
