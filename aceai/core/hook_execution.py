import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from aceai.llm.models import LLMResponse

from .hooks import (
    AfterModelResponseHookSpec,
    AfterToolExecuteHookSpec,
    BeforeModelRequestHookSpec,
    BeforeModelCallHookSpec,
    BeforeToolExecuteHookSpec,
    HookContext,
    HookExecutionError,
    ModelError,
    ModelErrorHookSpec,
    ModelRequestCommittedHookSpec,
    ModelRequestDraft,
    ModelRequestPatch,
    PreparedModelRequest,
    ToolExecutionOutcome,
    ToolExecutionPatch,
    ToolExecutionRequest,
)


TContext = TypeVar("TContext")
TResult = TypeVar("TResult")


async def call_before_model_request_hook(
    spec: BeforeModelRequestHookSpec[TContext],
    ctx: HookContext[TContext],
    draft: ModelRequestDraft,
) -> ModelRequestPatch:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="before_model_request",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, draft),
    )
    if result is None:
        return ModelRequestPatch()
    if not isinstance(result, ModelRequestPatch):
        raise HookExecutionError(
            f"hook {spec.name!r} returned {type(result).__name__}; "
            "before_model_request hooks must return ModelRequestPatch or None"
        )
    return result


async def call_before_model_call_hook(
    spec: BeforeModelCallHookSpec[TContext],
    ctx: HookContext[TContext],
    request: PreparedModelRequest,
) -> ModelRequestPatch:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="before_model_call",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, request),
    )
    if result is None:
        return ModelRequestPatch()
    if not isinstance(result, ModelRequestPatch):
        raise HookExecutionError(
            f"hook {spec.name!r} returned {type(result).__name__}; "
            "before_model_call hooks must return ModelRequestPatch or None"
        )
    return result


async def call_model_request_committed_hook(
    spec: ModelRequestCommittedHookSpec[TContext],
    ctx: HookContext[TContext],
    request: PreparedModelRequest,
) -> None:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="on_model_request_committed",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, request),
    )
    if result is not None:
        raise HookExecutionError(
            f"hook {spec.name!r} returned a value; "
            "on_model_request_committed hooks must return None"
        )


async def call_before_tool_execute_hook(
    spec: BeforeToolExecuteHookSpec[TContext],
    ctx: HookContext[TContext],
    request: ToolExecutionRequest,
) -> ToolExecutionPatch:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="before_tool_execute",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, request),
    )
    if result is None:
        return ToolExecutionPatch()
    if not isinstance(result, ToolExecutionPatch):
        raise HookExecutionError(
            f"hook {spec.name!r} returned {type(result).__name__}; "
            "before_tool_execute hooks must return ToolExecutionPatch or None"
        )
    return result


async def call_after_tool_execute_hook(
    spec: AfterToolExecuteHookSpec[TContext],
    ctx: HookContext[TContext],
    outcome: ToolExecutionOutcome,
) -> None:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="after_tool_execute",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, outcome),
    )
    if result is not None:
        raise HookExecutionError(
            f"hook {spec.name!r} returned a value; "
            "after_tool_execute hooks must return None"
        )


async def call_after_model_response_hook(
    spec: AfterModelResponseHookSpec[TContext],
    ctx: HookContext[TContext],
    response: LLMResponse,
) -> None:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="after_model_response",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, response),
    )
    if result is not None:
        raise HookExecutionError(
            f"hook {spec.name!r} returned a value; "
            "after_model_response hooks must return None"
        )


async def call_model_error_hook(
    spec: ModelErrorHookSpec[TContext],
    ctx: HookContext[TContext],
    error: ModelError,
) -> None:
    result = await _await_hook(
        hook_name=spec.name,
        hook_point="on_model_error",
        timeout_seconds=spec.timeout_seconds,
        call=spec.fn(ctx, error),
    )
    if result is not None:
        raise HookExecutionError(
            f"hook {spec.name!r} returned a value; "
            "on_model_error hooks must return None"
        )


async def _await_hook(
    *,
    hook_name: str,
    hook_point: str,
    timeout_seconds: float | None,
    call: Awaitable[TResult],
) -> TResult:
    try:
        if timeout_seconds is None:
            return await call
        return await asyncio.wait_for(call, timeout=timeout_seconds)
    except HookExecutionError:
        raise
    # Hook functions are user code; wrap their failures with the hook point.
    except Exception as exc:
        raise HookExecutionError(
            f"hook {hook_name!r} failed at {hook_point}: {exc}"
        ) from exc
