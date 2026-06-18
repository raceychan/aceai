import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar, overload

from aceai.llm.errors import AceAIConfigurationError, AceAIRuntimeError
from aceai.llm.models import (
    LLMMessage,
    LLMRequestMeta,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
)

from .models import ToolExecutionResult


TContext = TypeVar("TContext")
HookMode = Literal["preview", "execute"]
HookPoint = Literal[
    "before_model_request",
    "on_model_request_committed",
    "before_tool_execute",
    "after_tool_execute",
    "after_model_response",
    "on_model_error",
]


class HookRegistrationError(AceAIConfigurationError):
    """Raised when a hook cannot be registered into a hook plan."""


class HookExecutionError(AceAIRuntimeError):
    """Raised when a hook fails while a run is executing."""


def empty_request_meta() -> LLMRequestMeta:
    return {}


@dataclass(frozen=True)
class HookContext(Generic[TContext]):
    mode: HookMode
    run_id: str
    step_id: str
    step_index: int
    agent_id: str
    data: TContext | None = None


@dataclass(frozen=True)
class HookTraceSlot:
    slot: str
    source: str
    detail: str = ""
    event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class HookEffect:
    kind: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequestDraft:
    messages: tuple[LLMMessage, ...]
    tools: tuple[LLMToolSpec, ...] = ()
    metadata: LLMRequestMeta = field(default_factory=empty_request_meta)


@dataclass(frozen=True)
class ModelRequestPatch:
    prepend_messages: tuple[LLMMessage, ...] = ()
    append_messages: tuple[LLMMessage, ...] = ()
    tool_allowlist: frozenset[str] | None = None
    tool_denylist: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, object] = field(default_factory=dict)
    trace: tuple[HookTraceSlot, ...] = ()
    effects: tuple[HookEffect, ...] = ()


@dataclass(frozen=True)
class AppliedHookPatch:
    hook_name: str
    point: HookPoint
    prepended_message_count: int = 0
    appended_message_count: int = 0
    metadata_keys: tuple[str, ...] = ()
    tool_allowlist: frozenset[str] | None = None
    tool_denylist: frozenset[str] = field(default_factory=frozenset)
    trace: tuple[HookTraceSlot, ...] = ()
    effects: tuple[HookEffect, ...] = ()


@dataclass(frozen=True)
class PreparedModelRequest:
    request_id: str
    attempt_id: str
    run_id: str
    step_id: str
    step_index: int
    messages: tuple[LLMMessage, ...]
    tools: tuple[LLMToolSpec, ...]
    metadata: LLMRequestMeta
    patches: tuple[AppliedHookPatch, ...] = ()


@dataclass(frozen=True)
class ToolExecutionRequest:
    call: LLMToolCall
    tool_name: str
    approval_required: bool
    approved_tool_names: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolExecutionPatch:
    approved_tool_names: frozenset[str] | None = None


@dataclass(frozen=True)
class ToolExecutionOutcome:
    call: LLMToolCall
    tool_name: str
    result: ToolExecutionResult


@dataclass(frozen=True)
class ModelError:
    error: BaseException
    request: PreparedModelRequest | None = None
    request_committed: bool = False


class BeforeModelRequestHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        draft: ModelRequestDraft,
    ) -> ModelRequestPatch | None: ...


class ModelRequestCommittedHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        request: PreparedModelRequest,
    ) -> None: ...


class BeforeToolExecuteHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        request: ToolExecutionRequest,
    ) -> ToolExecutionPatch | None: ...


class AfterToolExecuteHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        outcome: ToolExecutionOutcome,
    ) -> None: ...


class AfterModelResponseHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        response: LLMResponse,
    ) -> None: ...


class ModelErrorHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        error: ModelError,
    ) -> None: ...


@dataclass(frozen=True)
class BeforeModelRequestHookSpec(Generic[TContext]):
    name: str
    order: int
    registration_index: int
    fn: BeforeModelRequestHook[TContext]
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ModelRequestCommittedHookSpec(Generic[TContext]):
    name: str
    order: int
    registration_index: int
    fn: ModelRequestCommittedHook[TContext]
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class BeforeToolExecuteHookSpec(Generic[TContext]):
    name: str
    order: int
    registration_index: int
    fn: BeforeToolExecuteHook[TContext]
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class AfterToolExecuteHookSpec(Generic[TContext]):
    name: str
    order: int
    registration_index: int
    fn: AfterToolExecuteHook[TContext]
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class AfterModelResponseHookSpec(Generic[TContext]):
    name: str
    order: int
    registration_index: int
    fn: AfterModelResponseHook[TContext]
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ModelErrorHookSpec(Generic[TContext]):
    name: str
    order: int
    registration_index: int
    fn: ModelErrorHook[TContext]
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class HookPlan(Generic[TContext]):
    before_model_request: tuple[BeforeModelRequestHookSpec[TContext], ...] = ()
    on_model_request_committed: tuple[
        ModelRequestCommittedHookSpec[TContext], ...
    ] = ()
    before_tool_execute: tuple[BeforeToolExecuteHookSpec[TContext], ...] = ()
    after_tool_execute: tuple[AfterToolExecuteHookSpec[TContext], ...] = ()
    after_model_response: tuple[AfterModelResponseHookSpec[TContext], ...] = ()
    on_model_error: tuple[ModelErrorHookSpec[TContext], ...] = ()

    @property
    def is_empty(self) -> bool:
        return (
            not self.before_model_request
            and not self.on_model_request_committed
            and not self.before_tool_execute
            and not self.after_tool_execute
            and not self.after_model_response
            and not self.on_model_error
        )

    @classmethod
    def empty(cls) -> "HookPlan[TContext]":
        return cls()

    def combine(self, other: "HookPlan[TContext]") -> "HookPlan[TContext]":
        return build_hook_plan(
            before_model_request=(
                *self.before_model_request,
                *other.before_model_request,
            ),
            on_model_request_committed=(
                *self.on_model_request_committed,
                *other.on_model_request_committed,
            ),
            before_tool_execute=(
                *self.before_tool_execute,
                *other.before_tool_execute,
            ),
            after_tool_execute=(
                *self.after_tool_execute,
                *other.after_tool_execute,
            ),
            after_model_response=(
                *self.after_model_response,
                *other.after_model_response,
            ),
            on_model_error=(
                *self.on_model_error,
                *other.on_model_error,
            ),
        )


class HookRegistry(Generic[TContext]):
    def __init__(self) -> None:
        self._before_model_request: list[BeforeModelRequestHookSpec[TContext]] = []
        self._on_model_request_committed: list[
            ModelRequestCommittedHookSpec[TContext]
        ] = []
        self._before_tool_execute: list[BeforeToolExecuteHookSpec[TContext]] = []
        self._after_tool_execute: list[AfterToolExecuteHookSpec[TContext]] = []
        self._after_model_response: list[AfterModelResponseHookSpec[TContext]] = []
        self._on_model_error: list[ModelErrorHookSpec[TContext]] = []
        self._next_registration_index = 0

    @overload
    def before_model_request(
        self,
        fn: BeforeModelRequestHook[TContext],
        /,
    ) -> BeforeModelRequestHook[TContext]: ...

    @overload
    def before_model_request(
        self,
        fn: None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> "BeforeModelRequestDecorator[TContext]": ...

    def before_model_request(
        self,
        fn: BeforeModelRequestHook[TContext] | None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> BeforeModelRequestHook[TContext] | "BeforeModelRequestDecorator[TContext]":
        if fn is not None:
            return self.add_before_model_request(
                fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        def register(
            hook_fn: BeforeModelRequestHook[TContext],
        ) -> BeforeModelRequestHook[TContext]:
            return self.add_before_model_request(
                hook_fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        return register

    def add_before_model_request(
        self,
        fn: BeforeModelRequestHook[TContext],
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> BeforeModelRequestHook[TContext]:
        _validate_hook_function(fn)
        _validate_timeout(timeout_seconds)
        self._before_model_request.append(
            BeforeModelRequestHookSpec(
                name=name or _hook_name(fn),
                order=order,
                registration_index=self._next_registration_index,
                fn=fn,
                timeout_seconds=timeout_seconds,
            )
        )
        self._next_registration_index += 1
        return fn

    @overload
    def on_model_request_committed(
        self,
        fn: ModelRequestCommittedHook[TContext],
        /,
    ) -> ModelRequestCommittedHook[TContext]: ...

    @overload
    def on_model_request_committed(
        self,
        fn: None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> "ModelRequestCommittedDecorator[TContext]": ...

    def on_model_request_committed(
        self,
        fn: ModelRequestCommittedHook[TContext] | None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> (
        ModelRequestCommittedHook[TContext]
        | "ModelRequestCommittedDecorator[TContext]"
    ):
        if fn is not None:
            return self.add_model_request_committed(
                fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        def register(
            hook_fn: ModelRequestCommittedHook[TContext],
        ) -> ModelRequestCommittedHook[TContext]:
            return self.add_model_request_committed(
                hook_fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        return register

    def add_model_request_committed(
        self,
        fn: ModelRequestCommittedHook[TContext],
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> ModelRequestCommittedHook[TContext]:
        _validate_hook_function(fn)
        _validate_timeout(timeout_seconds)
        self._on_model_request_committed.append(
            ModelRequestCommittedHookSpec(
                name=name or _hook_name(fn),
                order=order,
                registration_index=self._next_registration_index,
                fn=fn,
                timeout_seconds=timeout_seconds,
            )
        )
        self._next_registration_index += 1
        return fn

    @overload
    def before_tool_execute(
        self,
        fn: BeforeToolExecuteHook[TContext],
        /,
    ) -> BeforeToolExecuteHook[TContext]: ...

    @overload
    def before_tool_execute(
        self,
        fn: None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> "BeforeToolExecuteDecorator[TContext]": ...

    def before_tool_execute(
        self,
        fn: BeforeToolExecuteHook[TContext] | None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> BeforeToolExecuteHook[TContext] | "BeforeToolExecuteDecorator[TContext]":
        if fn is not None:
            return self.add_before_tool_execute(
                fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        def register(
            hook_fn: BeforeToolExecuteHook[TContext],
        ) -> BeforeToolExecuteHook[TContext]:
            return self.add_before_tool_execute(
                hook_fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        return register

    def add_before_tool_execute(
        self,
        fn: BeforeToolExecuteHook[TContext],
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> BeforeToolExecuteHook[TContext]:
        _validate_hook_function(fn)
        _validate_timeout(timeout_seconds)
        self._before_tool_execute.append(
            BeforeToolExecuteHookSpec(
                name=name or _hook_name(fn),
                order=order,
                registration_index=self._next_registration_index,
                fn=fn,
                timeout_seconds=timeout_seconds,
            )
        )
        self._next_registration_index += 1
        return fn

    @overload
    def after_tool_execute(
        self,
        fn: AfterToolExecuteHook[TContext],
        /,
    ) -> AfterToolExecuteHook[TContext]: ...

    @overload
    def after_tool_execute(
        self,
        fn: None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> "AfterToolExecuteDecorator[TContext]": ...

    def after_tool_execute(
        self,
        fn: AfterToolExecuteHook[TContext] | None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> AfterToolExecuteHook[TContext] | "AfterToolExecuteDecorator[TContext]":
        if fn is not None:
            return self.add_after_tool_execute(
                fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        def register(
            hook_fn: AfterToolExecuteHook[TContext],
        ) -> AfterToolExecuteHook[TContext]:
            return self.add_after_tool_execute(
                hook_fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        return register

    def add_after_tool_execute(
        self,
        fn: AfterToolExecuteHook[TContext],
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> AfterToolExecuteHook[TContext]:
        _validate_hook_function(fn)
        _validate_timeout(timeout_seconds)
        self._after_tool_execute.append(
            AfterToolExecuteHookSpec(
                name=name or _hook_name(fn),
                order=order,
                registration_index=self._next_registration_index,
                fn=fn,
                timeout_seconds=timeout_seconds,
            )
        )
        self._next_registration_index += 1
        return fn

    @overload
    def after_model_response(
        self,
        fn: AfterModelResponseHook[TContext],
        /,
    ) -> AfterModelResponseHook[TContext]: ...

    @overload
    def after_model_response(
        self,
        fn: None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> "AfterModelResponseDecorator[TContext]": ...

    def after_model_response(
        self,
        fn: AfterModelResponseHook[TContext] | None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> AfterModelResponseHook[TContext] | "AfterModelResponseDecorator[TContext]":
        if fn is not None:
            return self.add_after_model_response(
                fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        def register(
            hook_fn: AfterModelResponseHook[TContext],
        ) -> AfterModelResponseHook[TContext]:
            return self.add_after_model_response(
                hook_fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        return register

    def add_after_model_response(
        self,
        fn: AfterModelResponseHook[TContext],
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> AfterModelResponseHook[TContext]:
        _validate_hook_function(fn)
        _validate_timeout(timeout_seconds)
        self._after_model_response.append(
            AfterModelResponseHookSpec(
                name=name or _hook_name(fn),
                order=order,
                registration_index=self._next_registration_index,
                fn=fn,
                timeout_seconds=timeout_seconds,
            )
        )
        self._next_registration_index += 1
        return fn

    @overload
    def on_model_error(
        self,
        fn: ModelErrorHook[TContext],
        /,
    ) -> ModelErrorHook[TContext]: ...

    @overload
    def on_model_error(
        self,
        fn: None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> "ModelErrorDecorator[TContext]": ...

    def on_model_error(
        self,
        fn: ModelErrorHook[TContext] | None = None,
        /,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> ModelErrorHook[TContext] | "ModelErrorDecorator[TContext]":
        if fn is not None:
            return self.add_model_error(
                fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        def register(
            hook_fn: ModelErrorHook[TContext],
        ) -> ModelErrorHook[TContext]:
            return self.add_model_error(
                hook_fn,
                name=name,
                order=order,
                timeout_seconds=timeout_seconds,
            )

        return register

    def add_model_error(
        self,
        fn: ModelErrorHook[TContext],
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> ModelErrorHook[TContext]:
        _validate_hook_function(fn)
        _validate_timeout(timeout_seconds)
        self._on_model_error.append(
            ModelErrorHookSpec(
                name=name or _hook_name(fn),
                order=order,
                registration_index=self._next_registration_index,
                fn=fn,
                timeout_seconds=timeout_seconds,
            )
        )
        self._next_registration_index += 1
        return fn

    def build_plan(self) -> HookPlan[TContext]:
        return build_hook_plan(
            before_model_request=tuple(self._before_model_request),
            on_model_request_committed=tuple(self._on_model_request_committed),
            before_tool_execute=tuple(self._before_tool_execute),
            after_tool_execute=tuple(self._after_tool_execute),
            after_model_response=tuple(self._after_model_response),
            on_model_error=tuple(self._on_model_error),
        )


class BeforeModelRequestDecorator(Protocol[TContext]):
    def __call__(
        self,
        hook_fn: BeforeModelRequestHook[TContext],
    ) -> BeforeModelRequestHook[TContext]: ...


class ModelRequestCommittedDecorator(Protocol[TContext]):
    def __call__(
        self,
        hook_fn: ModelRequestCommittedHook[TContext],
    ) -> ModelRequestCommittedHook[TContext]: ...


class BeforeToolExecuteDecorator(Protocol[TContext]):
    def __call__(
        self,
        hook_fn: BeforeToolExecuteHook[TContext],
    ) -> BeforeToolExecuteHook[TContext]: ...


class AfterToolExecuteDecorator(Protocol[TContext]):
    def __call__(
        self,
        hook_fn: AfterToolExecuteHook[TContext],
    ) -> AfterToolExecuteHook[TContext]: ...


class AfterModelResponseDecorator(Protocol[TContext]):
    def __call__(
        self,
        hook_fn: AfterModelResponseHook[TContext],
    ) -> AfterModelResponseHook[TContext]: ...


class ModelErrorDecorator(Protocol[TContext]):
    def __call__(
        self,
        hook_fn: ModelErrorHook[TContext],
    ) -> ModelErrorHook[TContext]: ...


def build_hook_plan(
    *,
    before_model_request: tuple[BeforeModelRequestHookSpec[TContext], ...],
    on_model_request_committed: tuple[
        ModelRequestCommittedHookSpec[TContext], ...
    ],
    before_tool_execute: tuple[BeforeToolExecuteHookSpec[TContext], ...],
    after_tool_execute: tuple[AfterToolExecuteHookSpec[TContext], ...],
    after_model_response: tuple[AfterModelResponseHookSpec[TContext], ...],
    on_model_error: tuple[ModelErrorHookSpec[TContext], ...],
) -> HookPlan[TContext]:
    duplicates = _duplicated_hook_names(
        before_model_request=before_model_request,
        on_model_request_committed=on_model_request_committed,
        before_tool_execute=before_tool_execute,
        after_tool_execute=after_tool_execute,
        after_model_response=after_model_response,
        on_model_error=on_model_error,
    )
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise HookRegistrationError(f"duplicate hook name(s): {names}")
    return HookPlan(
        before_model_request=tuple(
            sorted(
                before_model_request,
                key=lambda spec: (spec.order, spec.registration_index),
            )
        ),
        on_model_request_committed=tuple(
            sorted(
                on_model_request_committed,
                key=lambda spec: (spec.order, spec.registration_index),
            )
        ),
        before_tool_execute=tuple(
            sorted(
                before_tool_execute,
                key=lambda spec: (spec.order, spec.registration_index),
            )
        ),
        after_tool_execute=tuple(
            sorted(
                after_tool_execute,
                key=lambda spec: (spec.order, spec.registration_index),
            )
        ),
        after_model_response=tuple(
            sorted(
                after_model_response,
                key=lambda spec: (spec.order, spec.registration_index),
            )
        ),
        on_model_error=tuple(
            sorted(
                on_model_error,
                key=lambda spec: (spec.order, spec.registration_index),
            )
        ),
    )


def _duplicated_hook_names(
    *,
    before_model_request: tuple[BeforeModelRequestHookSpec[TContext], ...],
    on_model_request_committed: tuple[
        ModelRequestCommittedHookSpec[TContext], ...
    ],
    before_tool_execute: tuple[BeforeToolExecuteHookSpec[TContext], ...],
    after_tool_execute: tuple[AfterToolExecuteHookSpec[TContext], ...],
    after_model_response: tuple[AfterModelResponseHookSpec[TContext], ...],
    on_model_error: tuple[ModelErrorHookSpec[TContext], ...],
) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in (
        *before_model_request,
        *on_model_request_committed,
        *before_tool_execute,
        *after_tool_execute,
        *after_model_response,
        *on_model_error,
    ):
        if spec.name in seen:
            duplicates.add(spec.name)
        seen.add(spec.name)
    return duplicates


def _validate_hook_function(fn: object) -> None:
    if inspect.iscoroutinefunction(fn):
        return
    name = _hook_name(fn)
    raise HookRegistrationError(f"hook {name!r} must be an async callable")


def _validate_timeout(timeout_seconds: float | None) -> None:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise HookRegistrationError("hook timeout_seconds must be positive")


def _hook_name(fn: object) -> str:
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        module = fn.__module__
        qualname = fn.__qualname__
    else:
        hook_type = type(fn)
        module = hook_type.__module__
        qualname = hook_type.__qualname__
    return f"{module}.{qualname}" if module else str(qualname)
