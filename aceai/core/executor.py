from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

from ididi import Graph
from msgspec import DecodeError, ValidationError
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from aceai.llm.errors import AceAIError
from aceai.llm.interface import UNSET, Unset, is_present, is_set
from aceai.llm.models import LLMHostedToolSpec, LLMToolCall
from aceai.llm.tracing import get_trace_ctx
from aceai.core.tools import IToolSpec, Tool
from aceai.core.run_state import ToolInvocation, ToolRunState
from aceai.core.skills import SkillLoader, SkillRegistry, format_skills_for_prompt


class ToolExecutionError(AceAIError):
    """Tool failure that should be returned to the model as tool output."""


class IExecutor:
    @property
    def prompt_instructions(self) -> str:
        "instructions that describe executor-provided capabilities"
        raise NotImplementedError

    @property
    def skill_registry(self) -> SkillRegistry:
        "skills available through this executor"
        raise NotImplementedError

    @property
    def hosted_tools(self) -> list[LLMHostedToolSpec]:
        "provider-hosted tools exposed by this executor"
        raise NotImplementedError

    def select_tools(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ) -> list[IToolSpec]:
        "select tools by names, good for dynamic tool selection"
        raise NotImplementedError

    def resolve_invocation(self, tool_call: LLMToolCall) -> ToolInvocation:
        "resolve a model-emitted tool call to a concrete local tool invocation"
        raise NotImplementedError

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        "execute a resolved tool invocation and return the result as string"
        raise NotImplementedError


class DummyExecutor(IExecutor):
    def __init__(self) -> None:
        self._skill_registry = SkillRegistry()
        self._hosted_tools: list[LLMHostedToolSpec] = []

    @property
    def prompt_instructions(self) -> str:
        return ""

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def hosted_tools(self) -> list[LLMHostedToolSpec]:
        return self._hosted_tools

    def select_tools(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ) -> list[IToolSpec]:
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        return []

    def resolve_invocation(self, tool_call: LLMToolCall) -> ToolInvocation:
        raise KeyError(tool_call.name)

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        raise KeyError(invocation.call.name)


class Executor(IExecutor):
    def __init__(
        self,
        graph: Graph,
        tools: list[Tool[Any, Any]],
        tracer: trace.Tracer | None = None,
        skill_path: str | Path | Literal["auto", "disable"] = "disable",
        enabled_skill_names: Unset[tuple[str, ...]] = UNSET,
        skill_loader_factory: Callable[[str], SkillLoader] = SkillLoader,
        extra_skill_paths: tuple[Path, ...] = (),
        hosted_tools: list[LLMHostedToolSpec] | None = None,
    ):
        self.graph = graph
        self.tools = {tool.name: tool for tool in tools}
        self._all_tools: list[IToolSpec] = []
        self._tracer = tracer or trace.get_tracer("aceai.executor")
        self._hosted_tools = hosted_tools if hosted_tools is not None else []
        self._skill_registry = SkillLoader.load_registry(
            skill_path,
            loader_factory=skill_loader_factory,
            extra_skill_paths=extra_skill_paths,
        )
        if is_set(enabled_skill_names):
            self._skill_registry = self._skill_registry.select(enabled_skill_names)
        if self._skill_registry.get_skills():
            self.register_tools(*self._skill_registry.as_tools())

    @property
    def prompt_instructions(self) -> str:
        return format_skills_for_prompt(self._skill_registry)

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def hosted_tools(self) -> list[LLMHostedToolSpec]:
        return self._hosted_tools

    def register_tools(self, *tools: Tool[Any, Any]) -> None:
        for tool in tools:
            self.tools[tool.name] = tool
        self._all_tools = []

    def select_tools(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ) -> list[IToolSpec]:
        "select tools by names, good for dynamic tool selection"

        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        if not include and not exclude:
            return self.all_tools

        include = include or set()
        exclude = exclude or set()

        selected_tools: list[IToolSpec] = []
        for tool_name, tool in self.tools.items():
            if (include and tool_name not in include) or (tool_name in exclude):
                continue
            selected_tools.append(tool.tool_spec)
        return selected_tools

    @property
    def all_tools(self) -> list[IToolSpec]:
        if not self._all_tools:
            self._all_tools = [tool.tool_spec for tool in self.tools.values()]
        return self._all_tools

    def resolve_invocation(self, tool_call: LLMToolCall) -> ToolInvocation:
        return ToolInvocation(call=tool_call, tool=self.tools[tool_call.name])

    async def resolve_tool_deps(self, tool: Tool[Any, Any], /, **params: Any) -> Any:
        dep_params = {
            dname: await self.graph.aresolve(dep, **params)
            for dname, dep in tool.signature.dep_nodes.items()
        }
        result = tool(**params, **dep_params)
        return await result if tool.is_async else result

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        tool_call = invocation.call
        tool = invocation.tool
        tool_name = tool.name
        param_json = tool_call.arguments
        max_calls_per_run = tool.metadata.max_calls_per_run
        if is_present(max_calls_per_run):
            if max_calls_per_run < 1:
                raise ValueError(
                    f"Tool {tool_name!r} has invalid max_calls_per_run={max_calls_per_run}"
                )
            current_count = tool_state.call_counts.get(tool_name, 0)
            if current_count >= max_calls_per_run:
                return (
                    f"the tool {tool_name} exceeds its max calls in this run, "
                    "do not call it again"
                )
        trace_ctx = get_trace_ctx()
        with self._tracer.start_as_current_span(
            f"tool.{tool_name}",
            kind=SpanKind.INTERNAL,
            record_exception=True,
            set_status_on_exception=True,
            context=trace_ctx,
            attributes={
                "tool.call_id": tool_call.call_id,
                "tool.arguments": param_json,
                "tool.dep_count": len(tool.signature.dep_nodes),
            },
        ):
            try:
                params = tool.decode_params(param_json)
            except (DecodeError, ValidationError) as exc:
                raise ToolExecutionError(
                    f"Invalid arguments for tool {tool_name}: {exc}"
                ) from exc
            result = await self.resolve_tool_deps(tool, **params)
            if is_present(max_calls_per_run):
                tool_state.call_counts[tool_name] = (
                    tool_state.call_counts.get(tool_name, 0) + 1
                )
            return tool.encode_return(result)


class ILogger:
    def info(self, msg: str, /, **kwargs: Any) -> None: ...

    def success(self, msg: str, /, **kwargs: Any) -> None: ...

    def exception(self, msg: str, /, **kwargs: Any) -> None: ...


type ITimer = Callable[[], float]


class LoggingExecutor(Executor):
    def __init__(
        self,
        graph: Graph,
        tools: list[Tool[Any, Any]],
        logger: ILogger,
        timer: ITimer = perf_counter,
        tracer: trace.Tracer | None = None,
        skill_path: str | Path | Literal["auto", "disable"] = "disable",
        enabled_skill_names: Unset[tuple[str, ...]] = UNSET,
        skill_loader_factory: Callable[[str], SkillLoader] = SkillLoader,
        extra_skill_paths: tuple[Path, ...] = (),
        hosted_tools: list[LLMHostedToolSpec] | None = None,
    ) -> None:
        super().__init__(
            graph,
            tools,
            tracer=tracer,
            skill_path=skill_path,
            enabled_skill_names=enabled_skill_names,
            skill_loader_factory=skill_loader_factory,
            extra_skill_paths=extra_skill_paths,
            hosted_tools=hosted_tools,
        )
        self.logger = logger
        self.timer = timer

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        call_id = invocation.call.call_id
        self.logger.info(
            f"Tool {invocation.tool.name} starting (call_id={call_id}) with {invocation.call.arguments}"
        )
        start = self.timer()
        try:
            result = await super().execute(
                invocation,
                tool_state=tool_state,
            )
        except Exception:
            duration = self.timer() - start
            self.logger.exception(
                f"Tool {invocation.tool.name} failed after {duration:.2f}s",
            )
            raise
        duration = self.timer() - start
        self.logger.success(
            f"Tool {invocation.tool.name} finished in {duration:.2f}s, result: {result}",
        )
        return result
