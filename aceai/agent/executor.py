from time import perf_counter
from typing import Any, Callable

from ididi import Graph
from msgspec import Struct, field
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from aceai.interface import is_present
from aceai.llm.models import LLMToolCall
from aceai.tracing import get_trace_ctx
from aceai.tools import BUILTIN_TOOLS, IToolSpec, Tool


class RunState(Struct, kw_only=True):
    tool_call_counts: dict[str, int] = field(default_factory=dict[str, int])


class IExecutor:
    def select_tools(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ) -> list[IToolSpec]:
        "select tools by names, good for dynamic tool selection"
        raise NotImplementedError

    async def execute_tool(
        self,
        tool_call: LLMToolCall,
        *,
        run_state: RunState,
    ) -> str:
        "execute a tool call and return the result as string"
        raise NotImplementedError


class ToolExecutor(IExecutor):
    def __init__(
        self,
        graph: Graph,
        tools: list[Tool[Any, Any]],
        tracer: trace.Tracer | None = None,
    ):
        self.graph = graph
        self.tools = {tool.name: tool for tool in (tools + BUILTIN_TOOLS)}
        self._all_tools: list[IToolSpec] = []
        self._tracer = tracer or trace.get_tracer("aceai.executor")

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

    async def resolve_tool(self, tool: Tool[Any, Any], /, **params: Any) -> Any:
        dep_params = {
            dname: await self.graph.aresolve(dep, **params)
            for dname, dep in tool.signature.dep_nodes.items()
        }
        result = tool(**params, **dep_params)
        return await result if tool.is_async else result

    async def execute_tool(
        self,
        tool_call: LLMToolCall,
        *,
        run_state: RunState,
    ) -> str:
        tool_name = tool_call.name
        param_json = tool_call.arguments
        tool = self.tools[tool_name]
        max_calls_per_run = tool.metadata.max_calls_per_run
        if is_present(max_calls_per_run):
            if max_calls_per_run < 1:
                raise ValueError(
                    f"Tool {tool_name!r} has invalid max_calls_per_run={max_calls_per_run}"
                )
            current_count = run_state.tool_call_counts.get(tool_name, 0)
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
            params = tool.decode_params(param_json)
            result = await self.resolve_tool(tool, **params)
            if is_present(max_calls_per_run):
                run_state.tool_call_counts[tool_name] = (
                    run_state.tool_call_counts.get(tool_name, 0) + 1
                )
            return tool.encode_return(result)


class ILogger:
    def info(self, msg: str, /, **kwargs: Any) -> None: ...

    def success(self, msg: str, /, **kwargs: Any) -> None: ...

    def exception(self, msg: str, /, **kwargs: Any) -> None: ...


type ITimer = Callable[[], float]


class LoggingToolExecutor(ToolExecutor):
    def __init__(
        self,
        graph: Graph,
        tools: list[Tool[Any, Any]],
        logger: ILogger,
        timer: ITimer = perf_counter,
        tracer: trace.Tracer | None = None,
    ) -> None:
        super().__init__(graph, tools, tracer=tracer)
        self.logger = logger
        self.timer = timer

    async def execute_tool(
        self,
        tool_call: LLMToolCall,
        *,
        run_state: RunState,
    ) -> str:
        call_id = tool_call.call_id
        self.logger.info(
            f"Tool {tool_call.name} starting (call_id={call_id}) with {tool_call.arguments}"
        )
        start = self.timer()
        try:
            result = await super().execute_tool(
                tool_call,
                run_state=run_state,
            )
        except Exception:
            duration = self.timer() - start
            self.logger.exception(
                f"Tool {tool_call.name} failed after {duration:.2f}s",
            )
            raise
        duration = self.timer() - start
        self.logger.success(
            f"Tool {tool_call.name} finished in {duration:.2f}s, result: {result}",
        )
        return result
