import inspect
from time import perf_counter
from typing import Any, Callable

from ididi import Graph
from opentelemetry.context import Context
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from aceai.llm.models import LLMToolCall
from aceai.tools import BUILTIN_TOOLS, IToolSpec, Tool


class ToolExecutor:
    def __init__(
        self,
        graph: Graph,
        tools: list[Tool[Any, Any]],
        tracer: trace.Tracer | None = None,
    ):
        self.graph = graph
        self.tools = {tool.name: tool for tool in (tools + BUILTIN_TOOLS)}
        self._analyze_tool_deps()
        self._tool_specs: list[IToolSpec] = []
        self._tracer = tracer or trace.get_tracer("aceai.executor")

    @property
    def tool_specs(self) -> list[IToolSpec]:
        """
        TODO:
        dynamic tool schema
        we might let planner take control over
        what tools are available to llm
        """
        if not self._tool_specs:
            self._tool_specs = [tool.tool_spec for tool in self.tools.values()]
        return self._tool_specs

    def _analyze_tool_deps(self) -> None:
        for tool in self.tools.values():
            self.graph.add_nodes(*tool.signature.dep_nodes.values())

    async def execute_tool(
        self, tool_call: LLMToolCall, *, trace_ctx: Context | None = None
    ) -> str:
        tool_name = tool_call.name
        param_json = tool_call.arguments
        tool = self.tools[tool_name]
        if trace_ctx is None:
            trace_ctx = Context()
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
            dep_params = {
                dname: await self.graph.aresolve(dep, **params)
                for dname, dep in tool.signature.dep_nodes.items()
            }
            result = tool(**params, **dep_params)
            if inspect.isawaitable(result):
                result = await result
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
        self, tool_call: LLMToolCall, *, trace_ctx: Context | None = None
    ) -> str:
        call_id = tool_call.call_id
        self.logger.info(
            f"Tool {tool_call.name} starting (call_id={call_id}) with {tool_call.arguments}"
        )
        start = self.timer()
        try:
            result = await super().execute_tool(tool_call, trace_ctx=trace_ctx)
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
