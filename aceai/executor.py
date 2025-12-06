import inspect
from time import perf_counter
from typing import Any, Callable

from ididi import Graph

from aceai.llm.models import LLMToolCall
from aceai.tools import Tool, tool


@tool
def final_answer(answer: str) -> str:
    """Tool to indicate the final answer from the agent."""
    return answer


BUILTIN_TOOLS = [final_answer]


class ToolExecutor:
    def __init__(self, graph: Graph, tools: list[Tool]):
        self.graph = graph
        self.tools = {tool.name: tool for tool in (tools + BUILTIN_TOOLS)}
        self._analyze_tool_deps()
        self.tool_schemas = [tool.tool_schema for tool in tools]

    def _analyze_tool_deps(self) -> None:
        for tool in self.tools.values():
            self.graph.add_nodes(*tool.signature.dep_nodes.values())

    async def execute_tool(self, tool_call: LLMToolCall) -> str:
        tool_name = tool_call.name
        param_json = tool_call.arguments
        tool = self.tools[tool_name]
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
        tools: list[Tool],
        logger: ILogger,
        timer: ITimer = perf_counter,
    ) -> None:
        super().__init__(graph, tools)
        self.logger = logger
        self.timer = timer

    async def execute_tool(self, tool_call: LLMToolCall) -> str:
        call_id = tool_call.call_id
        self.logger.info(
            f"Tool {tool_call.name} starting (call_id={call_id}) with {tool_call.arguments}"
        )
        start = self.timer()
        try:
            result = await super().execute_tool(tool_call)
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
