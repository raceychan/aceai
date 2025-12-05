from ididi import Graph

from aceai.llm.interface import LLMToolCall
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

    def _analyze_tool_deps(self):
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
        return tool.encode_return(result)
