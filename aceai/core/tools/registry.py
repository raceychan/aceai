"""
Load tools from py modules

"""

import importlib
from inspect import getmembers
from pathlib import Path
from typing import Any, cast

from .tool import Tool


class ToolTag:
    """
    registry = ToolRegistry()

    math_tag = registry.tag("math", '''
    all tools math related, e.g. arithmetic, algebra, calculus
    use this tag to group all math related tools
    ''')
    or
    math_tag = ToolTag("math")
    registry.register_tag(math_tag)


    @math_tag
    def add(a: int, b: int) -> int:
        return a + b

    @math_tag.tag("arithmetic")
    def multiply(a: int, b: int) -> int:
        return a * b
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._tools: dict[str, Tool[Any, Any]] = dict()


class ToolRegistry:
    """
    group tools by tag
    Example:

        @tool(tags=["math", "arithmetic"])
        def add(a: int, b: int) -> int:
            return a + b

        @tool(tags=["string"])
        def concat(s1: str, s2: str) -> str:
            return s1 + s2

        @tool(tags=["math", "string])
        def repeat_string(s: str, times: int) -> str:
            return s * times

        registry = ToolRegistry()


        registry.get_tools("math")  # [add]
        registry.get_tools("arithmetic")  # [add]
        registry.get_tools("string")  # [concat, repeat_string]

    """

    def __init__(self, *tools: Tool[Any, Any]) -> None:
        self._tools: dict[str, dict[str, Tool[Any, Any]]] = {}
        if tools:
            self.register_tools(*tools)

        self._modules: list[str] = []

    @property
    def tools(self) -> dict[str, dict[str, Tool[Any, Any]]]:
        return self._tools

    @property
    def modules(self) -> list[str]:
        return self._modules

    def register_tools(self, *tools: Tool[Any, Any]) -> None:
        for tool in tools:
            if tool.metadata.tags:
                for tag in tool.metadata.tags:
                    self._tools.setdefault(tag, {})[tool.name] = tool
            else:
                self._tools.setdefault("public", {})[tool.name] = tool

    def remove_tool(self, tool: Tool[Any, Any]) -> None:
        for tag in self._tools.keys():
            self._tools[tag].pop(tool.name, None)

    def get_tools(self, tag: str) -> list[Tool[Any, Any]]:
        return list(self._tools.get(tag, {}).values())

    def load_modules(self) -> None:
        for module_name in self._modules:
            module = importlib.import_module(module_name)
            tools = getmembers(module, predicate=lambda x: isinstance(x, Tool))
            for tool in tools:
                self.register_tools(cast(Tool[Any, Any], tool[1]))

    def register_module(self, module_path: str | Path, lazy: bool = False) -> None:
        if isinstance(module_path, str):
            if module_path.endswith(".py"):
                module_file = Path(module_path)
                if not module_file.exists():
                    raise FileNotFoundError(
                        f"Module path {module_file} does not exist."
                    )
                module_name = module_file.stem
            else:
                module_name = module_path
        else:
            if not module_path.exists():
                raise FileNotFoundError(f"Module path {module_path} does not exist.")
            module_name = module_path.stem

        self._modules.append(module_name)

        if not lazy:
            self.load_modules()
