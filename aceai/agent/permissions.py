from typing import Literal

ToolPermission = Literal["always", "ask", "never"]
TOOL_PERMISSION_OPTIONS: tuple[ToolPermission, ...] = ("always", "ask", "never")
