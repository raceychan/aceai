from typing import Literal

ToolPermission = Literal["always", "ask"]
TOOL_PERMISSION_OPTIONS: tuple[ToolPermission, ...] = ("always", "ask")
