from typing import TypedDict, Any, Literal

class ToolSpec(TypedDict):
    """Tool specification compatible with common LLM tool schemas (JSON Schema)."""
    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, Any]

