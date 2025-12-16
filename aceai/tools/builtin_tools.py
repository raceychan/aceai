from typing import Annotated as Annotated

from ._tool_sig import spec
from .base import tool


@tool
def final_answer(
    answer: Annotated[str, spec(description="Final response to emit")],
) -> str:
    """Tool to indicate the final answer from the agent."""
    return answer


BUILTIN_TOOLS = [final_answer]
