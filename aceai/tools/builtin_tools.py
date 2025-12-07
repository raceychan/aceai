from .base import tool


@tool
def final_answer(answer: str) -> str:
    """Tool to indicate the final answer from the agent."""
    return answer


BUILTIN_TOOLS = [final_answer]
