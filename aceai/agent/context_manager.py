from aceai.agent.events import ToolCompletedEvent
from aceai.llm.models import (
    LLMMessage,
    LLMResponse,
    LLMToolCallMessage,
    LLMToolUseMessage,
)


class ContextManager:
    """responsible for managing the context for the agent, dynamic context loading etc.

    also group messages as `context` for LLM input
    """

    def __init__(self, prompt: str):
        self._instructions: list[str] = [prompt]
        self._seen_instructions: set[str] = {prompt}
        self._system_message: LLMMessage | None = None
        self._context: list[LLMMessage]

    def add_instruction(self, instruction: str) -> None:
        if instruction not in self._seen_instructions:
            self._instructions.append(instruction)
            self._seen_instructions.add(instruction)

    def reset_context(self) -> None:
        self._context = []

    @property
    def system_message(self) -> LLMMessage:
        if self._system_message:
            return self._system_message
        content = "".join(c for c in self._instructions if c)
        self._system_message = LLMMessage.build(role="system", content=content)
        return self._system_message

    @property
    def context(self) -> list[LLMMessage]:
        return self._context

    def init_context(self, question: str) -> None:
        sys_msg = LLMMessage.build(
            role="system", content="".join(c for c in self._instructions if c)
        )
        self._context = [sys_msg, LLMMessage.build(role="user", content=question)]

    def add_tool_call(self, tool_call_resp: LLMResponse) -> None:
        assistant_msg = LLMToolCallMessage.from_content(
            content=tool_call_resp.text,
            tool_calls=tool_call_resp.tool_calls,
        )
        self._context.append(assistant_msg)

    def add_tool_use(self, event: ToolCompletedEvent) -> None:
        call = event.tool_call
        res = event.tool_result.output
        tool_use_msg = LLMToolUseMessage.from_content(
            name=call.name,
            call_id=call.call_id,
            content=res,
        )
        self._context.append(tool_use_msg)
