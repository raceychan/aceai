from aceai.agent.events import ToolCompletedEvent
from aceai.llm.models import (
    LLMMessage,
    LLMResponse,
    LLMToolCallMessage,
    LLMToolUseMessage,
)


class StepContext:
    """context for each agent step, tool calls, LLM responses etc."""

    step_id: str
    step_index: int
    tool_call: LLMToolCallMessage
    tool_uses: list[LLMToolUseMessage]


"""
[StepContext, StepContext]
"""


class RunContext:
    """context for the entire agent run, all steps etc."""

    run_id: str
    prompt: str
    steps: list[StepContext]

    def __init__(self, run_id: str, prompt: str) -> None:
        self.run_id = run_id
        self.prompt = prompt
        self.steps = []

    def as_messages(self) -> list[LLMMessage]:
        """
        convert the run context to a list of LLM messages
        """
        raise NotImplementedError


"""
class MemorySlice:
    ...

class Context:
    memories: list[MemorySlice]

class Agent:
    def memorize(self, slice: MemorySlice) -> None:
        ...

    def retrieve(self, query: str) -> list[MemorySlice]:
        ...

"""


class ContextManager:
    """responsible for managing the context for the agent, dynamic context loading etc.

    also group messages as `context` for LLM input

    it is used mainly to
    1. assemble cross-run instruction for agent such as token budget
    2. summarize context if necessary
    3. render context as LLM messages

    this does not capsulate the context, but instead
    provide abilities for agent to manage its context in runtime
    """

    def __init__(self, prompt: str):
        self._instructions: list[str] = [prompt]
        self._seen_instructions: set[str] = {prompt}
        self._system_message: LLMMessage | None = None

    def add_instruction(self, instruction: str) -> None:
        if instruction not in self._seen_instructions:
            self._instructions.append(instruction)
            self._seen_instructions.add(instruction)
            self._system_message = None

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

    def init_context(self, messages: list[LLMMessage]) -> None:
        sys_msg = LLMMessage.build(
            role="system", content="".join(c for c in self._instructions if c)
        )
        self._context = [sys_msg] + messages

    def add_tool_call(self, tool_call_resp: LLMResponse) -> None:
        assistant_msg = LLMToolCallMessage.from_content(
            content=tool_call_resp.text,
            tool_calls=tool_call_resp.tool_calls,
        )
        self._context.append(assistant_msg)

    def add_tool_use(self, event: ToolCompletedEvent) -> None:
        call = event.tool_call
        tool_use_msg = LLMToolUseMessage.from_content(
            name=call.name,
            call_id=call.call_id,
            content=event.tool_result.output,
        )
        self._context.append(tool_use_msg)
