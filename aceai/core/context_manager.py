import re
from typing import Annotated, cast

from msgspec import Meta

from aceai.core.events import ToolCompletedEvent, ToolFailedEvent
from aceai.llm.models import (
    LLMMessage,
    LLMMessagePart,
    LLMResponse,
    LLMToolCallMessage,
    LLMToolUseMessage,
)
from aceai.llm.service import ILLMService


PercentageThreshold = Annotated[
    str,
    Meta(pattern=r"^(?:100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)%$"),
]
CompressThreshold = PercentageThreshold | int | float
PERCENTAGE_THRESHOLD_PATTERN = re.compile(r"^(?:100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)%$")

CONTEXT_SUMMARY_OPEN = "<aceai_context_summary>"
CONTEXT_SUMMARY_CLOSE = "</aceai_context_summary>"
DEFAULT_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_KEEP_RECENT_MESSAGES = 8


class ContextCompressionPolicy:
    threshold: CompressThreshold
    context_window_tokens: int
    keep_recent_messages: int

    def __init__(
        self,
        threshold: CompressThreshold = "100%",
        *,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES,
    ) -> None:
        self.threshold = threshold
        self.context_window_tokens = context_window_tokens
        self.keep_recent_messages = keep_recent_messages
        self.validate()

    def validate(self) -> None:
        threshold = self.threshold
        if type(self.context_window_tokens) is not int:
            raise TypeError("context_window_tokens must be int")
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be positive")
        if type(self.keep_recent_messages) is not int:
            raise TypeError("keep_recent_messages must be int")
        if self.keep_recent_messages < 1:
            raise ValueError("keep_recent_messages must be positive")
        if type(threshold) is str:
            if PERCENTAGE_THRESHOLD_PATTERN.fullmatch(threshold) is None:
                raise ValueError("compress_threshold must be a percentage from 0% to 100%")
            return
        if type(threshold) is int:
            if threshold < 1:
                raise ValueError("compress_threshold token count must be positive")
            return
        if type(threshold) is float:
            if threshold <= 0 or threshold > 1:
                raise ValueError("compress_threshold ratio must be > 0 and <= 1")
            return
        raise TypeError("compress_threshold must be str, int, or float")

    @property
    def threshold_tokens(self) -> int:
        threshold = self.threshold
        if type(threshold) is int:
            return threshold
        if type(threshold) is float:
            return max(1, int(self.context_window_tokens * threshold))
        percent_text = cast(str, threshold)
        percent = float(percent_text[:-1])
        return max(1, int(self.context_window_tokens * percent / 100))


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

    def __init__(
        self,
        prompt: str,
        *,
        compression_policy: ContextCompressionPolicy | None = None,
    ):
        self._instructions: list[str] = [prompt]
        self._seen_instructions: set[str] = {prompt}
        self._system_message: LLMMessage | None = None
        self._compression_policy = compression_policy or ContextCompressionPolicy()
        self._compression_count = 0

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
    def instructions_text(self) -> str:
        return "".join(c for c in self._instructions if c)

    @property
    def context(self) -> list[LLMMessage]:
        return self._context

    @property
    def compression_count(self) -> int:
        return self._compression_count

    def init_context(self, messages: list[LLMMessage]) -> None:
        sys_msg = LLMMessage.build(
            role="system", content="".join(c for c in self._instructions if c)
        )
        self._context = [sys_msg] + messages

    async def prepare_for_llm(
        self,
        *,
        llm_service: ILLMService,
    ) -> list[LLMMessage]:
        if self._should_compress():
            await self.compress(llm_service=llm_service)
        return self._context

    async def compress(
        self,
        *,
        llm_service: ILLMService,
    ) -> bool:
        policy = self._compression_policy
        if len(self._context) <= policy.keep_recent_messages + 1:
            return False
        system_message = self._context[0]
        old_messages = self._context[1 : -policy.keep_recent_messages]
        recent_messages = self._context[-policy.keep_recent_messages :]
        if not old_messages:
            return False
        response = await llm_service.complete(
            messages=[
                LLMMessage.build(
                    role="system",
                    content=(
                        "Compress earlier AceAI conversation context for a future "
                        "agent turn. Preserve user goals, constraints, decisions, "
                        "open tasks, file paths, tool outcomes, and unresolved errors. "
                        "Do not invent facts. Return concise prose only."
                    ),
                ),
                LLMMessage.build(
                    role="user",
                    content=_format_messages_for_summary(old_messages),
                ),
            ],
        )
        summary_message = LLMMessage.build(
            role="system",
            content=(
                f"{CONTEXT_SUMMARY_OPEN}\n"
                f"{response.text}\n"
                f"{CONTEXT_SUMMARY_CLOSE}"
            ),
        )
        self._context = [system_message, summary_message] + recent_messages
        self._compression_count += 1
        return True

    def add_tool_call(self, tool_call_resp: LLMResponse) -> None:
        assistant_msg = LLMToolCallMessage.from_content(
            content=[],
            tool_calls=tool_call_resp.tool_calls,
            reasoning_content=tool_call_resp.reasoning_content,
        )
        self._context.append(assistant_msg)

    def add_tool_use(self, event: ToolCompletedEvent | ToolFailedEvent) -> None:
        call = event.tool_call
        tool_use_msg = LLMToolUseMessage.from_content(
            name=call.name,
            call_id=call.call_id,
            content=event.tool_result.output,
        )
        self._context.append(tool_use_msg)

    def _should_compress(self) -> bool:
        return (
            estimate_message_tokens(self._context)
            >= self._compression_policy.threshold_tokens
        )


def estimate_message_tokens(messages: list[LLMMessage]) -> int:
    total = 0
    for message in messages:
        total += 4
        total += _estimate_text_tokens(message.role)
        for part in message.content:
            total += _estimate_part_tokens(part)
        if isinstance(message, LLMToolCallMessage):
            for tool_call in message.tool_calls:
                total += _estimate_text_tokens(tool_call.name)
                total += _estimate_text_tokens(tool_call.arguments)
                total += _estimate_text_tokens(tool_call.call_id)
        if isinstance(message, LLMToolUseMessage):
            total += _estimate_text_tokens(message.name)
            total += _estimate_text_tokens(message.call_id)
    return total


def _estimate_part_tokens(part: LLMMessagePart) -> int:
    tokens = 0
    part_type = part["type"]
    tokens += _estimate_text_tokens(part_type)
    if part_type == "text":
        if "data" not in part:
            raise ValueError("text message part must include data")
        tokens += _estimate_text_tokens(part["data"])
    elif "url" in part:
        tokens += _estimate_text_tokens(part["url"])
    elif "mime_type" in part:
        tokens += _estimate_text_tokens(part["mime_type"])
    return tokens


def _estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _format_messages_for_summary(messages: list[LLMMessage]) -> str:
    lines: list[str] = [
        "Earlier messages to compress:",
    ]
    for index, message in enumerate(messages, start=1):
        lines.append(f"<message index=\"{index}\" role=\"{message.role}\">")
        lines.extend(_message_summary_lines(message))
        lines.append("</message>")
    return "\n".join(lines)


def _message_summary_lines(message: LLMMessage) -> list[str]:
    lines: list[str] = []
    for part in message.content:
        if part["type"] == "text":
            if "data" not in part:
                raise ValueError("text message part must include data")
            lines.append(part["data"])
        elif "url" in part:
            lines.append(f"{part['type']} url: {part['url']}")
        elif "mime_type" in part:
            lines.append(f"{part['type']} mime_type: {part['mime_type']}")
        else:
            lines.append(f"{part['type']} content")
    if isinstance(message, LLMToolCallMessage):
        for tool_call in message.tool_calls:
            lines.append(
                f"tool_call name={tool_call.name} call_id={tool_call.call_id}"
            )
            lines.append(tool_call.arguments)
    if isinstance(message, LLMToolUseMessage):
        lines.append(f"tool_result name={message.name} call_id={message.call_id}")
    return lines
