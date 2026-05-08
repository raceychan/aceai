import json
import re
from typing import Annotated, Literal, cast

from msgspec import Meta, Struct, field

from aceai.core.events import ToolCompletedEvent, ToolFailedEvent
from aceai.llm.errors import AceAIRuntimeError, LLMProviderError
from aceai.llm.models import (
    LLMMessage,
    LLMMessagePart,
    LLMResponse,
    LLMHostedToolSpec,
    LLMToolCallMessage,
    LLMToolSpec,
    LLMToolUseMessage,
)
from aceai.llm.service import ILLMService


PercentageThreshold = Annotated[
    str,
    Meta(pattern=r"^(?:100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)%$"),
]
CompressThreshold = PercentageThreshold | int | float
PERCENTAGE_THRESHOLD_PATTERN = re.compile(r"^(?:100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)%$")

ContextSummaryScope = Literal["prior_runs", "current_run"]

CONTEXT_SUMMARY_OPEN = "<aceai_context_summary"
CONTEXT_SUMMARY_CLOSE = "</aceai_context_summary>"
DEFAULT_CONTEXT_WINDOW_TOKENS = 128000
DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 4096


class ContextCompressionPolicy:
    threshold: CompressThreshold
    context_window_tokens: int
    recent_step_budget: int

    def __init__(
        self,
        threshold: CompressThreshold = "100%",
        *,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        recent_step_budget: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.context_window_tokens = context_window_tokens
        if recent_step_budget is None:
            recent_step_budget = max(1, context_window_tokens // 3)
        self.recent_step_budget = recent_step_budget
        self.validate()

    def validate(self) -> None:
        threshold = self.threshold
        if type(self.context_window_tokens) is not int:
            raise TypeError("context_window_tokens must be int")
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be positive")
        if type(self.recent_step_budget) is not int:
            raise TypeError("recent_step_budget must be int")
        if self.recent_step_budget < 1:
            raise ValueError("recent_step_budget must be positive")
        if type(threshold) is str:
            if PERCENTAGE_THRESHOLD_PATTERN.fullmatch(threshold) is None:
                raise ValueError(
                    "compress_threshold must be a percentage from 0% to 100%"
                )
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

    @property
    def safety_margin_tokens(self) -> int:
        return min(
            DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
            max(1, self.context_window_tokens // 10),
        )

    @property
    def preflight_threshold_tokens(self) -> int:
        return max(
            1,
            min(
                self.threshold_tokens,
                self.context_window_tokens - self.safety_margin_tokens,
            ),
        )

    @property
    def summary_request_budget_tokens(self) -> int:
        return self.context_window_tokens


class PriorRunSummary(Struct, kw_only=True):
    content: str


class CurrentRunSummary(Struct, kw_only=True):
    content: str


class StepUnit(Struct, kw_only=True):
    messages: list[LLMMessage] = field(default_factory=list[LLMMessage])


class OpenStepUnit(Struct, kw_only=True):
    messages: list[LLMMessage] = field(default_factory=list[LLMMessage])


class RunUnit(Struct, kw_only=True):
    user_message: LLMMessage
    steps: list[StepUnit] = field(default_factory=list[StepUnit])
    open_step: OpenStepUnit | None = None


class StructuredContext(Struct, kw_only=True):
    system_message: LLMMessage
    prior_run_summary: PriorRunSummary | None = None
    current_run_summary: CurrentRunSummary | None = None
    prior_runs: list[RunUnit] = field(default_factory=list[RunUnit])
    current_run: RunUnit | None = None


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
        tools: list[LLMToolSpec] | None = None,
    ) -> list[LLMMessage]:
        parse_context_units(self._context)
        if self._should_compress(tools=tools or []):
            compressed = await self.compress(llm_service=llm_service)
            if not compressed and self._exceeds_preflight_budget(tools=tools or []):
                raise LLMProviderError(_no_compressible_context_message())
        return self._context

    def needs_compression(self, *, tools: list[LLMToolSpec] | None = None) -> bool:
        parse_context_units(self._context)
        return self._should_compress(tools=tools or [])

    def has_compressible_context(
        self,
        *,
        force_current_run_steps: bool = False,
    ) -> bool:
        structured = parse_context_units(self._context)
        return _has_compressible_context(
            structured,
            policy=self._compression_policy,
            force_current_run_steps=force_current_run_steps,
        )

    async def compress(
        self,
        *,
        llm_service: ILLMService,
        force_current_run_steps: bool = False,
    ) -> bool:
        structured = parse_context_units(self._context)
        compacted = await _compact_structured_context(
            structured,
            llm_service=llm_service,
            policy=self._compression_policy,
            force_current_run_steps=force_current_run_steps,
        )
        if compacted is None:
            return False
        self._context = render_context_units(compacted)
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
            content=event.tool_result.model_output or event.tool_result.output,
        )
        self._context.append(tool_use_msg)

    def _should_compress(self, *, tools: list[LLMToolSpec]) -> bool:
        return (
            estimate_message_tokens(self._context) + estimate_tool_tokens(tools)
            >= self._compression_policy.preflight_threshold_tokens
        )

    def _exceeds_preflight_budget(self, *, tools: list[LLMToolSpec]) -> bool:
        return (
            estimate_message_tokens(self._context) + estimate_tool_tokens(tools)
            >= self._compression_policy.context_window_tokens
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


def estimate_tool_tokens(tools: list[LLMToolSpec]) -> int:
    total = 0
    for tool in tools:
        if isinstance(tool, LLMHostedToolSpec):
            total += _estimate_text_tokens(
                json.dumps(tool.asdict(), ensure_ascii=False)
            )
        else:
            total += _estimate_text_tokens(
                json.dumps(tool.generate_schema(), ensure_ascii=False)
            )
    return total


def parse_context_units(messages: list[LLMMessage]) -> StructuredContext:
    if not messages:
        raise AceAIRuntimeError("context requires a system message")
    system_message = messages[0]
    if system_message.role != "system":
        raise AceAIRuntimeError("context requires a system message")

    prior_run_summary: PriorRunSummary | None = None
    current_run_summary: CurrentRunSummary | None = None
    runs: list[RunUnit] = []
    current_run: RunUnit | None = None
    index = 1
    while index < len(messages):
        message = messages[index]
        if message.role == "system":
            scope = _summary_scope(message)
            if scope is None:
                if current_run is not None:
                    raise AceAIRuntimeError(
                        "context compression does not support system messages inside a run"
                    )
                raise AceAIRuntimeError(
                    "context compression only supports scoped context summary system messages"
                )
            if scope == "prior_runs":
                if current_run is not None:
                    raise AceAIRuntimeError(
                        "prior-run summary cannot appear inside a run"
                    )
                prior_run_summary = PriorRunSummary(
                    content=_summary_content(message),
                )
            else:
                if current_run is None:
                    raise AceAIRuntimeError(
                        "current-run summary requires a current user-message boundary"
                    )
                if current_run.steps or current_run.open_step is not None:
                    raise AceAIRuntimeError(
                        "current-run summary must appear before raw current-run steps"
                    )
                current_run_summary = CurrentRunSummary(
                    content=_summary_content(message),
                )
            index += 1
            continue
        if message.role == "user":
            if current_run is not None:
                runs.append(current_run)
            current_run = RunUnit(user_message=message)
            index += 1
            continue
        if current_run is None:
            raise AceAIRuntimeError(
                "context compression requires user-message run boundaries"
            )
        step, next_index = _parse_step(messages, index)
        if isinstance(step, OpenStepUnit):
            if next_index != len(messages):
                raise AceAIRuntimeError(
                    "open step must be the final model-facing context unit"
                )
            current_run.open_step = step
        else:
            current_run.steps.append(step)
        index = next_index

    if current_run is None:
        raise AceAIRuntimeError(
            "context compression requires user-message run boundaries"
        )
    return StructuredContext(
        system_message=system_message,
        prior_run_summary=prior_run_summary,
        current_run_summary=current_run_summary,
        prior_runs=runs,
        current_run=current_run,
    )


def render_context_units(context: StructuredContext) -> list[LLMMessage]:
    messages = [context.system_message]
    if context.prior_run_summary is not None:
        messages.append(
            _summary_message("prior_runs", context.prior_run_summary.content)
        )
    current_run = context.current_run
    if current_run is None:
        return messages
    messages.append(current_run.user_message)
    if context.current_run_summary is not None:
        messages.append(
            _summary_message("current_run", context.current_run_summary.content)
        )
    for step in current_run.steps:
        messages.extend(step.messages)
    if current_run.open_step is not None:
        messages.extend(current_run.open_step.messages)
    return messages


def _has_compressible_context(
    context: StructuredContext,
    *,
    policy: ContextCompressionPolicy,
    force_current_run_steps: bool,
) -> bool:
    if context.prior_runs:
        return True
    return (
        _split_current_run_steps(
            context=context,
            policy=policy,
            force_current_run_steps=force_current_run_steps,
        )
        is not None
    )


async def _compact_structured_context(
    context: StructuredContext,
    *,
    llm_service: ILLMService,
    policy: ContextCompressionPolicy,
    force_current_run_steps: bool,
) -> StructuredContext | None:
    current_run = context.current_run
    if current_run is None:
        raise AceAIRuntimeError("context compression requires a current run")

    prior_run_summary = context.prior_run_summary
    current_run_summary = context.current_run_summary
    compressed = False
    if context.prior_runs:
        prior_summary_text = await _summarize_units(
            llm_service=llm_service,
            scope="prior_runs",
            policy=policy,
            existing_summary=prior_run_summary.content
            if prior_run_summary is not None
            else "",
            messages=_run_messages(context.prior_runs),
        )
        prior_run_summary = PriorRunSummary(content=prior_summary_text)
        compressed = True

    step_split = _split_current_run_steps(
        context=StructuredContext(
            system_message=context.system_message,
            prior_run_summary=prior_run_summary,
            current_run_summary=current_run_summary,
            current_run=current_run,
        ),
        policy=policy,
        force_current_run_steps=force_current_run_steps,
    )
    if step_split is not None:
        summarized_steps, retained_steps = step_split
        current_summary_text = await _summarize_units(
            llm_service=llm_service,
            scope="current_run",
            policy=policy,
            existing_summary=current_run_summary.content
            if current_run_summary is not None
            else "",
            messages=_step_messages(summarized_steps),
        )
        current_run_summary = CurrentRunSummary(content=current_summary_text)
        current_run = RunUnit(
            user_message=current_run.user_message,
            steps=retained_steps,
            open_step=current_run.open_step,
        )
        compressed = True

    if not compressed:
        return None
    return StructuredContext(
        system_message=context.system_message,
        prior_run_summary=prior_run_summary,
        current_run_summary=current_run_summary,
        current_run=current_run,
    )


def _no_compressible_context_message() -> str:
    return (
        "Context compaction cannot reduce this request because there are no "
        "completed prior runs or completed current-run steps available to "
        "summarize. The oversized content is in required context such as the "
        "current user message, open tool exchange, system instructions, tool "
        "schemas, or attached context."
    )


def _split_current_run_steps(
    *,
    context: StructuredContext,
    policy: ContextCompressionPolicy,
    force_current_run_steps: bool,
) -> tuple[list[StepUnit], list[StepUnit]] | None:
    current_run = context.current_run
    if current_run is None:
        raise AceAIRuntimeError("context compression requires a current run")
    if len(current_run.steps) < 2:
        return None

    base_messages = [context.system_message, current_run.user_message]
    if context.prior_run_summary is not None:
        base_messages.append(
            _summary_message("prior_runs", context.prior_run_summary.content)
        )
    if context.current_run_summary is not None:
        base_messages.append(
            _summary_message("current_run", context.current_run_summary.content)
        )
    if current_run.open_step is not None:
        base_messages.extend(current_run.open_step.messages)
    base_tokens = estimate_message_tokens(base_messages)
    adaptive_budget = max(
        1,
        min(
            policy.recent_step_budget,
            policy.context_window_tokens - base_tokens - policy.safety_margin_tokens,
        ),
    )

    retained_reversed: list[StepUnit] = []
    retained_tokens = 0
    for step in reversed(current_run.steps):
        step_tokens = estimate_message_tokens(step.messages)
        if retained_reversed and retained_tokens + step_tokens > adaptive_budget:
            break
        retained_reversed.append(step)
        retained_tokens += step_tokens
    retained_steps = list(reversed(retained_reversed))
    summarized_count = len(current_run.steps) - len(retained_steps)
    if summarized_count == 0:
        if not force_current_run_steps:
            return None
        retained_steps = [current_run.steps[-1]]
        summarized_count = len(current_run.steps) - 1
    if summarized_count == 0:
        return None
    return current_run.steps[:summarized_count], retained_steps


def _parse_step(
    messages: list[LLMMessage],
    index: int,
) -> tuple[StepUnit | OpenStepUnit, int]:
    message = messages[index]
    if isinstance(message, LLMToolUseMessage):
        raise AceAIRuntimeError(
            "context compression found tool output without a tool call in the same step"
        )
    if isinstance(message, LLMToolCallMessage):
        call_ids = {tool_call.call_id for tool_call in message.tool_calls}
        if not call_ids:
            raise AceAIRuntimeError("assistant tool-call step has no tool calls")
        step_messages: list[LLMMessage] = [message]
        index += 1
        while index < len(messages):
            next_message = messages[index]
            if not isinstance(next_message, LLMToolUseMessage):
                break
            if next_message.call_id not in call_ids:
                raise AceAIRuntimeError(
                    "context compression found tool output without a tool call in the same step"
                )
            step_messages.append(next_message)
            call_ids.remove(next_message.call_id)
            index += 1
        if call_ids:
            return OpenStepUnit(messages=step_messages), index
        return StepUnit(messages=step_messages), index
    if message.role != "assistant":
        raise AceAIRuntimeError("context compression found unsupported run message")
    return StepUnit(messages=[message]), index + 1


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


async def _summarize_units(
    *,
    llm_service: ILLMService,
    scope: ContextSummaryScope,
    policy: ContextCompressionPolicy,
    existing_summary: str,
    messages: list[LLMMessage],
) -> str:
    summary = existing_summary
    chunks = _summary_chunks(
        scope=scope,
        existing_summary=summary,
        messages=messages,
        policy=policy,
    )
    for chunk in chunks:
        response = await llm_service.complete(
            messages=_summary_request_messages(
                scope=scope,
                existing_summary=summary,
                messages=chunk,
            ),
        )
        summary = response.text
    return summary


def _summary_chunks(
    *,
    scope: ContextSummaryScope,
    existing_summary: str,
    messages: list[LLMMessage],
    policy: ContextCompressionPolicy,
) -> list[list[LLMMessage]]:
    chunks: list[list[LLMMessage]] = []
    current_chunk: list[LLMMessage] = []
    for message in messages:
        candidate = [*current_chunk, message]
        if _summary_request_tokens(
            scope=scope,
            existing_summary=existing_summary,
            messages=candidate,
        ) <= policy.summary_request_budget_tokens:
            current_chunk = candidate
            continue
        if not current_chunk:
            raise LLMProviderError(
                "Context compaction cannot summarize this request because one "
                "context unit is larger than the compaction summary budget. "
                "Reduce the oversized tool output, attachment, or user input "
                "before it enters model-facing context."
            )
        chunks.append(current_chunk)
        current_chunk = [message]
        if _summary_request_tokens(
            scope=scope,
            existing_summary=existing_summary,
            messages=current_chunk,
        ) > policy.summary_request_budget_tokens:
            raise LLMProviderError(
                "Context compaction cannot summarize this request because one "
                "context unit is larger than the compaction summary budget. "
                "Reduce the oversized tool output, attachment, or user input "
                "before it enters model-facing context."
            )
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _summary_request_messages(
    *,
    scope: ContextSummaryScope,
    existing_summary: str,
    messages: list[LLMMessage],
) -> list[LLMMessage]:
    return [
        LLMMessage.build(
            role="system",
            content=_summary_instruction(scope),
        ),
        LLMMessage.build(
            role="user",
            content=_format_summary_request(
                existing_summary=existing_summary,
                messages=messages,
            ),
        ),
    ]


def _summary_request_tokens(
    *,
    scope: ContextSummaryScope,
    existing_summary: str,
    messages: list[LLMMessage],
) -> int:
    return estimate_message_tokens(
        _summary_request_messages(
            scope=scope,
            existing_summary=existing_summary,
            messages=messages,
        )
    )


def _summary_instruction(scope: ContextSummaryScope) -> str:
    if scope == "prior_runs":
        return (
            "Compress earlier AceAI runs for a future agent turn. Preserve durable "
            "user goals, constraints, decisions, open tasks, file paths, tool "
            "outcomes, and unresolved errors. Do not invent facts. Return concise "
            "prose only."
        )
    return (
        "Compress completed steps in the active AceAI run. Preserve what has "
        "already happened, tool results, intermediate conclusions, remaining "
        "work, file paths, and unresolved errors. Do not invent facts. Return "
        "concise prose only."
    )


def _format_summary_request(
    *,
    existing_summary: str,
    messages: list[LLMMessage],
) -> str:
    sections: list[str] = []
    if existing_summary != "":
        sections.extend(["Existing summary:", existing_summary, ""])
    sections.append(_format_messages_for_summary(messages))
    return "\n".join(sections)


def _summary_message(scope: ContextSummaryScope, content: str) -> LLMMessage:
    return LLMMessage.build(
        role="system",
        content=(
            f'<aceai_context_summary scope="{scope}">\n'
            f"{content}\n"
            f"{CONTEXT_SUMMARY_CLOSE}"
        ),
    )


def _summary_scope(message: LLMMessage) -> ContextSummaryScope | None:
    text = _single_text_content(message)
    if text.startswith('<aceai_context_summary scope="prior_runs">'):
        return "prior_runs"
    if text.startswith('<aceai_context_summary scope="current_run">'):
        return "current_run"
    return None


def _summary_content(message: LLMMessage) -> str:
    text = _single_text_content(message)
    start = text.find(">")
    end = text.rfind(CONTEXT_SUMMARY_CLOSE)
    if start == -1 or end == -1:
        raise AceAIRuntimeError("context summary message is malformed")
    return text[start + 1 : end]


def _single_text_content(message: LLMMessage) -> str:
    if len(message.content) != 1:
        raise AceAIRuntimeError("context summary message must contain one text part")
    part = message.content[0]
    if part["type"] != "text":
        raise AceAIRuntimeError("context summary message must be text")
    if "data" not in part:
        raise ValueError("text message part must include data")
    return part["data"]


def _run_messages(runs: list[RunUnit]) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    for run in runs:
        messages.append(run.user_message)
        messages.extend(_step_messages(run.steps))
        if run.open_step is not None:
            messages.extend(run.open_step.messages)
    return messages


def _step_messages(steps: list[StepUnit]) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    for step in steps:
        messages.extend(step.messages)
    return messages


def _format_messages_for_summary(messages: list[LLMMessage]) -> str:
    lines: list[str] = [
        "Earlier messages to compress:",
    ]
    for index, message in enumerate(messages, start=1):
        lines.append(f'<message index="{index}" role="{message.role}">')
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
            lines.append(f"tool_call name={tool_call.name} call_id={tool_call.call_id}")
            lines.append(tool_call.arguments)
    if isinstance(message, LLMToolUseMessage):
        lines.append(f"tool_result name={message.name} call_id={message.call_id}")
    return lines
