"""App-layer helpers that read LLM response and context-message internals on
behalf of the TUI.

The TUI conversion (`_agent_event_to_tui_event`) lives in `agent_core.tui`.
What lives here is the deep-access logic the boundary review flagged: cost
calculation that needs provider awareness, and context-summary text extraction
that has to know the `<aceai_context_summary>` message format.
"""

from agent_core.cost import CostEstimate, estimate_usage_cost
from aceai.core.events import ContextCompressedEvent, LLMCompletedEvent
from aceai.llm.interface import is_set
from aceai.llm.models import LLMUsage


def usage_for_llm_completed(event: LLMCompletedEvent) -> LLMUsage | None:
    response = event.step.llm_response
    if is_set(response.usage):
        return response.usage
    return None


def cost_for_llm_completed(
    event: LLMCompletedEvent,
    *,
    provider_name: str | None = None,
) -> CostEstimate | None:
    response = event.step.llm_response
    usage = usage_for_llm_completed(event)
    effective_provider = provider_name
    if effective_provider is None and response.provider_meta:
        effective_provider = response.provider_meta[0].provider_name
    return estimate_usage_cost(
        response.model,
        usage,
        provider_name=effective_provider,
    )


def context_summary_text(event: ContextCompressedEvent) -> str:
    """Extract the latest `<aceai_context_summary>` body from history."""

    for message in event.history:
        if message.role != "system":
            continue
        if len(message.content) != 1:
            continue
        part = message.content[0]
        if part["type"] != "text" or "data" not in part:
            continue
        text = part["data"]
        if text.startswith("<aceai_context_summary"):
            return _strip_context_summary_tags(text)
    return ""


def _strip_context_summary_tags(text: str) -> str:
    start = text.find(">")
    end = text.rfind("</aceai_context_summary>")
    if start == -1 or end == -1:
        return text
    return text[start + 1 : end]
