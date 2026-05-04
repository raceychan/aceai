"""Estimated token cost helpers for AceAI agent sessions."""

from aceai.agent.provider_catalog import (
    ModelTokenPrice,
    price_for_model,
    price_for_model_any_provider,
    pricing_source,
)
from aceai.llm.interface import Record
from aceai.llm.models import LLMUsage


PRICING_SOURCE = pricing_source()


class CostEstimate(Record, kw_only=True):
    model: str
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    pricing_source: str


def estimate_usage_cost(
    model: str | None,
    usage: LLMUsage | None,
    *,
    provider_name: str | None = None,
) -> CostEstimate | None:
    if model is None or usage is None:
        return None
    price = _price_for_model(model, provider_name=provider_name)
    if price is None:
        return None
    input_tokens = _tokens(usage.input_tokens)
    cached_input_tokens = _cached_tokens(usage.cached_input_tokens, input_tokens)
    uncached_input_tokens = input_tokens - cached_input_tokens
    output_tokens = _tokens(usage.output_tokens)
    input_cost = uncached_input_tokens * price.input_usd_per_million / 1_000_000
    cached_input_cost = (
        cached_input_tokens * price.cached_input_usd_per_million / 1_000_000
    )
    output_cost = output_tokens * price.output_usd_per_million / 1_000_000
    return CostEstimate(
        model=price.model,
        input_cost_usd=input_cost,
        cached_input_cost_usd=cached_input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + cached_input_cost + output_cost,
        input_usd_per_million=price.input_usd_per_million,
        cached_input_usd_per_million=price.cached_input_usd_per_million,
        output_usd_per_million=price.output_usd_per_million,
        pricing_source=PRICING_SOURCE,
    )


def format_usd(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _price_for_model(
    model: str,
    *,
    provider_name: str | None = None,
) -> ModelTokenPrice | None:
    if provider_name is not None:
        return price_for_model(provider_name, model)
    return price_for_model_any_provider(model)


def _tokens(value: int | None) -> int:
    if value is None:
        return 0
    return value


def _cached_tokens(value: int | None, input_tokens: int) -> int:
    if value is None:
        return 0
    if value > input_tokens:
        raise ValueError("Cached input tokens cannot exceed input tokens")
    return value
