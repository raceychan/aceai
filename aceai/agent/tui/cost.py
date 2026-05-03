"""Estimated token cost helpers for the AceAI TUI."""

from aceai.llm.interface import Record
from aceai.llm.models import LLMUsage


PRICING_SOURCE = "openai-api-pricing-2026-05-04"


class ModelTokenPrice(Record, kw_only=True):
    model: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


class TUICostEstimate(Record, kw_only=True):
    model: str
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    pricing_source: str


MODEL_TOKEN_PRICES: tuple[ModelTokenPrice, ...] = (
    ModelTokenPrice(
        model="gpt-5.5-pro",
        input_usd_per_million=30.0,
        cached_input_usd_per_million=30.0,
        output_usd_per_million=180.0,
    ),
    ModelTokenPrice(
        model="gpt-5.4-pro",
        input_usd_per_million=30.0,
        cached_input_usd_per_million=30.0,
        output_usd_per_million=180.0,
    ),
    ModelTokenPrice(
        model="gpt-5.5",
        input_usd_per_million=5.0,
        cached_input_usd_per_million=0.5,
        output_usd_per_million=30.0,
    ),
    ModelTokenPrice(
        model="gpt-5.4-mini",
        input_usd_per_million=0.75,
        cached_input_usd_per_million=0.075,
        output_usd_per_million=4.5,
    ),
    ModelTokenPrice(
        model="gpt-5.4",
        input_usd_per_million=2.5,
        cached_input_usd_per_million=0.25,
        output_usd_per_million=15.0,
    ),
)


def estimate_usage_cost(model: str | None, usage: LLMUsage | None) -> TUICostEstimate | None:
    if model is None or usage is None:
        return None
    price = _price_for_model(model)
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
    return TUICostEstimate(
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


def _price_for_model(model: str) -> ModelTokenPrice | None:
    for price in MODEL_TOKEN_PRICES:
        if model == price.model or model.startswith(f"{price.model}-"):
            return price
    return None


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
