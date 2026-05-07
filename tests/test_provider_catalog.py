from aceai.agent.provider_catalog import (
    context_window_for_model,
    default_model,
    load_provider_catalog,
    model_options,
    price_for_model,
    pricing_source,
    provider_options,
    stale_default_models,
    supported_models,
    supported_provider_names,
)


def test_provider_catalog_loads_supported_providers_and_models() -> None:
    catalog = load_provider_catalog()

    assert supported_provider_names() == ("openai", "deepseek")
    assert provider_options() == (("OpenAI", "openai"), ("DeepSeek", "deepseek"))
    assert catalog.pricing_source == "provider-api-pricing-2026-05-04"
    assert default_model("openai") == "gpt-5.5"
    assert stale_default_models("openai") == ("gpt-5.1",)
    assert model_options("openai")[0] == ("GPT-5.5", "gpt-5.5")
    assert supported_models("openai") == tuple(
        model_id for _, model_id in model_options("openai")
    )


def test_provider_catalog_loads_model_pricing() -> None:
    price = price_for_model("openai", "gpt-5.5")

    assert price is not None
    assert price.input_usd_per_million == 5.0
    assert price.cached_input_usd_per_million == 0.5
    assert price.output_usd_per_million == 30.0
    assert pricing_source() == "provider-api-pricing-2026-05-04"


def test_provider_catalog_loads_deepseek_models_and_pricing() -> None:
    price = price_for_model("deepseek", "deepseek-v4-pro")

    assert default_model("deepseek") == "deepseek-v4-pro"
    assert model_options("deepseek")[0] == ("DeepSeek V4 Flash", "deepseek-v4-flash")
    assert price is not None
    assert price.input_usd_per_million == 0.435
    assert price.cached_input_usd_per_million == 0.003625
    assert price.output_usd_per_million == 0.87


def test_provider_catalog_matches_versioned_model_suffixes() -> None:
    price = price_for_model("openai", "gpt-5.5-2026-05-04")

    assert price is not None
    assert price.output_usd_per_million == 30.0


def test_context_window_exact_match() -> None:
    assert context_window_for_model("openai", "gpt-5.5") == 1050000
    assert context_window_for_model("openai", "gpt-5.4") == 1050000
    assert context_window_for_model("openai", "gpt-5.4-mini") == 400000
    assert context_window_for_model("openai", "gpt-5.2") == 400000
    assert context_window_for_model("openai", "gpt-4o") == 128000
    assert context_window_for_model("openai", "o3") == 200000
    assert context_window_for_model("deepseek", "deepseek-v4-pro") == 1000000
    assert context_window_for_model("deepseek", "deepseek-v4-flash") == 1000000
    assert context_window_for_model("deepseek", "deepseek-chat") == 1000000


def test_context_window_versioned_model_suffix() -> None:
    assert context_window_for_model("openai", "gpt-5.5-2026-05-04") == 1050000


def test_context_window_returns_none_for_missing() -> None:
    assert context_window_for_model("openai", "nonexistent-model") is None


def test_context_window_raises_for_unsupported_provider() -> None:
    import pytest as pt

    with pt.raises(ValueError, match="Unsupported provider"):
        context_window_for_model("unsupported-provider", "gpt-5.5")
