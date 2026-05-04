"""Provider and model catalog loaded from package data."""

from functools import cache
from importlib import resources

import yaml
from typing_extensions import Self

from aceai.llm.interface import Record


CATALOG_RESOURCE = "provider_catalog.yaml"


class ModelTokenPrice(Record, kw_only=True):
    model: str
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float

    @classmethod
    def from_raw(cls, model_id: str, raw: object) -> Self:
        if type(raw) is not dict:
            raise TypeError("Provider catalog pricing must be a mapping")
        input_price = raw["input_usd_per_million"]
        cached_input_price = raw["cached_input_usd_per_million"]
        output_price = raw["output_usd_per_million"]
        if type(input_price) is not float:
            raise TypeError("Provider catalog input_usd_per_million must be float")
        if type(cached_input_price) is not float:
            raise TypeError("Provider catalog cached_input_usd_per_million must be float")
        if type(output_price) is not float:
            raise TypeError("Provider catalog output_usd_per_million must be float")
        return cls(
            model=model_id,
            input_usd_per_million=input_price,
            cached_input_usd_per_million=cached_input_price,
            output_usd_per_million=output_price,
        )


class ModelCatalogEntry(Record, kw_only=True):
    id: str
    label: str
    pricing: ModelTokenPrice | None = None

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        if type(raw) is not dict:
            raise TypeError("Provider catalog model must be a mapping")
        model_id = raw["id"]
        label = raw["label"]
        if type(model_id) is not str:
            raise TypeError("Provider catalog model id must be str")
        if type(label) is not str:
            raise TypeError("Provider catalog model label must be str")
        pricing = None
        if "pricing" in raw:
            pricing = ModelTokenPrice.from_raw(model_id, raw["pricing"])
        return cls(id=model_id, label=label, pricing=pricing)


class ProviderCatalogEntry(Record, kw_only=True):
    name: str
    label: str
    api_key_env: str
    default_model: str
    stale_default_models: tuple[str, ...]
    models: tuple[ModelCatalogEntry, ...]

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        if type(raw) is not dict:
            raise TypeError("Provider catalog provider must be a mapping")
        name = raw["name"]
        label = raw["label"]
        api_key_env_value = raw["api_key_env"]
        default_model_value = raw["default_model"]
        stale_default_model_values = raw["stale_default_models"]
        models = raw["models"]
        if type(name) is not str:
            raise TypeError("Provider catalog provider name must be str")
        if type(label) is not str:
            raise TypeError("Provider catalog provider label must be str")
        if type(api_key_env_value) is not str:
            raise TypeError("Provider catalog api_key_env must be str")
        if type(default_model_value) is not str:
            raise TypeError("Provider catalog default_model must be str")
        if type(stale_default_model_values) is not list:
            raise TypeError("Provider catalog stale_default_models must be a list")
        if type(models) is not list:
            raise TypeError("Provider catalog models must be a list")
        parsed_stale_models = tuple(
            _require_str(model, "Provider catalog stale model must be str")
            for model in stale_default_model_values
        )
        parsed_models = tuple(ModelCatalogEntry.from_raw(model) for model in models)
        return cls(
            name=name,
            label=label,
            api_key_env=api_key_env_value,
            default_model=default_model_value,
            stale_default_models=parsed_stale_models,
            models=parsed_models,
        )


class ProviderCatalog(Record, kw_only=True):
    providers: tuple[ProviderCatalogEntry, ...]
    pricing_source: str

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        if type(raw) is not dict:
            raise TypeError("Provider catalog must be a mapping")
        providers = raw["providers"]
        pricing_source_value = raw["pricing_source"]
        if type(providers) is not list:
            raise TypeError("Provider catalog providers must be a list")
        if type(pricing_source_value) is not str:
            raise TypeError("Provider catalog pricing_source must be str")
        return cls(
            providers=tuple(
                ProviderCatalogEntry.from_raw(provider) for provider in providers
            ),
            pricing_source=pricing_source_value,
        )


@cache
def load_provider_catalog() -> ProviderCatalog:
    text = resources.files("aceai.agent").joinpath(CATALOG_RESOURCE).read_text(
        encoding="utf-8"
    )
    raw = yaml.safe_load(text)
    return ProviderCatalog.from_raw(raw)


def supported_provider_names() -> tuple[str, ...]:
    return tuple(provider.name for provider in load_provider_catalog().providers)


def provider_options() -> tuple[tuple[str, str], ...]:
    return tuple(
        (provider.label, provider.name) for provider in load_provider_catalog().providers
    )


def get_provider_catalog(provider_name: str) -> ProviderCatalogEntry:
    for provider in load_provider_catalog().providers:
        if provider.name == provider_name:
            return provider
    raise ValueError("Unsupported provider")


def model_options(provider_name: str) -> tuple[tuple[str, str], ...]:
    provider = get_provider_catalog(provider_name)
    return tuple((model.label, model.id) for model in provider.models)


def supported_models(provider_name: str) -> tuple[str, ...]:
    provider = get_provider_catalog(provider_name)
    return tuple(model.id for model in provider.models)


def all_supported_models() -> tuple[str, ...]:
    models: list[str] = []
    for provider in load_provider_catalog().providers:
        models.extend(model.id for model in provider.models)
    return tuple(models)


def default_model(provider_name: str) -> str:
    return get_provider_catalog(provider_name).default_model


def stale_default_models(provider_name: str) -> tuple[str, ...]:
    return get_provider_catalog(provider_name).stale_default_models


def api_key_env(provider_name: str) -> str:
    return get_provider_catalog(provider_name).api_key_env


def price_for_model(provider_name: str, model: str) -> ModelTokenPrice | None:
    provider = get_provider_catalog(provider_name)
    for entry in provider.models:
        if entry.pricing is not None and model == entry.id:
            return entry.pricing
    for entry in provider.models:
        if entry.pricing is not None and model.startswith(f"{entry.id}-"):
            return entry.pricing
    return None


def price_for_model_any_provider(model: str) -> ModelTokenPrice | None:
    for provider in load_provider_catalog().providers:
        price = price_for_model(provider.name, model)
        if price is not None:
            return price
    return None


def pricing_source() -> str:
    return load_provider_catalog().pricing_source


def _require_str(value: object, message: str) -> str:
    if type(value) is not str:
        raise TypeError(message)
    return value
