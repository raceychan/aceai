"""Configuration records and persistence for the AceAI CLI TUI."""

from pathlib import Path
import yaml
from msgspec import field

from aceai.agent.provider_catalog import (
    default_model,
    model_options,
    stale_default_models,
    supported_models,
    supported_provider_names,
)
from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

ProviderName = str
CONFIG_VERSION = 2
DEFAULT_OPENAI_MODEL: OpenAIModel = default_model("openai")
STALE_DEFAULT_OPENAI_MODELS: tuple[OpenAIModel, ...] = stale_default_models("openai")
OPENAI_MODEL_OPTIONS: tuple[tuple[str, OpenAIModel], ...] = model_options("openai")
SUPPORTED_OPENAI_MODELS: tuple[OpenAIModel, ...] = supported_models("openai")


class AceAITUIConfig(Record, kw_only=True):
    provider: ProviderName
    api_key: str
    model: OpenAIModel
    api_keys: dict[str, str] = field(default_factory=dict[str, str])


def default_config_path() -> Path:
    return Path.home() / ".aceai" / "config.yaml"


def load_config(path: Path | None = None) -> AceAITUIConfig | None:
    target = path or default_config_path()
    if not target.exists():
        return None
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("AceAI config must be a mapping")
    provider = data["provider"]
    if provider not in supported_provider_names():
        raise ValueError("AceAI config provider is unsupported")
    api_keys = _load_api_keys(data)
    if provider not in api_keys:
        raise ValueError("AceAI config api_keys missing active provider")
    api_key = api_keys[provider]
    model = data["model"] if "model" in data else default_model(provider)
    has_config_version = "config_version" in data
    if has_config_version and type(data["config_version"]) is not int:
        raise TypeError("AceAI config config_version must be int")
    if model not in supported_models(provider):
        raise ValueError("AceAI config model is unsupported")
    if not has_config_version and model in stale_default_models(provider):
        model = default_model(provider)
    return AceAITUIConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        api_keys=api_keys,
    )


def save_config(config: AceAITUIConfig, path: Path | None = None) -> None:
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "config_version": CONFIG_VERSION,
                "provider": config.provider,
                "api_keys": {**config.api_keys, config.provider: config.api_key},
                "model": config.model,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target.chmod(0o600)


def _load_api_keys(data: dict[object, object]) -> dict[str, str]:
    api_keys: dict[str, str] = {}
    if "api_keys" not in data:
        return api_keys
    raw_api_keys = data["api_keys"]
    if not isinstance(raw_api_keys, dict):
        raise TypeError("AceAI config api_keys must be a mapping")
    for provider, api_key in raw_api_keys.items():
        if type(provider) is not str:
            raise TypeError("AceAI config api_keys provider names must be str")
        if provider not in supported_provider_names():
            raise ValueError("AceAI config api_keys provider is unsupported")
        if type(api_key) is not str:
            raise TypeError("AceAI config api_keys values must be str")
        api_keys[provider] = api_key
    return api_keys
