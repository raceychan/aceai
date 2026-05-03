"""Configuration records and persistence for the AceAI CLI TUI."""

from pathlib import Path
from typing import Literal

import yaml

from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

ProviderName = Literal["openai"]
CONFIG_VERSION = 2
DEFAULT_OPENAI_MODEL: OpenAIModel = "gpt-5.5"
STALE_DEFAULT_OPENAI_MODELS: tuple[OpenAIModel, ...] = ("gpt-5.1",)
OPENAI_MODEL_OPTIONS: tuple[tuple[str, OpenAIModel], ...] = (
    ("GPT-5.5", "gpt-5.5"),
    ("GPT-5.5 pro", "gpt-5.5-pro"),
    ("GPT-5.4", "gpt-5.4"),
    ("GPT-5.4 pro", "gpt-5.4-pro"),
    ("GPT-5.4 mini", "gpt-5.4-mini"),
    ("GPT-5.4 nano", "gpt-5.4-nano"),
    ("GPT-5.2", "gpt-5.2"),
    ("GPT-5.2 pro", "gpt-5.2-pro"),
    ("GPT-5.1", "gpt-5.1"),
    ("GPT-4o", "gpt-4o"),
    ("GPT-4o mini", "gpt-4o-mini"),
    ("o3", "o3"),
    ("o3 mini", "o3-mini"),
    ("o4 mini", "o4-mini"),
)
SUPPORTED_OPENAI_MODELS: tuple[OpenAIModel, ...] = tuple(
    option[1] for option in OPENAI_MODEL_OPTIONS
)


class AceAITUIConfig(Record, kw_only=True):
    provider: ProviderName
    api_key: str
    model: OpenAIModel


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
    api_key = data["api_key"]
    model = data["model"]
    has_config_version = "config_version" in data
    if has_config_version and type(data["config_version"]) is not int:
        raise TypeError("AceAI config config_version must be int")
    if provider != "openai":
        raise ValueError("AceAI config provider must be openai")
    if type(api_key) is not str:
        raise TypeError("AceAI config api_key must be str")
    if model not in SUPPORTED_OPENAI_MODELS:
        raise ValueError("AceAI config model is unsupported")
    if not has_config_version and model in STALE_DEFAULT_OPENAI_MODELS:
        model = DEFAULT_OPENAI_MODEL
    return AceAITUIConfig(provider=provider, api_key=api_key, model=model)


def save_config(config: AceAITUIConfig, path: Path | None = None) -> None:
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "config_version": CONFIG_VERSION,
                "provider": config.provider,
                "api_key": config.api_key,
                "model": config.model,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target.chmod(0o600)
