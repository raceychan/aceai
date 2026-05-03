"""Configuration records and persistence for the AceAI CLI TUI."""

from pathlib import Path
from typing import Literal

import yaml

from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

ProviderName = Literal["openai"]


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
    if provider != "openai":
        raise ValueError("AceAI config provider must be openai")
    if type(api_key) is not str:
        raise TypeError("AceAI config api_key must be str")
    if model not in (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5o",
        "gpt-5o-mini",
        "gpt-5.1",
        "o3-large",
        "o4-mini",
    ):
        raise ValueError("AceAI config model is unsupported")
    return AceAITUIConfig(provider=provider, api_key=api_key, model=model)


def save_config(config: AceAITUIConfig, path: Path | None = None) -> None:
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "provider": config.provider,
                "api_key": config.api_key,
                "model": config.model,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target.chmod(0o600)
