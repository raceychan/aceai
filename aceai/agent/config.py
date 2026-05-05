"""Agent app configuration schema, singleton state, and persistence."""

from pathlib import Path
from typing import Literal

import yaml
from msgspec import field

from aceai.agent.provider_catalog import (
    default_model,
    model_options,
    stale_default_models,
    supported_models,
    supported_provider_names,
)
from aceai.agent.ace_agent import ACE_AGENT_SKILLS_DIR
from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

ProviderName = str
ConfigValueType = Literal["string", "mapping", "list"]
SkillSelectionMode = Literal["all", "selected"]
CONFIG_VERSION = 2
DEFAULT_OPENAI_MODEL: OpenAIModel = default_model("openai")
STALE_DEFAULT_OPENAI_MODELS: tuple[OpenAIModel, ...] = stale_default_models("openai")
OPENAI_MODEL_OPTIONS: tuple[tuple[str, OpenAIModel], ...] = model_options("openai")
SUPPORTED_OPENAI_MODELS: tuple[OpenAIModel, ...] = supported_models("openai")


class ConfigField(Record, kw_only=True):
    name: str
    value_type: ConfigValueType
    required: bool
    description: str


class ConfigSchema(Record, kw_only=True):
    version: int
    fields: tuple[ConfigField, ...]


class AgentAppConfig(Record, kw_only=True):
    provider: ProviderName
    api_key: str
    model: OpenAIModel
    default_model: OpenAIModel = DEFAULT_OPENAI_MODEL
    skills: str = str(ACE_AGENT_SKILLS_DIR)
    skill_selection_mode: SkillSelectionMode = "all"
    enabled_skills: list[str] = field(default_factory=list[str])
    api_keys: dict[str, str] = field(default_factory=dict[str, str])


AceAITUIConfig = AgentAppConfig

APP_CONFIG_SCHEMA = ConfigSchema(
    version=CONFIG_VERSION,
    fields=(
        ConfigField(
            name="provider",
            value_type="string",
            required=True,
            description="Active model provider.",
        ),
        ConfigField(
            name="model",
            value_type="string",
            required=True,
            description="Active model for new agent turns.",
        ),
        ConfigField(
            name="default_model",
            value_type="string",
            required=True,
            description="Default model used when building the app agent.",
        ),
        ConfigField(
            name="api_key",
            value_type="string",
            required=True,
            description="API key for the active provider.",
        ),
        ConfigField(
            name="skills",
            value_type="string",
            required=True,
            description="Internal skill loading path or mode.",
        ),
        ConfigField(
            name="skill_selection_mode",
            value_type="string",
            required=True,
            description="Whether all loaded skills or selected skills are enabled.",
        ),
        ConfigField(
            name="enabled_skills",
            value_type="list",
            required=True,
            description="Skill names enabled for the active agent model.",
        ),
        ConfigField(
            name="api_keys",
            value_type="mapping",
            required=True,
            description="Provider-key mapping persisted for known providers.",
        ),
    ),
)

_ACTIVE_CONFIG: AgentAppConfig | None = None


def config_schema() -> ConfigSchema:
    return APP_CONFIG_SCHEMA


def current_config() -> AgentAppConfig | None:
    return _ACTIVE_CONFIG


def replace_config(config: AgentAppConfig) -> AgentAppConfig:
    global _ACTIVE_CONFIG
    validate_config(config)
    old_config = _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    del old_config
    return _ACTIVE_CONFIG


def clear_config() -> None:
    global _ACTIVE_CONFIG
    old_config = _ACTIVE_CONFIG
    _ACTIVE_CONFIG = None
    del old_config


def validate_config(config: AgentAppConfig) -> None:
    if config.provider not in supported_provider_names():
        raise ValueError("AceAI config provider is unsupported")
    if config.model not in supported_models(config.provider):
        raise ValueError("AceAI config model is unsupported")
    if config.default_model not in supported_models(config.provider):
        raise ValueError("AceAI config default_model is unsupported")
    if config.api_key == "":
        raise ValueError("AceAI config api_key is required")
    if config.skills == "":
        raise ValueError("AceAI config skills is required")
    if config.skill_selection_mode not in ("all", "selected"):
        raise ValueError("AceAI config skill_selection_mode is unsupported")
    for skill_name in config.enabled_skills:
        if type(skill_name) is not str:
            raise TypeError("AceAI config enabled_skills values must be str")
    if config.provider not in config.api_keys:
        raise ValueError("AceAI config api_keys missing active provider")
    if config.api_keys[config.provider] != config.api_key:
        raise ValueError("AceAI config active api_key does not match api_keys")


def default_config_path() -> Path:
    return Path.home() / ".aceai" / "config.yaml"


def load_config(path: Path | None = None) -> AgentAppConfig | None:
    target = path or default_config_path()
    if not target.exists():
        clear_config()
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
    configured_default_model = (
        data["default_model"] if "default_model" in data else model
    )
    skills = data["skills"] if "skills" in data else str(ACE_AGENT_SKILLS_DIR)
    skill_selection_mode = (
        data["skill_selection_mode"] if "skill_selection_mode" in data else "all"
    )
    enabled_skills = _load_enabled_skills(data)
    has_config_version = "config_version" in data
    if has_config_version and type(data["config_version"]) is not int:
        raise TypeError("AceAI config config_version must be int")
    if model not in supported_models(provider):
        raise ValueError("AceAI config model is unsupported")
    if configured_default_model not in supported_models(provider):
        raise ValueError("AceAI config default_model is unsupported")
    if type(skills) is not str:
        raise TypeError("AceAI config skills must be str")
    if skill_selection_mode not in ("all", "selected"):
        raise ValueError("AceAI config skill_selection_mode is unsupported")
    if not has_config_version and model in stale_default_models(provider):
        model = default_model(provider)
    if (
        not has_config_version
        and configured_default_model in stale_default_models(provider)
    ):
        configured_default_model = default_model(provider)
    return replace_config(
        AgentAppConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            default_model=configured_default_model,
            skills=skills,
            skill_selection_mode=skill_selection_mode,
            enabled_skills=enabled_skills,
            api_keys=api_keys,
        )
    )


def save_config(config: AgentAppConfig, path: Path | None = None) -> None:
    validate_config(config)
    replace_config(config)
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "config_version": CONFIG_VERSION,
                "provider": config.provider,
                "api_keys": {**config.api_keys, config.provider: config.api_key},
                "model": config.model,
                "default_model": config.default_model,
                "skills": config.skills,
                "skill_selection_mode": config.skill_selection_mode,
                "enabled_skills": config.enabled_skills,
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


def _load_enabled_skills(data: dict[object, object]) -> list[str]:
    if "enabled_skills" not in data:
        return []
    raw_enabled_skills = data["enabled_skills"]
    if type(raw_enabled_skills) is not list:
        raise TypeError("AceAI config enabled_skills must be a list")
    enabled_skills: list[str] = []
    for skill_name in raw_enabled_skills:
        if type(skill_name) is not str:
            raise TypeError("AceAI config enabled_skills values must be str")
        enabled_skills.append(skill_name)
    return enabled_skills
