"""Agent app configuration schema, singleton state, and persistence."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import yaml
from msgspec import field

from aceai.agent.provider_catalog import (
    default_model,
    model_options,
    reasoning_effort_options,
    stale_default_models,
    supported_models,
    supported_provider_names,
)
from aceai.agent.permissions import ToolPermission
from aceai.agent.ace_agent import ACE_AGENT_SKILL_PATH
from aceai.core.context_manager import CompressThreshold, ContextCompressionPolicy
from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

ProviderName = str
ConfigValueType = Literal["string", "mapping", "list"]
SkillSelectionMode = Literal["all", "selected"]
ReasoningLevel = Literal["auto", "low", "medium", "high", "max"]
CONFIG_VERSION = 4
DEFAULT_OPENAI_MODEL: OpenAIModel = default_model("openai")
STALE_DEFAULT_OPENAI_MODELS: tuple[OpenAIModel, ...] = stale_default_models("openai")
OPENAI_MODEL_OPTIONS: tuple[tuple[str, OpenAIModel], ...] = model_options("openai")
SUPPORTED_OPENAI_MODELS: tuple[OpenAIModel, ...] = supported_models("openai")
LEGACY_AGENT_SKILLS_DIR = Path(__file__).parent / "features" / "skills"


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
    skills: str = ACE_AGENT_SKILL_PATH
    skill_selection_mode: SkillSelectionMode = "all"
    enabled_skills: list[str] = field(default_factory=list[str])
    api_keys: dict[str, str] = field(default_factory=dict[str, str])
    tool_permissions: dict[str, ToolPermission] = field(
        default_factory=dict[str, ToolPermission]
    )
    tool_enabled: dict[str, bool] = field(default_factory=dict[str, bool])
    tool_max_calls: dict[str, int] = field(default_factory=dict[str, int])
    compress_threshold: CompressThreshold = "100%"
    reasoning_level: ReasoningLevel = "auto"


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
        ConfigField(
            name="tool_permissions",
            value_type="mapping",
            required=True,
            description="Per-tool permission policy: always or ask.",
        ),
        ConfigField(
            name="tool_enabled",
            value_type="mapping",
            required=True,
            description="Per-tool enabled state. Disabled tools are not loaded.",
        ),
        ConfigField(
            name="tool_max_calls",
            value_type="mapping",
            required=True,
            description="Per-tool maximum call count. Missing tools are unlimited.",
        ),
        ConfigField(
            name="compress_threshold",
            value_type="string",
            required=True,
            description="Context compression threshold as a percentage or token count.",
        ),
        ConfigField(
            name="reasoning_level",
            value_type="string",
            required=True,
            description="Reasoning effort level for models that support it.",
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
    if config.reasoning_level not in ("auto", "low", "medium", "high", "max"):
        raise ValueError("AceAI config reasoning_level is unsupported")
    if config.reasoning_level != "auto":
        options = reasoning_effort_options(config.provider, config.model)
        if config.reasoning_level not in options:
            raise ValueError("AceAI config reasoning_level is unsupported for model")
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
    for tool_name, permission in config.tool_permissions.items():
        if type(tool_name) is not str:
            raise TypeError("AceAI config tool_permissions tool names must be str")
        if permission not in ("always", "ask"):
            raise ValueError("AceAI config tool_permissions value is unsupported")
    for tool_name, enabled in config.tool_enabled.items():
        if type(tool_name) is not str:
            raise TypeError("AceAI config tool_enabled tool names must be str")
        if type(enabled) is not bool:
            raise TypeError("AceAI config tool_enabled values must be bool")
    for tool_name, max_calls in config.tool_max_calls.items():
        if type(tool_name) is not str:
            raise TypeError("AceAI config tool_max_calls tool names must be str")
        if type(max_calls) is not int:
            raise TypeError("AceAI config tool_max_calls values must be int")
        if max_calls < 1:
            raise ValueError("AceAI config tool_max_calls values must be positive")
    ContextCompressionPolicy(config.compress_threshold)


def default_config_path() -> Path:
    return Path.home() / ".aceai" / "config.yaml"


def project_config_path(project_dir: Path | None = None) -> Path:
    root = project_dir if project_dir is not None else Path.cwd()
    return root / ".aceai" / "config.yml"


def effective_config_path(project_dir: Path | None = None) -> Path:
    project_path = project_config_path(project_dir)
    if project_path.exists():
        return project_path
    return default_config_path()


def load_config(path: Path | None = None) -> AgentAppConfig | None:
    target = path if path is not None else effective_config_path()
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
    skills = data["skills"] if "skills" in data else ACE_AGENT_SKILL_PATH
    legacy_skill_path = False
    if skills == str(LEGACY_AGENT_SKILLS_DIR):
        skills = ACE_AGENT_SKILL_PATH
        legacy_skill_path = True
    skill_selection_mode = (
        data["skill_selection_mode"] if "skill_selection_mode" in data else "all"
    )
    enabled_skills = _load_enabled_skills(data)
    tool_permissions = _load_tool_permissions(data)
    tool_enabled = _load_tool_enabled(data)
    tool_max_calls = _load_tool_max_calls(data)
    compress_threshold = _load_compress_threshold(data)
    reasoning_level = _load_reasoning_level(data)
    if legacy_skill_path:
        skill_selection_mode = "all"
        enabled_skills = []
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
    if not has_config_version and configured_default_model in stale_default_models(
        provider
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
            tool_permissions=tool_permissions,
            tool_enabled=tool_enabled,
            tool_max_calls=tool_max_calls,
            compress_threshold=compress_threshold,
            reasoning_level=reasoning_level,
        )
    )


def save_config(config: AgentAppConfig, path: Path | None = None) -> None:
    validate_config(config)
    replace_config(config)
    target = path if path is not None else project_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        {
            "config_version": CONFIG_VERSION,
            "provider": config.provider,
            "api_keys": {**config.api_keys, config.provider: config.api_key},
            "model": config.model,
            "default_model": config.default_model,
            "skills": config.skills,
            "skill_selection_mode": config.skill_selection_mode,
            "enabled_skills": config.enabled_skills,
            "tool_permissions": config.tool_permissions,
            "tool_enabled": config.tool_enabled,
            "tool_max_calls": config.tool_max_calls,
            "compress_threshold": config.compress_threshold,
            "reasoning_level": config.reasoning_level,
        },
        sort_keys=False,
    )
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(target)
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


def _load_tool_permissions(data: dict[object, object]) -> dict[str, ToolPermission]:
    if "tool_permissions" not in data:
        return {}
    raw_permissions = data["tool_permissions"]
    if not isinstance(raw_permissions, dict):
        raise TypeError("AceAI config tool_permissions must be a mapping")
    permissions: dict[str, ToolPermission] = {}
    for tool_name, permission in raw_permissions.items():
        if type(tool_name) is not str:
            raise TypeError("AceAI config tool_permissions tool names must be str")
        if permission not in ("always", "ask"):
            raise ValueError("AceAI config tool_permissions value is unsupported")
        permissions[tool_name] = permission
    return permissions


def _load_tool_enabled(data: dict[object, object]) -> dict[str, bool]:
    if "tool_enabled" not in data:
        return {}
    raw_enabled = data["tool_enabled"]
    if not isinstance(raw_enabled, dict):
        raise TypeError("AceAI config tool_enabled must be a mapping")
    enabled_by_tool: dict[str, bool] = {}
    for tool_name, enabled in raw_enabled.items():
        if type(tool_name) is not str:
            raise TypeError("AceAI config tool_enabled tool names must be str")
        if type(enabled) is not bool:
            raise TypeError("AceAI config tool_enabled values must be bool")
        enabled_by_tool[tool_name] = enabled
    return enabled_by_tool


def _load_tool_max_calls(data: dict[object, object]) -> dict[str, int]:
    if "tool_max_calls" not in data:
        return {}
    raw_max_calls = data["tool_max_calls"]
    if not isinstance(raw_max_calls, dict):
        raise TypeError("AceAI config tool_max_calls must be a mapping")
    max_calls_by_tool: dict[str, int] = {}
    for tool_name, max_calls in raw_max_calls.items():
        if type(tool_name) is not str:
            raise TypeError("AceAI config tool_max_calls tool names must be str")
        if type(max_calls) is not int:
            raise TypeError("AceAI config tool_max_calls values must be int")
        if max_calls < 1:
            raise ValueError("AceAI config tool_max_calls values must be positive")
        max_calls_by_tool[tool_name] = max_calls
    return max_calls_by_tool


def _load_compress_threshold(data: dict[object, object]) -> CompressThreshold:
    if "compress_threshold" not in data:
        return "100%"
    threshold = data["compress_threshold"]
    if type(threshold) is str:
        ContextCompressionPolicy(threshold)
        return threshold
    if type(threshold) is int:
        ContextCompressionPolicy(threshold)
        return threshold
    if type(threshold) is float:
        ContextCompressionPolicy(threshold)
        return threshold
    raise TypeError("AceAI config compress_threshold must be str, int, or float")


def _load_reasoning_level(data: dict[object, object]) -> ReasoningLevel:
    if "reasoning_level" not in data:
        return "auto"
    level = data["reasoning_level"]
    if type(level) is not str:
        raise TypeError("AceAI config reasoning_level must be str")
    if level not in ("auto", "low", "medium", "high", "max"):
        raise ValueError("AceAI config reasoning_level is unsupported")
    return level
