"""Agent app configuration schema, singleton state, and persistence."""

import fcntl
import inspect
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Literal, TypedDict

import yaml
from msgspec import field

from aceai.agent.provider_catalog import (
    api_key_env,
    default_model,
    model_options,
    reasoning_effort_options,
    stale_default_models,
    supported_models,
    supported_provider_names,
)
from aceai.agent.provider_auth import default_api_key_for_provider
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
_PROCESS_START_HOME = Path.home()


class ConfigAuditState(TypedDict):
    provider: object
    model: object
    default_model: object
    skills: object
    skill_selection_mode: object
    enabled_skills: object
    disabled_providers: object
    api_key_providers: list[str]
    tool_permissions: object
    tool_enabled: object
    tool_max_calls: object
    compress_threshold: object
    reasoning_level: object


class ConfigAuditEntry(Record, kw_only=True):
    timestamp: str
    actor: str
    pid: int
    cwd: str
    target: str
    caller: tuple[str, ...]
    changed_fields: tuple[str, ...]
    before: ConfigAuditState | None
    after: ConfigAuditState


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
    disabled_providers: list[str] = field(default_factory=list[str])


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
            name="disabled_providers",
            value_type="list",
            required=True,
            description="Providers that cannot be selected or used.",
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
    for provider in config.disabled_providers:
        if type(provider) is not str:
            raise TypeError("AceAI config disabled_providers values must be str")
        if provider not in supported_provider_names():
            raise ValueError("AceAI config disabled_providers value is unsupported")
    if config.provider in config.disabled_providers:
        raise ValueError("AceAI config provider is disabled")
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


def config_audit_path() -> Path:
    return default_config_path().parent / "config.audit.jsonl"


def load_config_audit(
    *,
    limit: int = 100,
    target: Path | None = None,
) -> tuple[ConfigAuditEntry, ...]:
    audit_path = config_audit_path()
    if not audit_path.exists():
        return ()
    target_value = str(target) if target is not None else None
    entries: list[ConfigAuditEntry] = []
    with _config_file_lock(audit_path, exclusive=False):
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line == "":
            continue
        entry = _config_audit_entry_from_mapping(json.loads(line))
        if target_value is not None and entry.target != target_value:
            continue
        entries.append(entry)
    return tuple(reversed(entries[-limit:]))


def load_config(path: Path | None = None) -> AgentAppConfig | None:
    target = path if path is not None else effective_config_path()
    if not target.exists():
        clear_config()
        return None
    with _config_file_lock(target, exclusive=False):
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("AceAI config must be a mapping")
    provider = data["provider"]
    if provider not in supported_provider_names():
        raise ValueError("AceAI config provider is unsupported")
    api_keys = _load_api_keys(data)
    disabled_providers = _load_disabled_providers(data)
    configured_provider = provider
    selected_provider = _select_loadable_provider(
        provider,
        api_keys,
        disabled_providers,
    )
    if selected_provider is None:
        clear_config()
        return None
    provider = selected_provider
    api_key = _api_key_for_loadable_provider(provider, api_keys)
    api_keys[provider] = api_key
    model = data["model"] if "model" in data else default_model(provider)
    if configured_provider != provider and model not in supported_models(provider):
        model = default_model(provider)
    configured_default_model = (
        data["default_model"] if "default_model" in data else model
    )
    if (
        configured_provider != provider
        and configured_default_model not in supported_models(provider)
    ):
        configured_default_model = default_model(provider)
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
            disabled_providers=disabled_providers,
        )
    )


def save_config(config: AgentAppConfig, path: Path | None = None) -> None:
    validate_config(config)
    replace_config(config)
    target = path if path is not None else project_config_path()
    _reject_test_write_to_repo_config(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _config_file_lock(target, exclusive=True):
        before_state = _config_audit_state_from_file(target)
        after_state = _config_audit_state_from_config(config)
        payload = yaml.safe_dump(
            {
                "config_version": CONFIG_VERSION,
                "provider": config.provider,
                "api_keys": {**config.api_keys, config.provider: config.api_key},
                "disabled_providers": config.disabled_providers,
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
        if before_state != after_state:
            _append_config_audit_entry(
                target,
                before=before_state,
                after=after_state,
            )


@contextmanager
def _config_file_lock(target: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = _config_lock_path(target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _config_lock_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.lock")


def _repo_project_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".aceai" / "config.yml"


def _reject_test_write_to_repo_config(target: Path) -> None:
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return
    if target.resolve() != _repo_project_config_path().resolve():
        return
    raise RuntimeError(
        "Tests must not write the real AceAI project config; use tmp_path instead"
    )


def _config_audit_state_from_file(target: Path) -> ConfigAuditState | None:
    if not target.exists():
        return None
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("AceAI config must be a mapping")
    return _config_audit_state_from_mapping(data)


def _config_audit_state_from_config(config: AgentAppConfig) -> ConfigAuditState:
    return {
        "provider": config.provider,
        "model": config.model,
        "default_model": config.default_model,
        "skills": config.skills,
        "skill_selection_mode": config.skill_selection_mode,
        "enabled_skills": list(config.enabled_skills),
        "disabled_providers": list(config.disabled_providers),
        "api_key_providers": sorted(
            {provider for provider, api_key in config.api_keys.items() if api_key != ""}
            | {config.provider}
        ),
        "tool_permissions": dict(config.tool_permissions),
        "tool_enabled": dict(config.tool_enabled),
        "tool_max_calls": dict(config.tool_max_calls),
        "compress_threshold": config.compress_threshold,
        "reasoning_level": config.reasoning_level,
    }


def _config_audit_state_from_mapping(data: dict[object, object]) -> ConfigAuditState:
    raw_api_keys = data["api_keys"] if "api_keys" in data else {}
    if not isinstance(raw_api_keys, dict):
        raise TypeError("AceAI config api_keys must be a mapping")
    api_key_providers = sorted(
        provider
        for provider, api_key in raw_api_keys.items()
        if type(provider) is str and type(api_key) is str and api_key != ""
    )
    return {
        "provider": data["provider"] if "provider" in data else None,
        "model": data["model"] if "model" in data else None,
        "default_model": data["default_model"] if "default_model" in data else None,
        "skills": data["skills"] if "skills" in data else None,
        "skill_selection_mode": data["skill_selection_mode"]
        if "skill_selection_mode" in data
        else None,
        "enabled_skills": data["enabled_skills"] if "enabled_skills" in data else [],
        "disabled_providers": data["disabled_providers"]
        if "disabled_providers" in data
        else [],
        "api_key_providers": api_key_providers,
        "tool_permissions": data["tool_permissions"]
        if "tool_permissions" in data
        else {},
        "tool_enabled": data["tool_enabled"] if "tool_enabled" in data else {},
        "tool_max_calls": data["tool_max_calls"] if "tool_max_calls" in data else {},
        "compress_threshold": data["compress_threshold"]
        if "compress_threshold" in data
        else None,
        "reasoning_level": data["reasoning_level"] if "reasoning_level" in data else None,
    }


def _append_config_audit_entry(
    target: Path,
    *,
    before: ConfigAuditState | None,
    after: ConfigAuditState,
) -> None:
    audit_path = config_audit_path()
    _reject_test_write_to_real_home_audit(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": os.environ["USER"] if "USER" in os.environ else "",
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "target": str(target),
        "caller": _config_audit_caller(),
        "changed_fields": _config_changed_fields(before, after),
        "before": before,
        "after": after,
    }
    with _config_file_lock(audit_path, exclusive=True):
        with audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    audit_path.chmod(0o600)


def _config_changed_fields(
    before: ConfigAuditState | None,
    after: ConfigAuditState,
) -> list[str]:
    if before is None:
        return sorted(after.keys())
    return sorted(field for field in after if before[field] != after[field])


def _reject_test_write_to_real_home_audit(audit_path: Path) -> None:
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return
    real_audit_path = _PROCESS_START_HOME / ".aceai" / "config.audit.jsonl"
    if audit_path.resolve() != real_audit_path.resolve():
        return
    raise RuntimeError(
        "Tests must not write the real AceAI config audit; isolate HOME with tmp_path"
    )


def _config_audit_entry_from_mapping(data: object) -> ConfigAuditEntry:
    if not isinstance(data, dict):
        raise TypeError("AceAI config audit entry must be a mapping")
    caller = data["caller"]
    if type(caller) is not list:
        raise TypeError("AceAI config audit caller must be a list")
    changed_fields = data["changed_fields"]
    if type(changed_fields) is not list:
        raise TypeError("AceAI config audit changed_fields must be a list")
    before = data["before"]
    after = data["after"]
    if before is not None and not isinstance(before, dict):
        raise TypeError("AceAI config audit before must be a mapping")
    if not isinstance(after, dict):
        raise TypeError("AceAI config audit after must be a mapping")
    return ConfigAuditEntry(
        timestamp=data["timestamp"],
        actor=data["actor"],
        pid=data["pid"],
        cwd=data["cwd"],
        target=data["target"],
        caller=tuple(caller),
        changed_fields=tuple(changed_fields),
        before=before,
        after=after,
    )


def _config_audit_caller() -> list[str]:
    callers: list[str] = []
    for frame in inspect.stack()[2:8]:
        callers.append(f"{Path(frame.filename).name}:{frame.lineno}:{frame.function}")
    return callers


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


def _load_disabled_providers(data: dict[object, object]) -> list[str]:
    if "disabled_providers" not in data:
        return []
    raw_disabled_providers = data["disabled_providers"]
    if type(raw_disabled_providers) is not list:
        raise TypeError("AceAI config disabled_providers must be a list")
    disabled_providers: list[str] = []
    for provider in raw_disabled_providers:
        if type(provider) is not str:
            raise TypeError("AceAI config disabled_providers values must be str")
        if provider not in supported_provider_names():
            raise ValueError("AceAI config disabled_providers value is unsupported")
        disabled_providers.append(provider)
    return disabled_providers


def _select_loadable_provider(
    provider: str,
    api_keys: dict[str, str],
    disabled_providers: list[str],
) -> str | None:
    if provider in disabled_providers:
        return _first_loadable_provider(api_keys, disabled_providers)
    if _api_key_for_loadable_provider(provider, api_keys) != "":
        return provider
    return _first_loadable_provider(api_keys, disabled_providers)


def _first_loadable_provider(
    api_keys: dict[str, str],
    disabled_providers: list[str],
) -> str | None:
    for candidate in api_keys:
        if candidate in disabled_providers:
            continue
        if _api_key_for_loadable_provider(candidate, api_keys) != "":
            return candidate
    return None


def _api_key_for_loadable_provider(provider: str, api_keys: dict[str, str]) -> str:
    if provider in api_keys and api_keys[provider] != "":
        return api_keys[provider]
    env_name = api_key_env(provider)
    if env_name in os.environ:
        return os.environ[env_name]
    return default_api_key_for_provider(provider)


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
