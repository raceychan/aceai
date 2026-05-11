from pathlib import Path
from typing import Literal

from ididi import Graph
from openai import AsyncOpenAI

from aceai.core import Agent, Executor
from aceai.core.context_manager import DEFAULT_CONTEXT_WINDOW_TOKENS, CompressThreshold
from aceai.llm.interface import UNSET, Unset, is_set
from aceai.llm.anthropic import (
    ANTHROPIC_OAUTH_PROVIDER_NAME,
    ANTHROPIC_PROVIDER_NAME,
    Anthropic,
)
from aceai.llm.deepseek import DeepSeek
from aceai.llm.models import LLMHostedToolSpec
from aceai.llm.openai import OpenAI, OpenAIModel
from aceai.llm.openai_codex import OpenAICodex
from aceai.llm.service import LLMService

from .features import (
    build_background_subagent_tools,
    build_delegate_to_subagent_tool,
    default_agent_tools,
)
from .permissions import ToolPermission
from .provider_auth import resolve_provider_api_key
from .provider_catalog import context_window_for_model


ACE_AGENT_SYSTEM_PROMPT = """
You are AceAI, a concise and capable agent app.

You can inspect files, edit files, search code, run shell commands, search the
web when current external information is needed, and use skills when they match
the user's task. Prefer concrete action over abstract advice.
When working in a repository, inspect the real files before changing behavior and
run the repository's tests after meaningful code changes.
"""

ACE_AGENT_SKILL_PATH = "auto"
ACE_AGENT_BUILTIN_SKILL_PATHS = (Path(__file__).parent / "builtin_skills",)
ACE_AGENT_BUILTIN_SKILL_NAMES = ("skill-creator",)
ACE_AGENT_CODEX_INSTRUCTIONS = "You are AceAI, a concise coding agent."
ACE_AGENT_API_TIMEOUT_SECONDS = 300.0
ACE_AGENT_STREAM_START_TIMEOUT_SECONDS = 120.0
ACE_AGENT_STREAM_EVENT_TIMEOUT_SECONDS = 60.0
ACE_AGENT_HOSTED_TOOLS = [
    LLMHostedToolSpec(
        provider_name="openai",
        native_name="web_search",
    )
]


def build_ace_agent(
    *,
    api_key: str,
    model: OpenAIModel,
    provider_name: str = "openai",
    hosted_tools: list[LLMHostedToolSpec] | None = None,
    skill_path: str | Path | Literal["auto", "disable"] = ACE_AGENT_SKILL_PATH,
    enabled_skill_names: Unset[tuple[str, ...]] = UNSET,
    tool_permissions: dict[str, ToolPermission] | None = None,
    tool_enabled: dict[str, bool] | None = None,
    tool_max_calls: dict[str, int] | None = None,
    compress_threshold: CompressThreshold = "100%",
    api_timeout_seconds: float = ACE_AGENT_API_TIMEOUT_SECONDS,
    stream_start_timeout_seconds: float = ACE_AGENT_STREAM_START_TIMEOUT_SECONDS,
    stream_event_timeout_seconds: float = ACE_AGENT_STREAM_EVENT_TIMEOUT_SECONDS,
) -> Agent:
    if provider_name == "openai":
        provider = OpenAI(
            client=AsyncOpenAI(api_key=api_key),
            default_meta={"model": model},
        )
    elif provider_name == "codex":
        provider = OpenAICodex(
            api_key=resolve_provider_api_key(provider_name, api_key),
            default_meta={"model": model},
            instructions=ACE_AGENT_CODEX_INSTRUCTIONS,
        )
    elif provider_name == "deepseek":
        provider = DeepSeek(
            api_key=api_key,
            default_meta={"model": model},
        )
    elif provider_name == ANTHROPIC_PROVIDER_NAME:
        provider = Anthropic(
            api_key=api_key,
            default_meta={"model": model},
        )
    elif provider_name == ANTHROPIC_OAUTH_PROVIDER_NAME:
        provider = Anthropic(
            api_key=resolve_provider_api_key(provider_name, api_key),
            default_meta={"model": model},
            provider_name=ANTHROPIC_OAUTH_PROVIDER_NAME,
            auth_mode="oauth",
        )
    else:
        raise ValueError("Unsupported provider")
    llm_service = LLMService(
        [provider],
        timeout_seconds=api_timeout_seconds,
        stream_start_timeout_seconds=stream_start_timeout_seconds,
        stream_event_timeout_seconds=stream_event_timeout_seconds,
    )
    context_window_tokens = context_window_for_model(provider_name, model)
    if context_window_tokens is None:
        context_window_tokens = DEFAULT_CONTEXT_WINDOW_TOKENS
    if hosted_tools is None and provider_name in ("openai", "codex"):
        selected_hosted_tools = ACE_AGENT_HOSTED_TOOLS
    elif hosted_tools is None:
        selected_hosted_tools = []
    else:
        selected_hosted_tools = hosted_tools
    app_tools = default_agent_tools(
        tool_permissions=tool_permissions,
        tool_enabled=tool_enabled,
        tool_max_calls=tool_max_calls,
    )
    app_tools.append(
        build_delegate_to_subagent_tool(
            llm_service=llm_service,
            default_model=model,
            available_tools=app_tools,
            available_hosted_tools=selected_hosted_tools,
            compress_threshold=compress_threshold,
            context_window_tokens=context_window_tokens,
        )
    )
    app_tools.extend(
        build_background_subagent_tools(
            llm_service=llm_service,
            default_model=model,
            available_tools=app_tools,
            available_hosted_tools=selected_hosted_tools,
            compress_threshold=compress_threshold,
            context_window_tokens=context_window_tokens,
        )
    )
    executor = Executor(
        Graph(),
        app_tools,
        skill_path=skill_path,
        enabled_skill_names=_enabled_skill_names_with_builtin_defaults(
            skill_path=skill_path,
            enabled_skill_names=enabled_skill_names,
        ),
        extra_skill_paths=ACE_AGENT_BUILTIN_SKILL_PATHS,
        hosted_tools=selected_hosted_tools,
    )
    return Agent(
        prompt=ACE_AGENT_SYSTEM_PROMPT,
        default_model=model,
        llm_service=llm_service,
        executor=executor,
        compress_threshold=compress_threshold,
        context_window_tokens=context_window_tokens,
    )


def _enabled_skill_names_with_builtin_defaults(
    *,
    skill_path: str | Path | Literal["auto", "disable"],
    enabled_skill_names: Unset[tuple[str, ...]],
) -> Unset[tuple[str, ...]]:
    if skill_path == "disable" or not is_set(enabled_skill_names):
        return enabled_skill_names
    names = list(enabled_skill_names)
    for builtin_name in ACE_AGENT_BUILTIN_SKILL_NAMES:
        if builtin_name not in names:
            names.append(builtin_name)
    return tuple(names)
