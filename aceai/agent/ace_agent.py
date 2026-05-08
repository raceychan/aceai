from pathlib import Path
from typing import Literal

from ididi import Graph
from openai import AsyncOpenAI

from aceai.core import Agent, Executor
from aceai.core.context_manager import CompressThreshold
from aceai.llm.interface import UNSET, Unset, is_set
from aceai.llm.deepseek import DeepSeek
from aceai.llm.models import LLMHostedToolSpec
from aceai.llm.openai import OpenAI, OpenAIModel
from aceai.llm.openai_codex import OpenAICodex
from aceai.llm.service import LLMService

from .features import build_delegate_to_subagent_tool, default_agent_tools
from .permissions import ToolPermission


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
) -> Agent:
    if provider_name == "openai":
        provider = OpenAI(
            client=AsyncOpenAI(api_key=api_key),
            default_meta={"model": model},
        )
    elif provider_name == "codex":
        provider = OpenAICodex(
            api_key=api_key,
            default_meta={"model": model},
        )
    elif provider_name == "deepseek":
        provider = DeepSeek(
            api_key=api_key,
            default_meta={"model": model},
        )
    else:
        raise ValueError("Unsupported provider")
    llm_service = LLMService([provider], timeout_seconds=120.0)
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
        )
    )
    if hosted_tools is None and provider_name == "openai":
        selected_hosted_tools = ACE_AGENT_HOSTED_TOOLS
    elif hosted_tools is None:
        selected_hosted_tools = []
    else:
        selected_hosted_tools = hosted_tools
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
