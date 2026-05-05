from pathlib import Path
from typing import Literal

from ididi import Graph
from openai import AsyncOpenAI

from aceai.core import AgentBase, ToolExecutor
from aceai.llm.interface import UNSET, Unset
from aceai.llm.deepseek import DeepSeek
from aceai.llm.models import LLMHostedToolSpec
from aceai.llm.openai import OpenAI, OpenAIModel
from aceai.llm.service import LLMService

from .features import default_agent_tools


ACE_AGENT_SYSTEM_PROMPT = """
You are AceAI, a concise and capable agent app.

You can inspect files, edit files, search code, run shell commands, search the
web when current external information is needed, and use skills when they match
the user's task. Prefer concrete action over abstract advice.
When working in a repository, inspect the real files before changing behavior and
run the repository's tests after meaningful code changes.
"""

ACE_AGENT_SKILLS_DIR = Path(__file__).parent / "features" / "skills"
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
    skill_path: str | Path | Literal["auto", "disable"] = ACE_AGENT_SKILLS_DIR,
    enabled_skill_names: Unset[tuple[str, ...]] = UNSET,
) -> AgentBase:
    if provider_name == "openai":
        provider = OpenAI(
            client=AsyncOpenAI(api_key=api_key),
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
    executor = ToolExecutor(Graph(), default_agent_tools())
    if hosted_tools is None and provider_name == "openai":
        selected_hosted_tools = ACE_AGENT_HOSTED_TOOLS
    elif hosted_tools is None:
        selected_hosted_tools = []
    else:
        selected_hosted_tools = hosted_tools
    return AgentBase(
        prompt=ACE_AGENT_SYSTEM_PROMPT,
        default_model=model,
        llm_service=llm_service,
        executor=executor,
        skill_path=skill_path,
        enabled_skill_names=enabled_skill_names,
        hosted_tools=selected_hosted_tools,
    )
