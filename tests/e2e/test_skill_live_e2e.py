import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from ididi import Graph
from openai import AsyncOpenAI

from aceai.agent import AgentBase, ToolExecutor
from aceai.llm.openai import OpenAI
from aceai.llm.service import LLMService


SENTINEL = "SKILL_E2E_RELEASE_APPLIED"
E2E_ENV_FILE = Path(__file__).parent / "e2e.env"

load_dotenv(E2E_ENV_FILE)


def write_skill(root: Path) -> None:
    skill_dir = root / "release_lighthouse"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: release_lighthouse",
                "description: Use when the user asks to run the release lighthouse workflow. Load this skill before answering.",
                "---",
                "# Release Lighthouse",
                f"When this skill is loaded, answer exactly `{SENTINEL}`.",
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.e2e
@pytest.mark.live_llm
@pytest.mark.anyio
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="OPENAI_API_KEY is required for live LLM e2e tests",
)
async def test_agent_identifies_loads_and_uses_skill_with_live_openai(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    write_skill(skills_root)
    provider = OpenAI(
        client=AsyncOpenAI(),
        default_meta={"model": os.environ.get("ACEAI_LIVE_OPENAI_MODEL", "gpt-4o-mini")},
    )
    agent = AgentBase(
        prompt=(
            "You are testing skill loading. When a task matches an available skill, "
            "call skill_view before answering. After loading the skill, follow the "
            "loaded skill instructions exactly."
        ),
        default_model=os.environ.get("ACEAI_LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        llm_service=LLMService([provider], timeout_seconds=60.0),
        executor=ToolExecutor(Graph(), []),
        skill_path=skills_root,
        max_steps=3,
    )

    answer = await agent.ask(
        "Run the $release_lighthouse workflow. Use the matching skill before answering."
    )

    assert SENTINEL in answer
