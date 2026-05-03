import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from ididi import Graph
from openai import AsyncOpenAI

from aceai.core import AgentBase, ToolExecutor
from aceai.core.events import RunCompletedEvent, ToolCompletedEvent
from aceai.llm import LLMResponse
from aceai.llm.models import LLMMessage, LLMStreamEvent, LLMToolCall
from aceai.llm.openai import OpenAI
from aceai.llm.service import LLMService


SENTINEL = "SKILL_CREATOR_DEEP_E2E_COMPLETE"
E2E_ENV_FILE = Path(__file__).parent / "e2e.env"
MATURE_SKILL_SOURCE = "https://github.com/anthropics/skills/tree/main/skills/skill-creator"

load_dotenv(E2E_ENV_FILE)


class DeepSkillWorkflowLLMService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        call_index = len(self.calls)
        messages: list[LLMMessage] = request["messages"]

        if call_index == 1:
            tools = request["tools"]
            system_text = messages[0].content[0]["data"]
            user_text = messages[-1].content[0]["data"]
            tool_names = {tool.name for tool in tools}

            assert "$skill-lab" in user_text
            assert "<available_skills>" in system_text
            assert "<name>skill-lab</name>" in system_text
            assert "skill_view" in tool_names
            assert "skills_list" in tool_names

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="discovering available skills",
                    tool_calls=[
                        LLMToolCall(
                            name="skills_list",
                            arguments="{}",
                            call_id="skills-list-1",
                        )
                    ],
                ),
            )
            return

        if call_index == 2:
            tool_text = self._last_tool_text(messages, "skills_list")
            assert "skill-lab" in tool_text
            assert "progressive disclosure" in tool_text

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="loading entry instructions",
                    tool_calls=[
                        LLMToolCall(
                            name="skill_view",
                            arguments='{"name":"skill-lab"}',
                            call_id="skill-entry-1",
                        )
                    ],
                ),
            )
            return

        if call_index == 3:
            tool_text = self._last_tool_text(messages, "skill_view")
            assert "Inspired by Anthropic's skill-creator skill" in tool_text
            assert "references/eval-schema.md" in tool_text
            assert "scripts/aggregate_benchmark.py" in tool_text
            assert "assets/report-template.md" in tool_text

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="loading eval schema",
                    tool_calls=[
                        LLMToolCall(
                            name="skill_view",
                            arguments=(
                                '{"name":"skill-lab",'
                                '"file_path":"references/eval-schema.md"}'
                            ),
                            call_id="skill-reference-1",
                        )
                    ],
                ),
            )
            return

        if call_index == 4:
            tool_text = self._last_tool_text(messages, "skill_view")
            assert "expectation_pass_rate" in tool_text
            assert "evals[].expectations" in tool_text

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="loading benchmark script",
                    tool_calls=[
                        LLMToolCall(
                            name="skill_view",
                            arguments=(
                                '{"name":"skill-lab",'
                                '"file_path":"scripts/aggregate_benchmark.py"}'
                            ),
                            call_id="skill-script-1",
                        )
                    ],
                ),
            )
            return

        if call_index == 5:
            tool_text = self._last_tool_text(messages, "skill_view")
            assert "def calculate_pass_rate" in tool_text
            assert "statistics.mean" in tool_text

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="loading report template",
                    tool_calls=[
                        LLMToolCall(
                            name="skill_view",
                            arguments=(
                                '{"name":"skill-lab",'
                                '"file_path":"assets/report-template.md"}'
                            ),
                            call_id="skill-asset-1",
                        )
                    ],
                ),
            )
            return

        if call_index == 6:
            tool_text = self._last_tool_text(messages, "skill_view")
            assert "## Benchmark summary" in tool_text
            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text=(
                        f"{SENTINEL}\n"
                        "loaded: entry, references/eval-schema.md, "
                        "scripts/aggregate_benchmark.py, assets/report-template.md\n"
                        "verdict: progressive disclosure resources were reachable"
                    ),
                ),
            )
            return

        raise AssertionError("Unexpected extra LLM call")

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("AgentBase should not call complete() in streaming mode")

    def _last_tool_text(self, messages: list[LLMMessage], tool_name: str) -> str:
        tool_messages = [
            message
            for message in messages
            if message.role == "tool" and message.name == tool_name
        ]
        assert tool_messages
        return tool_messages[-1].content[0]["data"]


def write_skill_creator_style_skill(root: Path) -> None:
    skill_dir = root / "skill-lab"
    skill_dir.mkdir(parents=True)
    (skill_dir / "references").mkdir()
    (skill_dir / "scripts").mkdir()
    (skill_dir / "assets").mkdir()
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: skill-lab",
                "description: Use when asked to design, audit, benchmark, or improve an agent skill with progressive disclosure, eval schemas, benchmark aggregation, or skill report templates.",
                "---",
                "# Skill Lab",
                "",
                "Inspired by Anthropic's skill-creator skill, this skill tests a",
                "progressive disclosure workflow rather than a single loaded prompt.",
                f"Source pattern: {MATURE_SKILL_SOURCE}",
                "",
                "For a deep skill audit:",
                "1. Read `references/eval-schema.md` before describing eval quality.",
                "2. Read `scripts/aggregate_benchmark.py` before discussing metrics.",
                "3. Read `assets/report-template.md` before writing the final report.",
                "",
                "Final answers should mention the loaded resources and include the",
                f"sentinel `{SENTINEL}`.",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "references" / "eval-schema.md").write_text(
        "\n".join(
            [
                "# Eval Schema",
                "",
                "A skill eval contains `skill_name`, `evals[].prompt`,",
                "`evals[].expected_output`, and `evals[].expectations`.",
                "",
                "A benchmark summary contains `expectation_pass_rate`,",
                "`duration_seconds`, `token_count`, and `loaded_resources`.",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "aggregate_benchmark.py").write_text(
        "\n".join(
            [
                "import statistics",
                "",
                "",
                "def calculate_pass_rate(passed: int, total: int) -> float:",
                "    return passed / total",
                "",
                "",
                "def summarize_rates(rates: list[float]) -> dict[str, float]:",
                "    return {",
                '        "mean": statistics.mean(rates),',
                '        "min": min(rates),',
                '        "max": max(rates),',
                "    }",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "assets" / "report-template.md").write_text(
        "\n".join(
            [
                "# Skill audit report",
                "",
                "## Benchmark summary",
                "",
                "## Resource loading evidence",
                "",
                "## Recommended next eval",
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.e2e
@pytest.mark.anyio
async def test_agent_runs_deep_progressive_disclosure_skill_e2e(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    write_skill_creator_style_skill(skills_root)
    llm_service = DeepSkillWorkflowLLMService()
    executor = ToolExecutor(Graph(), [])
    agent = AgentBase(
        prompt=(
            "You are testing a mature agent skill loading workflow. Use tools across "
            "multiple reasoning steps. Start with skills_list, then load the matching "
            "skill and every supporting file that the skill tells you to inspect."
        ),
        default_model="test-model",
        llm_service=llm_service,
        executor=executor,
        skill_path=skills_root,
        max_steps=6,
    )

    answer = await agent.ask(
        "Run a deep audit with $skill-lab and verify progressive disclosure works."
    )

    assert SENTINEL in answer
    assert "references/eval-schema.md" in answer
    assert "scripts/aggregate_benchmark.py" in answer
    assert "assets/report-template.md" in answer
    assert len(llm_service.calls) == 6


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
    write_skill_creator_style_skill(skills_root)
    provider = OpenAI(
        client=AsyncOpenAI(),
        default_meta={"model": os.environ.get("ACEAI_LIVE_OPENAI_MODEL", "gpt-4o-mini")},
    )
    agent = AgentBase(
        prompt=(
            "You are testing mature skill loading. When a task matches an available "
            "skill, call skill_view before answering. Follow progressive disclosure: "
            "if the skill points to supporting references, scripts, or assets, load "
            "those exact files with skill_view before the final answer. The final "
            "answer must be concise and include the sentinel exactly."
        ),
        default_model=os.environ.get("ACEAI_LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        llm_service=LLMService([provider], timeout_seconds=60.0),
        executor=ToolExecutor(Graph(), []),
        skill_path=skills_root,
        max_steps=6,
    )

    answer = ""
    tool_file_paths: list[str] = []
    async for event in agent.run(
        "Run a deep audit with $skill-lab. Load the skill, then load its referenced "
        "eval schema, benchmark script, and report template before answering."
    ):
        if isinstance(event, ToolCompletedEvent) and event.tool_name == "skill_view":
            arguments = json.loads(event.tool_call.arguments)
            if "file_path" in arguments:
                tool_file_paths.append(arguments["file_path"])
        elif isinstance(event, RunCompletedEvent):
            answer = event.final_answer

    assert SENTINEL in answer
    assert tool_file_paths[-3:] == [
        "references/eval-schema.md",
        "scripts/aggregate_benchmark.py",
        "assets/report-template.md",
    ]
