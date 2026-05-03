from pathlib import Path

import pytest
from ididi import Graph

from aceai.core import AgentBase, ToolExecutor
from aceai.llm import LLMResponse
from aceai.llm.models import LLMMessage, LLMStreamEvent, LLMToolCall


class StubSkillSelectingLLMService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        call_index = len(self.calls)

        if call_index == 1:
            messages: list[LLMMessage] = request["messages"]
            tools = request["tools"]
            system_text = messages[0].content[0]["data"]
            user_text = messages[-1].content[0]["data"]
            tool_names = {tool.name for tool in tools}

            assert "prepare the release" in user_text
            assert "<available_skills>" in system_text
            assert "<name>release</name>" in system_text
            assert "Use for release workflows." in system_text
            assert "skill_view" in tool_names

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="loading release skill",
                    tool_calls=[
                        LLMToolCall(
                            name="skill_view",
                            arguments='{"name":"release"}',
                            call_id="skill-call-1",
                        )
                    ],
                ),
            )
            return

        if call_index == 2:
            messages = request["messages"]
            tool_messages = [message for message in messages if message.role == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0].name == "skill_view"
            assert "Always run the release checklist." in tool_messages[0].content[0][
                "data"
            ]

            yield LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="Release skill loaded and applied."),
            )
            return

        raise AssertionError("Unexpected extra LLM call")

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("AgentBase should not call complete() in streaming mode")


def write_skill(root: Path) -> Path:
    skill_dir = root / "release"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: release",
                "description: Use for release workflows.",
                "---",
                "# Release",
                "Always run the release checklist.",
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


@pytest.mark.anyio
async def test_agent_loads_skill_after_model_requests_skill_view(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    write_skill(skills_root)
    llm_service = StubSkillSelectingLLMService()
    executor = ToolExecutor(Graph(), [])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        skill_path=skills_root,
        max_steps=2,
    )

    answer = await agent.ask("please prepare the release")

    assert answer == "Release skill loaded and applied."
    assert len(llm_service.calls) == 2
    assert executor.tools["skill_view"].name == "skill_view"
