from pathlib import Path

from ididi import Graph
import pytest

from aceai.core.executor import RunState, ToolExecutor
from aceai.core.base import AgentBase, ToolExecutionFailure
from aceai.llm.errors import AceAIConfigurationError
from aceai.llm.models import LLMToolCall


def write_skill(root: Path, name: str, description: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


def test_build_agent_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent = AgentBase(
        prompt="You are a helpful assistant.",
        default_model="gpt-4",
        llm_service=None,  # type: ignore
        executor=None,  # type: ignore
        skill_path=tmp_path / "empty-skills",
    )
    assert agent.system_message.content[0]["data"] == "You are a helpful assistant."
    assert agent._default_model == "gpt-4"
    assert agent.max_steps == 5
    assert hasattr(agent, "run")


def test_agent_base_add_instruction_updates_system_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent = AgentBase(
        prompt="Initial",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
        skill_path=tmp_path / "empty-skills",
    )
    agent.add_instruction(" + More")
    agent.add_instruction(" + More")
    assert agent.system_message.content[0]["data"] == "Initial + More"


def test_agent_base_add_instruction_rejects_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent = AgentBase(
        prompt="Initial",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
        skill_path=tmp_path / "empty-skills",
    )
    with pytest.raises(ValueError, match="Empty Instruction"):
        agent.add_instruction("")


def test_agent_base_auto_loads_global_and_project_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    write_skill(home / ".aceai" / "skills", "release", "Release workflow.", "# Release")
    write_skill(cwd / ".agent" / "skills", "review", "Review workflow.", "# Review")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=None,
    )

    assert set(agent.skill_registry.skills) == {"release", "review"}
    system_text = agent.system_message.content[0]["data"]
    assert "<available_skills>" in system_text
    assert "<name>release</name>" in system_text
    assert "<name>review</name>" in system_text


def test_agent_base_disable_skill_path_skips_all_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    write_skill(home / ".aceai" / "skills", "global", "Global skill.", "# Global")
    write_skill(cwd / ".agent" / "skills", "project", "Project skill.", "# Project")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    executor = ToolExecutor(Graph(), [])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=executor,
        skill_path="disable",
    )

    assert agent.skill_registry.skills == {}
    assert agent.system_message.content[0]["data"] == "Prompt"
    assert "skills_list" not in executor.tools
    assert "skill_view" not in executor.tools


def test_agent_base_explicit_skill_path_skips_project_auto_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    explicit = tmp_path / "explicit-skills"
    write_skill(home / ".aceai" / "skills", "global", "Global skill.", "# Global")
    write_skill(cwd / ".agent" / "skills", "project", "Project skill.", "# Project")
    write_skill(explicit, "explicit", "Explicit skill.", "# Explicit")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=None,
        skill_path=explicit,
    )

    assert set(agent.skill_registry.skills) == {"global", "explicit"}


@pytest.mark.anyio
async def test_agent_base_registers_skill_tools_on_tool_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    explicit = tmp_path / "skills"
    write_skill(explicit, "release", "Release workflow.", "# Release\nDo release work.")
    monkeypatch.setenv("HOME", str(home))
    executor = ToolExecutor(Graph(), [])

    AgentBase(
        prompt="Prompt",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=executor,
        skill_path=explicit,
    )

    assert "skills_list" in executor.tools
    assert "skill_view" in executor.tools
    result = await executor.execute_tool(
        LLMToolCall(
            name="skill_view",
            arguments='{"name":"release"}',
            call_id="call-skill",
        ),
        run_state=RunState(),
    )
    assert "Do release work." in result


def test_agent_base_requires_positive_max_steps() -> None:
    with pytest.raises(AceAIConfigurationError):
        AgentBase(
            prompt="Prompt",
            default_model="gpt-4",
            llm_service=None,  # type: ignore[arg-type]
            executor=None,  # type: ignore[arg-type]
            max_steps=0,
        )


def test_agent_base_rejects_chunk_size_argument() -> None:
    with pytest.raises(TypeError):
        AgentBase(
            prompt="Prompt",
            default_model="gpt-4",
            llm_service=None,  # type: ignore[arg-type]
            executor=None,  # type: ignore[arg-type]
            delta_chunk_size=1,
        )


def test_agent_base_rejects_reasoning_log_argument() -> None:
    with pytest.raises(TypeError):
        AgentBase(
            prompt="Prompt",
            default_model="gpt-4",
            llm_service=None,  # type: ignore[arg-type]
            executor=None,  # type: ignore[arg-type]
            reasoning_log_max_chars=10,
        )


def test_tool_execution_failure_preserves_original_error() -> None:
    tool_call = LLMToolCall(name="calc", arguments="{}", call_id="call-1")
    cause = ValueError("explode")

    failure = ToolExecutionFailure(tool_call=tool_call, error=cause)

    assert str(failure) == "explode"
    assert failure.tool_call is tool_call
    assert failure.original_error is cause
