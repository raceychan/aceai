from pathlib import Path

import pytest

from aceai.agent.ace_agent import ACE_AGENT_SKILL_PATH, build_ace_agent
from aceai.agent.features import default_agent_tools
from aceai.agent.features.tools import read_text_file
from aceai.core import ToolExecutionError, ToolExecutor
from aceai.llm.interface import UNSET


def write_skill(root: Path, name: str, description: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n"
        f"# {name}\n",
        encoding="utf-8",
    )


def test_default_agent_tools_are_product_capabilities() -> None:
    tools = default_agent_tools()
    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "list_directory",
        "read_text_file",
        "write_text_file",
        "replace_text_in_file",
        "preview_patch",
        "apply_patch",
        "git_status",
        "git_diff",
        "run_shell_command",
        "search_text",
    }
    approval_policies = {tool.name: tool.metadata.approval_policy for tool in tools}
    assert approval_policies["write_text_file"] == "filesystem_write"
    assert approval_policies["replace_text_in_file"] == "filesystem_write"
    assert approval_policies["apply_patch"] == "filesystem_patch"
    assert approval_policies["run_shell_command"] == "shell_command"
    assert {tool.name for tool in tools if tool.metadata.require_approval} == {
        "write_text_file",
        "replace_text_in_file",
        "apply_patch",
        "run_shell_command",
    }


def test_default_agent_tools_apply_permission_overrides() -> None:
    default_tool_map = {tool.name: tool for tool in default_agent_tools()}
    tools = default_agent_tools(
        {
            "read_text_file": "ask",
            "run_shell_command": "always",
            "apply_patch": "never",
        }
    )
    tool_names = {tool.name for tool in tools}
    approval_required = {tool.name for tool in tools if tool.metadata.require_approval}

    assert "apply_patch" not in tool_names
    assert "read_text_file" in approval_required
    assert "run_shell_command" not in approval_required
    assert default_tool_map["run_shell_command"].metadata.require_approval


def test_build_ace_agent_wires_app_tools_and_project_skills(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    write_skill(project / ".agent" / "skills", "aceai-release", "Release workflow.")
    monkeypatch.chdir(project)

    agent = build_ace_agent(api_key="test-key", model="gpt-5.5")

    assert agent.default_model == "gpt-5.5"
    assert agent.max_steps is UNSET
    assert ACE_AGENT_SKILL_PATH == "auto"
    assert isinstance(agent._executor, ToolExecutor)
    assert agent._hosted_tools[0].provider_name == "openai"
    assert agent._hosted_tools[0].native_name == "web_search"
    assert set(agent._executor.tools) >= {
        "list_directory",
        "read_text_file",
        "write_text_file",
        "replace_text_in_file",
        "preview_patch",
        "apply_patch",
        "git_status",
        "git_diff",
        "run_shell_command",
        "search_text",
        "skills_list",
        "skill_view",
    }
    assert set(agent.skill_registry.skills) == {"aceai-release"}


def test_build_ace_agent_supports_deepseek_without_openai_hosted_tools(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    agent = build_ace_agent(
        provider_name="deepseek",
        api_key="test-key",
        model="deepseek-v4-flash",
    )

    assert agent.default_model == "deepseek-v4-flash"
    assert agent._hosted_tools == []


def test_app_file_tool_reports_missing_file_as_tool_execution_error(tmp_path) -> None:
    missing_path = tmp_path / "missing.py"

    with pytest.raises(ToolExecutionError, match="No such file or directory"):
        read_text_file(path=str(missing_path))
