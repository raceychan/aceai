import json
from pathlib import Path

import pytest

from aceai.agent.ace_agent import ACE_AGENT_SKILL_PATH, build_ace_agent
from aceai.agent.features import default_agent_tools
from aceai.agent.features.tools import read_text_file
from aceai.core import ToolExecutionError, Executor
from aceai.llm.interface import UNSET, is_present
from aceai.llm.openai_codex import CODEX_CLI_AUTH_SENTINEL


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
    assert not {
        tool.name for tool in tools if is_present(tool.metadata.max_calls_per_run)
    }


def test_default_agent_tools_apply_permission_overrides() -> None:
    default_tool_map = {tool.name: tool for tool in default_agent_tools()}
    tools = default_agent_tools(
        tool_permissions={
            "read_text_file": "ask",
            "run_shell_command": "always",
        },
        tool_enabled={"apply_patch": False},
    )
    tool_names = {tool.name for tool in tools}
    approval_required = {tool.name for tool in tools if tool.metadata.require_approval}

    assert "apply_patch" not in tool_names
    assert "read_text_file" in approval_required
    assert "run_shell_command" not in approval_required
    assert default_tool_map["run_shell_command"].metadata.require_approval


def test_default_agent_tools_apply_max_call_overrides() -> None:
    tools = default_agent_tools(
        tool_permissions={"run_shell_command": "always"},
        tool_max_calls={"run_shell_command": 3},
    )
    tool_map = {tool.name: tool for tool in tools}

    assert tool_map["run_shell_command"].metadata.max_calls_per_run == 3
    assert not tool_map["run_shell_command"].metadata.require_approval


def test_build_ace_agent_wires_app_tools_and_project_skills(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    write_skill(project / ".agents" / "skills", "aceai-release", "Release workflow.")
    monkeypatch.chdir(project)

    agent = build_ace_agent(api_key="test-key", model="gpt-5.5")

    assert agent.default_model == "gpt-5.5"
    assert agent.max_steps is UNSET
    assert ACE_AGENT_SKILL_PATH == "auto"
    assert isinstance(agent._executor, Executor)
    assert agent._executor.hosted_tools[0].provider_name == "openai"
    assert agent._executor.hosted_tools[0].native_name == "web_search"
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
    assert "delegate_to_subagent" in agent._executor.tools
    assert set(agent.skill_registry.skills) == {"aceai-release", "skill-creator"}


def test_build_ace_agent_wires_delegation_tool(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    agent = build_ace_agent(
        api_key="test-key",
        model="gpt-5.5",
    )

    assert "delegate_to_subagent" in agent._executor.tools
    assert "skill-creator" in agent.skill_registry.skills
    assert agent._executor.tools["delegate_to_subagent"].metadata.tags == [
        "agent_app",
        "delegation",
    ]
    assert (
        "Delegate a bounded, independent task to a subagent"
        in agent._executor.tools["delegate_to_subagent"].metadata.description
    )


def test_build_ace_agent_keeps_builtin_skills_enabled_in_selected_mode(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    write_skill(project / ".agents" / "skills", "aceai-release", "Release workflow.")
    monkeypatch.chdir(project)

    agent = build_ace_agent(
        api_key="test-key",
        model="gpt-5.5",
        enabled_skill_names=("aceai-release",),
    )

    assert set(agent.skill_registry.skills) == {"aceai-release", "skill-creator"}


def test_build_ace_agent_excludes_disabled_app_tools(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    agent = build_ace_agent(
        api_key="test-key",
        model="gpt-5.5",
        tool_enabled={"run_shell_command": False},
    )

    assert "run_shell_command" not in agent._executor.tools
    assert "read_text_file" in agent._executor.tools


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
    assert agent._executor.hosted_tools == []


def test_build_ace_agent_supports_openai_codex_without_hosted_tools(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    agent = build_ace_agent(
        provider_name="codex",
        api_key="codex-access-token",
        model="gpt-5.4",
    )

    assert agent.default_model == "gpt-5.4"
    assert agent._executor.hosted_tools == []


def test_build_ace_agent_can_use_codex_cli_auth_sentinel(
    tmp_path, monkeypatch
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "codex-access-token",
                    "refresh_token": "codex-refresh-token",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    agent = build_ace_agent(
        provider_name="codex",
        api_key=CODEX_CLI_AUTH_SENTINEL,
        model="gpt-5.4",
    )

    assert agent.default_model == "gpt-5.4"
    assert agent._executor.hosted_tools == []


def test_app_file_tool_reports_missing_file_as_tool_execution_error(tmp_path) -> None:
    missing_path = tmp_path / "missing.py"

    with pytest.raises(ToolExecutionError, match="No such file or directory"):
        read_text_file(path=str(missing_path))
