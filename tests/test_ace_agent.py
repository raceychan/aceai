import pytest

from aceai.agent.ace_agent import ACE_AGENT_SKILLS_DIR, build_ace_agent
from aceai.agent.features import default_agent_tools
from aceai.agent.features.tools import read_text_file
from aceai.core import ToolExecutionError, ToolExecutor
from aceai.llm.interface import UNSET


def test_default_agent_tools_are_product_capabilities() -> None:
    tool_names = {tool.name for tool in default_agent_tools()}

    assert tool_names == {
        "list_directory",
        "read_text_file",
        "write_text_file",
        "replace_text_in_file",
        "run_shell_command",
        "search_text",
    }


def test_build_ace_agent_wires_app_tools_and_builtin_skills(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    agent = build_ace_agent(api_key="test-key", model="gpt-5.5")

    assert agent.default_model == "gpt-5.5"
    assert agent.max_steps is UNSET
    assert ACE_AGENT_SKILLS_DIR.exists()
    assert isinstance(agent._executor, ToolExecutor)
    assert agent._hosted_tools[0].provider_name == "openai"
    assert agent._hosted_tools[0].native_name == "web_search"
    assert set(agent._executor.tools) >= {
        "list_directory",
        "read_text_file",
        "write_text_file",
        "replace_text_in_file",
        "run_shell_command",
        "search_text",
        "skills_list",
        "skill_view",
    }
    assert "developer" in agent.skill_registry.skills


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
