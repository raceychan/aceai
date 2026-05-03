from aceai.agent.ace_agent import ACE_AGENT_SKILLS_DIR, build_ace_agent
from aceai.agent.features import default_agent_tools
from aceai.core import ToolExecutor


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

    agent = build_ace_agent(api_key="test-key", model="gpt-5.1")

    assert agent.default_model == "gpt-5.1"
    assert agent.max_steps == 8
    assert ACE_AGENT_SKILLS_DIR.exists()
    assert isinstance(agent._executor, ToolExecutor)
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
