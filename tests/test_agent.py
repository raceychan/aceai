from aceai.agent import AgentBase


def test_build_agent_base() -> None:
    agent = AgentBase(
        prompt="You are a helpful assistant.",
        default_model="gpt-4",
        llm_service=None,  # type: ignore
        executor=None,  # type: ignore
    )
    assert agent.prompt == "You are a helpful assistant."
    assert agent.default_model == "gpt-4"
    assert agent.max_turns == 5
    assert AgentBase.agent_registry["AgentBase"] is agent