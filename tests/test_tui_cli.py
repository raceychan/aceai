from aceai.agent.tui import cli


class StubAgent:
    pass


def test_cli_question_runs_single_question_tui(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def run_agent_tui(received_agent, question):
        calls.append(("single", (received_agent, question)))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(cli, "run_agent_tui", run_agent_tui)

    cli.main(["hello", "world"])

    assert calls == [
        ("build", ("key", "gpt-5.1")),
        ("single", (agent, "hello world")),
    ]


def test_cli_without_question_runs_interactive_tui(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def run_interactive_tui(received_agent):
        calls.append(("interactive", received_agent))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("ACEAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main([])

    assert calls == [
        ("build", ("key", "gpt-4o-mini")),
        ("interactive", agent),
    ]


def test_cli_without_config_opens_provider_setup_tui(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, object | None, str, str]]] = []

    def run_configured_tui(
        agent_factory,
        *,
        initial_config,
        initial_question,
        default_model,
    ):
        calls.append(
            (
                "configured",
                (agent_factory, initial_config, initial_question, default_model),
            )
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda: None)
    monkeypatch.setattr(cli, "run_configured_tui", run_configured_tui)

    cli.main(["hello"])

    assert calls[0][0] == "configured"
    payload = calls[0][1]
    assert payload[1] is None
    assert payload[2] == "hello"
    assert payload[3] == "gpt-5.1"
