from aceai.agent.tui import cli


class StubAgent:
    pass


class StubMetadata:
    session_id = "session-1"


def test_cli_question_runs_single_question_tui(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def run_agent_tui(received_agent, question, **kwargs):
        calls.append(("single", (received_agent, question, kwargs)))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: (object(), StubMetadata(), [], []),
    )
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: "recorder")
    monkeypatch.setattr(cli, "run_agent_tui", run_agent_tui)

    cli.main(["hello", "world"])

    assert calls == [
        ("build", ("key", "gpt-5.1")),
        (
            "single",
            (
                agent,
                "hello world",
                {
                    "initial_events": [],
                    "initial_history": [],
                    "session_recorder": "recorder",
                    "session_id": "session-1",
                },
            ),
        ),
    ]


def test_cli_without_question_runs_interactive_tui(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def run_interactive_tui(received_agent, **kwargs):
        calls.append(("interactive", (received_agent, kwargs)))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("ACEAI_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: (object(), StubMetadata(), ["event"], ["history"]),
    )
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: "recorder")
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main([])

    assert calls == [
        ("build", ("key", "gpt-4o-mini")),
        (
            "interactive",
            (
                agent,
                {
                    "initial_events": ["event"],
                    "initial_history": ["history"],
                    "session_recorder": "recorder",
                    "session_id": "session-1",
                },
            ),
        ),
    ]


def test_cli_without_config_opens_provider_setup_tui(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, object | None, str, str, dict[str, object]]]] = []

    def run_configured_tui(
        agent_factory,
        *,
        initial_config,
        initial_question,
        default_model,
        **kwargs,
    ):
        calls.append(
            (
                "configured",
                (agent_factory, initial_config, initial_question, default_model, kwargs),
            )
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda: None)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: (object(), StubMetadata(), ["event"], ["history"]),
    )
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: "recorder")
    monkeypatch.setattr(cli, "run_configured_tui", run_configured_tui)

    cli.main(["hello"])

    assert calls[0][0] == "configured"
    payload = calls[0][1]
    assert payload[1] is None
    assert payload[2] == "hello"
    assert payload[3] == "gpt-5.1"
    assert payload[4] == {
        "initial_events": ["event"],
        "initial_history": ["history"],
        "session_recorder": "recorder",
        "session_id": "session-1",
    }


def test_build_default_agent_uses_main_ace_agent(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    agent = StubAgent()

    def build_ace_agent(*, api_key, model):
        calls.append((api_key, model))
        return agent

    monkeypatch.setattr(cli, "build_ace_agent", build_ace_agent)

    result = cli.build_default_agent(api_key="key", model="gpt-5.1")

    assert result is agent
    assert calls == [("key", "gpt-5.1")]


def test_cli_resume_loads_existing_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def create_session_context(*, resume_session_id):
        calls.append(("session", resume_session_id))
        return object(), StubMetadata(), ["restored"], ["history"]

    def run_interactive_tui(received_agent, **kwargs):
        calls.append(("interactive", (received_agent, kwargs)))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(cli, "create_session_context", create_session_context)
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: "recorder")
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main(["resume", "session-1"])

    assert calls == [
        ("session", "session-1"),
        ("build", ("key", "gpt-5.1")),
        (
            "interactive",
            (
                agent,
                {
                    "initial_events": ["restored"],
                    "initial_history": ["history"],
                    "session_recorder": "recorder",
                    "session_id": "session-1",
                },
            ),
        ),
    ]


def test_cli_export_prints_session_text(monkeypatch, capsys) -> None:
    class StubStore:
        def export_text(self, session_id):
            assert session_id == "session-1"
            return "# AceAI session session-1\n\n## user\nhello\n"

    monkeypatch.setattr(cli, "SessionStore", StubStore)

    cli.main(["export", "session-1"])

    captured = capsys.readouterr()
    assert captured.out == "# AceAI session session-1\n\n## user\nhello\n"
