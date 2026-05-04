import pytest

from aceai.agent.tui import cli


class StubAgent:
    pass


class StubMetadata:
    session_id = "session-1"


class StubRecorder:
    def __init__(self, *, saved: bool = True) -> None:
        self.saved = saved


class StubSessionState:
    selected_provider = ""
    selected_model = ""


class StubDeepSeekSessionState:
    selected_provider = "deepseek"
    selected_model = "deepseek-v4-flash"


def test_cli_missing_tui_extra_explains_install(monkeypatch) -> None:
    monkeypatch.setattr(cli, "SessionRecorder", None)
    monkeypatch.setattr(cli, "SessionStore", None)
    monkeypatch.setattr(cli, "event_log_to_tui_events", None)
    monkeypatch.setattr(cli, "run_configured_tui", None)
    monkeypatch.setattr(cli, "run_interactive_tui", None)

    def import_module(name: str):
        if name == "aceai.agent.session":
            raise ModuleNotFoundError("No module named 'sqlalchemy'", name="sqlalchemy")
        raise AssertionError(name)

    monkeypatch.setattr(cli.importlib, "import_module", import_module)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert str(exc_info.value) == cli.TUI_EXTRA_INSTALL_HINT


def test_cli_rejects_direct_question_without_creating_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return StubAgent()

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: calls.append(("session", resume_session_id)),
    )

    with pytest.raises(ValueError, match="only accepts no arguments"):
        cli.main(["hello", "world"])

    assert calls == []


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
        lambda *, resume_session_id: (
            object(),
            StubMetadata(),
            ["event"],
            ["history"],
            StubSessionState(),
        ),
    )
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
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
                    "session_recorder": recorder,
                    "session_id": "session-1",
                },
            ),
        ),
    ]


def test_cli_uses_deepseek_env_provider(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model, provider="openai"):
        calls.append(("build", (provider, api_key, model)))
        return agent

    def run_interactive_tui(received_agent, **kwargs):
        calls.append(("interactive", (received_agent, kwargs)))

    monkeypatch.setenv("ACEAI_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: (
            object(),
            StubMetadata(),
            ["event"],
            ["history"],
            StubSessionState(),
        ),
    )
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main([])

    assert calls[0] == ("build", ("deepseek", "key", "deepseek-v4-pro"))


def test_cli_resume_prefers_persisted_session_model(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model, provider="openai"):
        calls.append(("build", (provider, api_key, model)))
        return agent

    def run_interactive_tui(received_agent, **kwargs):
        calls.append(("interactive", (received_agent, kwargs)))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("ACEAI_PROVIDER", "deepseek")
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: (
            object(),
            StubMetadata(),
            ["event"],
            ["history"],
            StubDeepSeekSessionState(),
        ),
    )
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main(["resume", "session-1"])

    assert calls[0] == ("build", ("deepseek", "key", "deepseek-v4-flash"))


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
        lambda *, resume_session_id: (
            object(),
            StubMetadata(),
            ["event"],
            ["history"],
            StubSessionState(),
        ),
    )
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
    monkeypatch.setattr(cli, "run_configured_tui", run_configured_tui)

    cli.main([])

    assert calls[0][0] == "configured"
    payload = calls[0][1]
    assert payload[1] is None
    assert payload[2] == ""
    assert payload[3] == "gpt-5.5"
    assert payload[4] == {
        "initial_events": ["event"],
        "initial_history": ["history"],
        "session_recorder": recorder,
        "session_id": "session-1",
    }


def test_build_default_agent_uses_main_ace_agent(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    agent = StubAgent()

    def build_ace_agent(*, api_key, model):
        calls.append((api_key, model))
        return agent

    monkeypatch.setattr(cli, "build_ace_agent", build_ace_agent)

    result = cli.build_default_agent(api_key="key", model="gpt-5.5")

    assert result is agent
    assert calls == [("key", "gpt-5.5")]


def test_cli_resume_loads_existing_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def create_session_context(*, resume_session_id):
        calls.append(("session", resume_session_id))
        return object(), StubMetadata(), ["restored"], ["history"], StubSessionState()

    def run_interactive_tui(received_agent, **kwargs):
        calls.append(("interactive", (received_agent, kwargs)))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(cli, "create_session_context", create_session_context)
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main(["resume", "session-1"])

    assert calls == [
        ("session", "session-1"),
        ("build", ("key", "gpt-5.5")),
        (
            "interactive",
            (
                agent,
                {
                    "initial_events": ["restored"],
                    "initial_history": ["history"],
                    "session_recorder": recorder,
                    "session_id": "session-1",
                },
            ),
        ),
    ]


def test_cli_resume_without_session_id_loads_latest_updated_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    class StubStore:
        def list_sessions(self):
            class LatestMetadata:
                session_id = "latest-session"

            return [LatestMetadata()]

    def build_default_agent(*, api_key, model):
        calls.append(("build", (api_key, model)))
        return agent

    def create_session_context(*, resume_session_id):
        calls.append(("session", resume_session_id))
        return object(), StubMetadata(), ["restored"], ["history"], StubSessionState()

    def run_interactive_tui(received_agent, **kwargs):
        calls.append(("interactive", (received_agent, kwargs)))

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "SessionStore", StubStore)
    monkeypatch.setattr(cli, "build_default_agent", build_default_agent)
    monkeypatch.setattr(cli, "create_session_context", create_session_context)
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main(["resume"])

    assert calls == [
        ("session", "latest-session"),
        ("build", ("key", "gpt-5.5")),
        (
            "interactive",
            (
                agent,
                {
                    "initial_events": ["restored"],
                    "initial_history": ["history"],
                    "session_recorder": recorder,
                    "session_id": "session-1",
                },
            ),
        ),
    ]


def test_cli_resume_rejects_extra_question(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: calls.append(("session", resume_session_id)),
    )

    with pytest.raises(ValueError, match="aceai resume requires a session_id"):
        cli.main(["resume", "session-1", "hello"])

    assert calls == []


def test_cli_does_not_print_saved_for_empty_deleted_session(monkeypatch, capsys) -> None:
    agent = StubAgent()
    recorder = StubRecorder(saved=True)

    def run_interactive_tui(received_agent, **kwargs):
        assert received_agent is agent
        assert kwargs["session_recorder"] is recorder
        recorder.saved = False

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(cli, "build_default_agent", lambda *, api_key, model: agent)
    monkeypatch.setattr(
        cli,
        "create_session_context",
        lambda *, resume_session_id: (
            object(),
            StubMetadata(),
            [],
            [],
            StubSessionState(),
        ),
    )
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)
    monkeypatch.setattr(cli, "run_interactive_tui", run_interactive_tui)

    cli.main([])

    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_export_prints_session_text(monkeypatch, capsys) -> None:
    class StubStore:
        def export_text(self, session_id):
            assert session_id == "session-1"
            return "# AceAI session session-1\n\n## user\nhello\n"

    monkeypatch.setattr(cli, "SessionStore", StubStore)

    cli.main(["export", "session-1"])

    captured = capsys.readouterr()
    assert captured.out == "# AceAI session session-1\n\n## user\nhello\n"


def test_cli_export_writes_session_text_to_new_file(monkeypatch, tmp_path, capsys) -> None:
    class StubStore:
        def export_text(self, session_id):
            assert session_id == "session-1"
            return "# AceAI session session-1\n\n## user\nhello\n"

    target = tmp_path / "debug.md"
    monkeypatch.setattr(cli, "SessionStore", StubStore)

    cli.main(["export", "session-1", f"--file={target}"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert target.read_text(encoding="utf-8") == (
        "# AceAI session session-1\n\n## user\nhello\n"
    )


def test_cli_export_file_fails_when_target_exists(monkeypatch, tmp_path) -> None:
    class StubStore:
        def export_text(self, session_id):
            assert session_id == "session-1"
            return "# AceAI session session-1\n"

    target = tmp_path / "debug.md"
    target.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(cli, "SessionStore", StubStore)

    with pytest.raises(FileExistsError):
        cli.main(["export", "session-1", f"--file={target}"])

    assert target.read_text(encoding="utf-8") == "existing\n"


def test_cli_cost_prints_total_session_cost(monkeypatch, capsys) -> None:
    class StubStore:
        def total_cost_usd(self):
            return 0.012345

    monkeypatch.setattr(cli, "SessionStore", StubStore)

    cli.main(["cost"])

    captured = capsys.readouterr()
    assert captured.out == "$0.0123\n"
