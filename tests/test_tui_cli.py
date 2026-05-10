import pytest

from aceai import __version__
from aceai.agent.config import (
    AgentAppConfig,
    clear_config,
    default_config_path,
    save_config,
)
from aceai.agent.tui import cli
from aceai.agent.provider_auth import CODEX_CLI_AUTH_SENTINEL


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


class StubEventLog:
    def __init__(self, label: str) -> None:
        self.label = label

    def replay_llm_history(self):
        return [f"{self.label}-history"]


@pytest.fixture(autouse=True)
def isolated_cli_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    clear_config()
    yield
    clear_config()


def install_tui_extra_stub(monkeypatch, *, on_launch=None, on_run=None) -> None:
    class StubConfiguredTUI:
        def __init__(
            self,
            agent_factory,
            *,
            initial_config,
            initial_question,
            default_model,
            initial_events,
            initial_history,
            session_recorder,
            session_id,
            **kwargs,
        ):
            self._agent_factory = agent_factory
            self._initial_config = initial_config
            self._initial_events = initial_events
            self._initial_history = initial_history
            self._session_recorder = session_recorder
            self._session_id = session_id
            if on_launch is not None:
                on_launch(
                    initial_config=initial_config,
                    agent_factory=agent_factory,
                    initial_events=initial_events,
                    initial_history=initial_history,
                    session_recorder=session_recorder,
                    session_id=session_id,
                )

        def run(self, **kwargs) -> None:
            if on_run is not None:
                on_run(**kwargs)

    monkeypatch.setattr(cli, "require_tui_extra", lambda: None)
    monkeypatch.setattr(cli, "AceAIConfiguredTUI", StubConfiguredTUI)


def test_cli_missing_tui_extra_explains_install(monkeypatch) -> None:
    monkeypatch.setattr(cli, "SessionRecorder", None)
    monkeypatch.setattr(cli, "SessionStore", None)
    monkeypatch.setattr(cli, "event_log_to_tui_events", None)
    monkeypatch.setattr(cli, "AceAIConfiguredTUI", None)

    def import_module(name: str):
        if name == "aceai.agent.session":
            raise ModuleNotFoundError("No module named 'sqlalchemy'", name="sqlalchemy")
        raise AssertionError(name)

    monkeypatch.setattr(cli.importlib, "import_module", import_module)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert str(exc_info.value) == cli.TUI_EXTRA_INSTALL_HINT


def test_cli_missing_runtime_dependency_explains_refresh(monkeypatch) -> None:
    monkeypatch.setattr(cli, "SessionRecorder", None)
    monkeypatch.setattr(cli, "SessionStore", None)
    monkeypatch.setattr(cli, "event_log_to_tui_events", None)
    monkeypatch.setattr(cli, "AceAIConfiguredTUI", None)

    class StubModule:
        pass

    def import_module(name: str):
        if name in ("aceai.agent.session", "aceai.agent.tui.session_replay"):
            return StubModule()
        if name == "aceai.agent.tui.runner":
            raise ModuleNotFoundError(
                "No module named 'rapidfuzz'",
                name="rapidfuzz",
            )
        raise AssertionError(name)

    monkeypatch.setattr(cli.importlib, "import_module", import_module)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert str(exc_info.value) == cli.TUI_EXTRA_INSTALL_HINT
    assert "uv tool install --force --refresh-package aceai 'aceai[tui]'" in str(
        exc_info.value
    )


def test_cli_version_prints_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"aceai {__version__}\n"


def test_cli_rejects_direct_question_without_creating_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def build_agent(config):
        calls.append(("build", (config.api_key, config.model)))
        return StubAgent()

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch)
    monkeypatch.setattr(cli, "build_agent", build_agent)
    monkeypatch.setattr(
        cli,
        "load_session_context",
        lambda *, session_id: calls.append(("session", session_id)),
    )

    with pytest.raises(ValueError, match="only accepts no arguments"):
        cli.main(["hello", "world"])

    assert calls == []


def test_cli_without_question_runs_interactive_tui(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.api_key, config.model)))
        return agent

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(
            (
                "interactive",
                (
                    built,
                    {
                        "initial_events": kwargs["initial_events"],
                        "initial_history": kwargs["initial_history"],
                        "session_recorder": kwargs["session_recorder"],
                        "session_id": kwargs["session_id"],
                    },
                ),
            )
        )

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("ACEAI_MODEL", "gpt-4o-mini")
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)

    cli.main([])

    assert calls == [
        ("build", ("key", "gpt-4o-mini")),
        (
            "interactive",
            (
                agent,
                {
                    "initial_events": [],
                    "initial_history": [],
                    "session_recorder": None,
                    "session_id": None,
                },
            ),
        ),
    ]


def test_cli_uses_deepseek_env_provider(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.provider, config.api_key, config.model)))
        return agent

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(("interactive", (built, kwargs)))

    monkeypatch.setenv("ACEAI_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)

    cli.main([])

    assert calls[0] == ("build", ("deepseek", "key", "deepseek-v4-pro"))


def test_cli_uses_codex_cli_auth_default(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.provider, config.api_key, config.model)))
        return agent

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(("interactive", (built, kwargs)))

    monkeypatch.setenv("ACEAI_PROVIDER", "codex")
    monkeypatch.delenv("ACEAI_CODEX_TOKEN", raising=False)
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)

    cli.main([])

    assert calls[0] == (
        "build",
        ("codex", CODEX_CLI_AUTH_SENTINEL, "gpt-5.5"),
    )


def test_cli_prefers_project_config_over_env_provider(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.provider, config.api_key, config.model)))
        return agent

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(("interactive", (built, kwargs)))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ACEAI_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    save_config(
        AgentAppConfig(
            provider="openai",
            api_key="project-key",
            model="gpt-4o-mini",
            api_keys={"openai": "project-key"},
        )
    )
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)

    cli.main([])

    assert calls[0] == ("build", ("openai", "project-key", "gpt-4o-mini"))


def test_cli_uses_project_key_without_merging_global_key(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.provider, config.api_key, config.model)))
        return agent

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(("interactive", (built, kwargs)))

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    save_config(
        AgentAppConfig(
            provider="openai",
            api_key="real-global-key",
            model="gpt-5.5",
            api_keys={"openai": "real-global-key"},
        ),
        path=default_config_path(),
    )
    save_config(
        AgentAppConfig(
            provider="openai",
            api_key="test-key",
            model="gpt-5.5",
            api_keys={"openai": "test-key"},
        )
    )
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)

    cli.main([])

    assert calls[0] == ("build", ("openai", "test-key", "gpt-5.5"))


def test_cli_resume_prefers_persisted_session_model(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.provider, config.api_key, config.model)))
        return agent

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(("interactive", (built, kwargs)))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("ACEAI_PROVIDER", "deepseek")
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)
    monkeypatch.setattr(
        cli,
        "load_session_context",
        lambda *, session_id: (
            object(),
            StubMetadata(),
            ["event"],
            ["history"],
            StubDeepSeekSessionState(),
        ),
    )
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)

    cli.main(["resume", "session-1"])

    assert calls[0] == ("build", ("deepseek", "key", "deepseek-v4-flash"))


def test_session_state_cannot_switch_to_disabled_provider() -> None:
    config = AgentAppConfig(
        provider="openai",
        api_key="openai-key",
        model="gpt-5.5",
        api_keys={"openai": "openai-key", "deepseek": "deepseek-key"},
        disabled_providers=["deepseek"],
    )

    resolved = cli.apply_session_state_to_initial_config(
        config,
        StubDeepSeekSessionState(),
    )

    assert resolved == config


def test_load_session_context_uses_main_thread_history(monkeypatch) -> None:
    class Store:
        def get_session(self, session_id):
            return StubMetadata()

        def load_event_log(self, session_id):
            raise AssertionError("resume must not replay all session threads")

        def load_thread_event_log(self, session_id, thread_id):
            assert session_id == "session-1"
            assert thread_id == "main"
            return StubEventLog("main")

        def get_session_state(self, session_id):
            return StubSessionState()

    monkeypatch.setattr(cli, "require_tui_extra", lambda: None)
    monkeypatch.setattr(cli, "SessionStore", Store)
    monkeypatch.setattr(cli, "MAIN_THREAD_ID", "main")
    monkeypatch.setattr(
        cli,
        "event_log_to_tui_events",
        lambda event_log: [f"{event_log.label}-event"],
    )

    _, metadata, events, history, state = cli.load_session_context(
        session_id="session-1"
    )

    assert metadata.session_id == "session-1"
    assert events == ["main-event"]
    assert history == ["main-history"]
    assert isinstance(state, StubSessionState)


def test_cli_without_config_opens_provider_setup_tui(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def on_launch(**kwargs):
        calls.append(kwargs)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ACEAI_MODEL", raising=False)
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)
    monkeypatch.setattr(cli, "load_config", lambda: None)

    cli.main([])

    assert calls[0]["initial_config"] is None
    assert calls[0]["initial_events"] == []
    assert calls[0]["initial_history"] == []
    assert calls[0]["session_recorder"] is None
    assert calls[0]["session_id"] is None


def test_build_agent_uses_main_ace_agent(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    agent = StubAgent()

    def build_ace_agent(*, api_key, model, **kwargs):
        calls.append({"api_key": api_key, "model": model, **kwargs})
        return agent

    monkeypatch.setattr(cli, "build_ace_agent", build_ace_agent)

    result = cli.build_agent(
        AgentAppConfig(
            provider="openai",
            api_key="key",
            model="gpt-4o-mini",
            default_model="gpt-5.5",
            skills="disable",
            skill_selection_mode="selected",
            enabled_skills=["review"],
            api_keys={"openai": "key"},
            tool_permissions={"read_text_file": "ask"},
            tool_enabled={"run_shell_command": False},
            tool_max_calls={"search_text": 2},
            compress_threshold="80%",
        )
    )

    assert result is agent
    assert calls == [
        {
            "api_key": "key",
            "model": "gpt-5.5",
            "provider_name": "openai",
            "skill_path": "disable",
            "enabled_skill_names": ("review",),
            "tool_permissions": {"read_text_file": "ask"},
            "tool_enabled": {"run_shell_command": False},
            "tool_max_calls": {"search_text": 2},
            "compress_threshold": "80%",
        }
    ]


def test_cli_resume_loads_existing_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    agent = StubAgent()

    def build_agent(config):
        calls.append(("build", (config.api_key, config.model)))
        return agent

    def load_session_context(*, session_id):
        calls.append(("session", session_id))
        return object(), StubMetadata(), ["restored"], ["history"], StubSessionState()

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(
            (
                "interactive",
                (
                    built,
                    {
                        "initial_events": kwargs["initial_events"],
                        "initial_history": kwargs["initial_history"],
                        "session_recorder": kwargs["session_recorder"],
                        "session_id": kwargs["session_id"],
                    },
                ),
            )
        )

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)
    monkeypatch.setattr(cli, "load_session_context", load_session_context)
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)

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

    def build_agent(config):
        calls.append(("build", (config.api_key, config.model)))
        return agent

    def load_session_context(*, session_id):
        calls.append(("session", session_id))
        return object(), StubMetadata(), ["restored"], ["history"], StubSessionState()

    def on_launch(**kwargs):
        built = build_agent(kwargs["initial_config"])
        calls.append(
            (
                "interactive",
                (
                    built,
                    {
                        "initial_events": kwargs["initial_events"],
                        "initial_history": kwargs["initial_history"],
                        "session_recorder": kwargs["session_recorder"],
                        "session_id": kwargs["session_id"],
                    },
                ),
            )
        )

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)
    monkeypatch.setattr(cli, "SessionStore", StubStore)
    monkeypatch.setattr(cli, "load_session_context", load_session_context)
    recorder = StubRecorder()
    monkeypatch.setattr(cli, "SessionRecorder", lambda store, session_id: recorder)

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
    install_tui_extra_stub(monkeypatch)
    monkeypatch.setattr(
        cli,
        "load_session_context",
        lambda *, session_id: calls.append(("session", session_id)),
    )

    with pytest.raises(ValueError, match="aceai resume requires a session_id"):
        cli.main(["resume", "session-1", "hello"])

    assert calls == []


def test_cli_does_not_print_saved_for_empty_deleted_session(monkeypatch, capsys) -> None:
    agent = StubAgent()

    def on_launch(**kwargs):
        assert kwargs["session_recorder"] is None
        assert kwargs["session_id"] is None

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch, on_launch=on_launch)

    cli.main([])

    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_export_prints_session_text(monkeypatch, capsys) -> None:
    class StubStore:
        def export_text(self, session_id, *, include_threads=False):
            assert session_id == "session-1"
            assert include_threads is False
            return "# AceAI session session-1\n\n## user\nhello\n"

    monkeypatch.setattr(cli, "SessionStore", StubStore)

    cli.main(["export", "session-1"])

    captured = capsys.readouterr()
    assert captured.out == "# AceAI session session-1\n\n## user\nhello\n"


def test_cli_export_threads_prints_thread_aware_session_text(monkeypatch, capsys) -> None:
    class StubStore:
        def export_text(self, session_id, *, include_threads=False):
            assert session_id == "session-1"
            assert include_threads is True
            return "# AceAI thread main\n"

    monkeypatch.setattr(cli, "SessionStore", StubStore)

    cli.main(["export", "session-1", "--threads"])

    captured = capsys.readouterr()
    assert captured.out == "# AceAI thread main\n"


def test_cli_export_writes_session_text_to_new_file(monkeypatch, tmp_path, capsys) -> None:
    class StubStore:
        def export_text(self, session_id, *, include_threads=False):
            assert session_id == "session-1"
            assert include_threads is False
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
        def export_text(self, session_id, *, include_threads=False):
            assert session_id == "session-1"
            assert include_threads is False
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


def test_cli_inline_runs_textual_app_inline(monkeypatch) -> None:
    run_calls: list[dict[str, object]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch, on_run=lambda **kwargs: run_calls.append(kwargs))

    cli.main(["--inline"])

    assert run_calls == [{"inline": True, "inline_no_clear": True, "size": (80, 25)}]


def test_cli_default_runs_textual_app_fullscreen(monkeypatch) -> None:
    run_calls: list[dict[str, object]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch, on_run=lambda **kwargs: run_calls.append(kwargs))

    cli.main([])

    assert run_calls == [{"inline": False, "inline_no_clear": False, "size": None}]

def test_cli_inline_height_overrides_terminal_height(monkeypatch) -> None:
    run_calls: list[dict[str, object]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    install_tui_extra_stub(monkeypatch, on_run=lambda **kwargs: run_calls.append(kwargs))

    cli.main(["--inline", "--inline-height", "40"])

    assert run_calls == [
        {"inline": True, "inline_no_clear": True, "size": (80, 40)}
    ]
