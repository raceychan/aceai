import pytest
import json

from aceai.agent import config as config_module
from aceai.agent.provider_catalog import auth_mode, provider_options
from aceai.agent.config import (
    AgentAppConfig,
    LEGACY_AGENT_SKILLS_DIR,
    clear_config,
    config_audit_path,
    config_schema,
    current_config,
    effective_config_path,
    load_config,
    load_config_audit,
    project_config_path,
    replace_config,
    save_config,
)


def test_provider_catalog_records_provider_auth_modes() -> None:
    assert auth_mode("openai") == "api_key"
    assert auth_mode("deepseek") == "api_key"
    assert auth_mode("codex") == "subscription"
    assert ("Codex (subscription)", "codex") in provider_options()


def test_config_schema_lists_required_fields() -> None:
    schema = config_schema()

    fields = {field.name: field for field in schema.fields}

    assert fields["provider"].value_type == "string"
    assert fields["model"].required
    assert fields["default_model"].required
    assert fields["api_key"].required
    assert fields["skills"].value_type == "string"
    assert fields["skill_selection_mode"].value_type == "string"
    assert fields["enabled_skills"].value_type == "list"
    assert fields["api_keys"].value_type == "mapping"
    assert fields["disabled_providers"].value_type == "list"
    assert fields["tool_permissions"].value_type == "mapping"
    assert fields["tool_enabled"].value_type == "mapping"
    assert fields["tool_max_calls"].value_type == "mapping"
    assert fields["compress_threshold"].value_type == "string"
    assert fields["reasoning_level"].value_type == "string"


def test_save_and_load_config_round_trips(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-4o-mini",
        api_keys={"openai": "secret"},
        tool_permissions={"run_shell_command": "ask"},
        tool_enabled={"read_text_file": False},
        tool_max_calls={"search_text": 4},
        compress_threshold="80%",
        reasoning_level="auto",
        disabled_providers=["deepseek"],
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
    assert current_config() == config
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_save_config_takes_exclusive_file_lock(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    calls: list[int] = []

    def record_lock(file_descriptor: int, operation: int) -> None:
        calls.append(operation)

    monkeypatch.setattr(config_module.fcntl, "flock", record_lock)
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
    )

    save_config(config, path)

    assert calls == [
        config_module.fcntl.LOCK_EX,
        config_module.fcntl.LOCK_EX,
        config_module.fcntl.LOCK_UN,
        config_module.fcntl.LOCK_UN,
    ]
    assert (tmp_path / "config.yaml.lock").exists()


def test_load_config_takes_shared_file_lock(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
    )
    save_config(config, path)
    calls: list[int] = []

    def record_lock(file_descriptor: int, operation: int) -> None:
        calls.append(operation)

    monkeypatch.setattr(config_module.fcntl, "flock", record_lock)

    assert load_config(path) == config
    assert calls == [config_module.fcntl.LOCK_SH, config_module.fcntl.LOCK_UN]


def test_save_config_audits_changes_to_global_log(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = tmp_path / "project" / ".aceai" / "config.yml"
    first = AgentAppConfig(
        provider="codex",
        api_key="codex-cli",
        model="gpt-5.5",
        api_keys={"openai": "openai-secret", "codex": "codex-cli"},
        disabled_providers=["openai"],
    )
    second = AgentAppConfig(
        provider="codex",
        api_key="codex-cli",
        model="gpt-5.5",
        default_model="gpt-5.5",
        api_keys={"openai": "openai-secret", "codex": "codex-cli"},
        disabled_providers=["openai"],
    )

    save_config(first, path)
    save_config(second, path)

    global_audit_path = tmp_path / "home" / ".aceai" / "config.audit.jsonl"
    entries = [
        json.loads(line)
        for line in global_audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 1
    assert entries[-1]["target"] == str(path)
    assert "timestamp" in entries[-1]
    assert "pid" in entries[-1]
    assert "cwd" in entries[-1]
    assert "caller" in entries[-1]
    assert entries[-1]["changed_fields"] == sorted(entries[-1]["after"].keys())
    assert entries[-1]["before"] is None
    assert entries[-1]["after"]["api_key_providers"] == ["codex", "openai"]
    assert entries[-1]["after"]["disabled_providers"] == ["openai"]
    assert entries[-1]["after"]["provider"] == "codex"
    assert entries[-1]["after"]["model"] == "gpt-5.5"
    assert "openai-secret" not in global_audit_path.read_text(encoding="utf-8")
    assert not (path.parent / "config.yml.providers.audit.jsonl").exists()


def test_load_config_audit_returns_latest_entries_first(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    first_path = tmp_path / "project-a" / ".aceai" / "config.yml"
    second_path = tmp_path / "project-b" / ".aceai" / "config.yml"

    save_config(
        AgentAppConfig(
            provider="openai",
            api_key="secret",
            model="gpt-5.5",
            api_keys={"openai": "secret"},
        ),
        first_path,
    )
    save_config(
        AgentAppConfig(
            provider="deepseek",
            api_key="secret",
            model="deepseek-v4-pro",
            default_model="deepseek-v4-pro",
            api_keys={"deepseek": "secret"},
        ),
        second_path,
    )

    entries = load_config_audit(limit=2)

    assert config_audit_path() == tmp_path / "home" / ".aceai" / "config.audit.jsonl"
    assert [entry.target for entry in entries] == [str(second_path), str(first_path)]
    assert entries[0].after["provider"] == "deepseek"
    assert load_config_audit(limit=10, target=first_path)[0].target == str(first_path)


def test_save_config_audit_records_only_changed_fields(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = tmp_path / "project" / ".aceai" / "config.yml"
    first = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
        tool_permissions={"read_text_file": "ask"},
    )
    second = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
        tool_permissions={"read_text_file": "always"},
    )

    save_config(first, path)
    save_config(second, path)

    entries = load_config_audit(limit=10, target=path)
    assert entries[0].changed_fields == ("tool_permissions",)
    assert entries[0].before["tool_permissions"] == {"read_text_file": "ask"}
    assert entries[0].after["tool_permissions"] == {"read_text_file": "always"}


def test_tests_cannot_write_real_project_config() -> None:
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
    )

    with pytest.raises(RuntimeError, match="must not write the real AceAI"):
        save_config(config, config_module._repo_project_config_path())


def test_tests_cannot_write_real_home_config_audit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(config_module._PROCESS_START_HOME))
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
    )

    with pytest.raises(RuntimeError, match="must not write the real AceAI config audit"):
        save_config(config, tmp_path / "project" / ".aceai" / "config.yml")


def test_config_rejects_active_disabled_provider() -> None:
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
        disabled_providers=["openai"],
    )

    with pytest.raises(ValueError, match="provider is disabled"):
        replace_config(config)


def test_save_and_load_config_round_trips_reasoning_level(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
        reasoning_level="high",
    )

    save_config(config, path)

    assert load_config(path) == config


def test_save_and_load_config_round_trips_deepseek_reasoning_level(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AgentAppConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        api_keys={"deepseek": "secret"},
        reasoning_level="max",
    )

    save_config(config, path)

    assert load_config(path) == config


def test_load_config_rejects_reasoning_level_for_unsupported_model(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            (
                "version: 4",
                "provider: openai",
                "model: gpt-4o",
                "default_model: gpt-4o",
                "api_key: secret",
                "skills: auto",
                "skill_selection_mode: all",
                "enabled_skills: []",
                "api_keys:",
                "  openai: secret",
                "tool_permissions: {}",
                "tool_enabled: {}",
                "tool_max_calls: {}",
                "compress_threshold: 100%",
                "reasoning_level: high",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_level is unsupported for model"):
        load_config(path)


def test_load_config_rejects_provider_specific_unsupported_reasoning_level(
    tmp_path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            (
                "version: 4",
                "provider: deepseek",
                "model: deepseek-v4-pro",
                "default_model: deepseek-v4-pro",
                "api_key: secret",
                "skills: auto",
                "skill_selection_mode: all",
                "enabled_skills: []",
                "api_keys:",
                "  deepseek: secret",
                "tool_permissions: {}",
                "tool_enabled: {}",
                "tool_max_calls: {}",
                "compress_threshold: 100%",
                "reasoning_level: medium",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_level is unsupported for model"):
        load_config(path)


def test_load_config_prefers_project_config_over_global(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    global_path = tmp_path / "home" / ".aceai" / "config.yaml"
    project_path = tmp_path / "project" / ".aceai" / "config.yml"
    global_path.parent.mkdir(parents=True)
    project_path.parent.mkdir(parents=True)
    save_config(
        AgentAppConfig(
            provider="openai",
            api_key="global",
            model="gpt-4o-mini",
            api_keys={"openai": "global"},
        ),
        global_path,
    )
    project_config = AgentAppConfig(
        provider="codex",
        api_key="codex-cli",
        model="gpt-5.5",
        api_keys={"codex": "codex-cli"},
        disabled_providers=["openai"],
    )
    save_config(project_config, project_path)

    assert effective_config_path() == project_path
    assert load_config() == project_config


def test_load_config_does_not_merge_global_api_keys_into_project_config(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    global_path = tmp_path / "home" / ".aceai" / "config.yaml"
    project_path = tmp_path / "project" / ".aceai" / "config.yml"
    global_path.parent.mkdir(parents=True)
    project_path.parent.mkdir(parents=True)
    save_config(
        AgentAppConfig(
            provider="openai",
            api_key="global",
            model="gpt-4o-mini",
            api_keys={"openai": "global"},
        ),
        global_path,
    )
    project_path.write_text(
        "config_version: 4\n"
        "provider: deepseek\n"
        "api_keys: {}\n"
        "disabled_providers:\n"
        "  - openai\n"
        "model: deepseek-v4-pro\n"
        "default_model: deepseek-v4-pro\n",
        encoding="utf-8",
    )

    assert effective_config_path() == project_path
    assert load_config() is None


def test_load_config_falls_back_to_global_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    global_path = tmp_path / "home" / ".aceai" / "config.yaml"
    global_config = AgentAppConfig(
        provider="openai",
        api_key="global",
        model="gpt-4o-mini",
        api_keys={"openai": "global"},
    )
    save_config(global_config, global_path)

    assert effective_config_path() == global_path
    assert load_config() == global_config


def test_save_config_defaults_to_project_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = AgentAppConfig(
        provider="codex",
        api_key="codex-cli",
        model="gpt-5.5",
        api_keys={"codex": "codex-cli"},
        disabled_providers=["openai"],
    )

    save_config(config)

    path = project_config_path()
    assert path == tmp_path / ".aceai" / "config.yml"
    assert load_config(path) == config
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_save_config_replaces_config_without_backup_files(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    first = AgentAppConfig(
        provider="openai",
        api_key="first",
        model="gpt-4o-mini",
        api_keys={"openai": "first"},
    )
    second = AgentAppConfig(
        provider="openai",
        api_key="second",
        model="gpt-5.5",
        api_keys={"openai": "second"},
    )

    save_config(first, path)
    first_inode = path.stat().st_ino
    save_config(second, path)

    assert load_config(path) == second
    assert path.stat().st_ino != first_inode
    assert list(tmp_path.glob("*.bak")) == []
    assert list(tmp_path.glob(".config.yaml.*.tmp")) == []
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_replace_config_uses_single_active_config_object() -> None:
    first = AgentAppConfig(
        provider="openai",
        api_key="first",
        model="gpt-5.5",
        api_keys={"openai": "first"},
    )
    second = AgentAppConfig(
        provider="openai",
        api_key="second",
        model="gpt-4o",
        api_keys={"openai": "second"},
    )

    replace_config(first)
    active_first = current_config()
    replace_config(second)

    assert active_first is first
    assert current_config() is second
    clear_config()
    assert current_config() is None


def test_load_config_migrates_stale_default_model(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "provider: openai\napi_keys:\n  openai: secret\nmodel: gpt-5.1\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded == AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
    )


def test_load_config_preserves_versioned_gpt51_selection(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.1",
        api_keys={"openai": "secret"},
    )
    save_config(config, path)

    loaded = load_config(path)

    assert loaded == config


def test_save_and_load_deepseek_config_round_trips(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AgentAppConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        api_keys={"deepseek": "secret"},
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config


def test_load_config_uses_provider_default_when_model_is_missing(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 2\nprovider: deepseek\napi_keys:\n  deepseek: secret\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded == AgentAppConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        api_keys={"deepseek": "secret"},
    )


def test_load_config_replaces_legacy_builtin_skill_path_with_auto(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 2\n"
        "provider: openai\n"
        "api_keys:\n"
        "  openai: secret\n"
        "model: gpt-5.5\n"
        "default_model: gpt-5.5\n"
        f"skills: {LEGACY_AGENT_SKILLS_DIR}\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded == AgentAppConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "secret"},
    )


def test_load_config_ignores_top_level_api_key_without_crashing(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 2\nprovider: openai\napi_key: secret\n",
        encoding="utf-8",
    )

    assert load_config(path) is None


def test_load_config_uses_keyed_provider_when_active_provider_has_no_key(
    tmp_path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 4\n"
        "provider: openai\n"
        "api_keys:\n"
        "  deepseek: secret\n"
        "model: gpt-5.5\n"
        "default_model: gpt-5.5\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded == AgentAppConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        api_keys={"deepseek": "secret"},
    )


def test_load_config_skips_disabled_active_provider(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 4\n"
        "provider: openai\n"
        "disabled_providers:\n"
        "  - openai\n"
        "api_keys:\n"
        "  openai: expensive\n"
        "  deepseek: cheap\n"
        "model: gpt-5.5\n"
        "default_model: gpt-5.5\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded == AgentAppConfig(
        provider="deepseek",
        api_key="cheap",
        model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        api_keys={"openai": "expensive", "deepseek": "cheap"},
        disabled_providers=["openai"],
    )


def test_load_config_rejects_disabled_as_tool_permission(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 3\n"
        "provider: openai\n"
        "api_keys:\n"
        "  openai: secret\n"
        "model: gpt-5.5\n"
        "tool_permissions:\n"
        "  run_shell_command: never\n",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert str(exc) == "AceAI config tool_permissions value is unsupported"
    else:
        raise AssertionError("load_config accepted disabled permission policy")


def test_load_config_rejects_invalid_tool_max_calls(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 3\n"
        "provider: openai\n"
        "api_keys:\n"
        "  openai: secret\n"
        "model: gpt-5.5\n"
        "tool_max_calls:\n"
        "  run_shell_command: 0\n",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert str(exc) == "AceAI config tool_max_calls values must be positive"
    else:
        raise AssertionError("load_config accepted non-positive tool max calls")


def test_load_config_rejects_invalid_compress_threshold(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 3\n"
        "provider: openai\n"
        "api_keys:\n"
        "  openai: secret\n"
        "model: gpt-5.5\n"
        "compress_threshold: 101%\n",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert str(exc) == "compress_threshold must be a percentage from 0% to 100%"
    else:
        raise AssertionError("load_config accepted invalid compress threshold")


def test_load_config_accepts_numeric_compress_threshold(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 3\n"
        "provider: openai\n"
        "api_keys:\n"
        "  openai: secret\n"
        "model: gpt-5.5\n"
        "compress_threshold: 2048\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded is not None
    assert loaded.compress_threshold == 2048
