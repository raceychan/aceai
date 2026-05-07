from aceai.agent.config import (
    AgentAppConfig,
    LEGACY_AGENT_SKILLS_DIR,
    clear_config,
    config_schema,
    current_config,
    effective_config_path,
    load_config,
    project_config_path,
    replace_config,
    save_config,
)


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
    assert fields["tool_permissions"].value_type == "mapping"
    assert fields["tool_enabled"].value_type == "mapping"
    assert fields["tool_max_calls"].value_type == "mapping"
    assert fields["compress_threshold"].value_type == "string"


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
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
    assert current_config() == config
    assert oct(path.stat().st_mode & 0o777) == "0o600"


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
        provider="openai",
        api_key="project",
        model="gpt-5.5",
        api_keys={"openai": "project"},
    )
    save_config(project_config, project_path)

    assert effective_config_path() == project_path
    assert load_config() == project_config


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
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
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


def test_load_config_rejects_top_level_api_key(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config_version: 2\nprovider: openai\napi_key: secret\n",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert str(exc) == "AceAI config api_keys missing active provider"
    else:
        raise AssertionError("load_config accepted legacy top-level api_key")


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
