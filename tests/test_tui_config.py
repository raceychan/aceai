from aceai.agent.config import (
    AceAITUIConfig,
    clear_config,
    config_schema,
    current_config,
    load_config,
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


def test_save_and_load_config_round_trips(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AceAITUIConfig(
        provider="openai",
        api_key="secret",
        model="gpt-4o-mini",
        api_keys={"openai": "secret"},
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
    assert current_config() == config
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_replace_config_uses_single_active_config_object() -> None:
    first = AceAITUIConfig(
        provider="openai",
        api_key="first",
        model="gpt-5.5",
        api_keys={"openai": "first"},
    )
    second = AceAITUIConfig(
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

    assert loaded == AceAITUIConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
        api_keys={"openai": "secret"},
    )


def test_load_config_preserves_versioned_gpt51_selection(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AceAITUIConfig(
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
    config = AceAITUIConfig(
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

    assert loaded == AceAITUIConfig(
        provider="deepseek",
        api_key="secret",
        model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        api_keys={"deepseek": "secret"},
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
