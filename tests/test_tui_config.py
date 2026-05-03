from aceai.agent.tui.config import AceAITUIConfig, load_config, save_config


def test_save_and_load_config_round_trips(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AceAITUIConfig(
        provider="openai",
        api_key="secret",
        model="gpt-4o-mini",
    )

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_load_config_migrates_stale_default_model(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "provider: openai\napi_key: secret\nmodel: gpt-5.1\n",
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded == AceAITUIConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.5",
    )


def test_load_config_preserves_versioned_gpt51_selection(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = AceAITUIConfig(
        provider="openai",
        api_key="secret",
        model="gpt-5.1",
    )
    save_config(config, path)

    loaded = load_config(path)

    assert loaded == config
