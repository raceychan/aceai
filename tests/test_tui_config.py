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
