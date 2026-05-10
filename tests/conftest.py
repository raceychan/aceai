import pytest


@pytest.fixture(autouse=True)
def isolate_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
