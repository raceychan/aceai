from pathlib import Path

import pytest

import scripts.release as release
from git import Repo


def test_read_current_version_does_not_exec_module(tmp_path: Path) -> None:
    version_file = tmp_path / "__init__.py"
    version_file.write_text(
        '__version__ = "1.2.3"\n\nfrom .agent import AgentBase as AgentBase\n'
    )
    original_version_file = release.VERSION_FILE
    release.VERSION_FILE = version_file
    try:
        assert release.read_current_version() == "1.2.3"
    finally:
        release.VERSION_FILE = original_version_file


def test_read_current_version_requires_version_assignment(tmp_path: Path) -> None:
    version_file = tmp_path / "__init__.py"
    version_file.write_text('from .agent import AgentBase as AgentBase\n')
    original_version_file = release.VERSION_FILE
    release.VERSION_FILE = version_file
    try:
        with pytest.raises(SystemExit, match="Could not locate __version__ assignment"):
            release.read_current_version()
    finally:
        release.VERSION_FILE = original_version_file


def test_read_latest_remote_tag_version_reads_from_origin(tmp_path: Path) -> None:
    origin_path = tmp_path / "origin.git"
    origin_repo = Repo.init(origin_path, bare=True)

    work_path = tmp_path / "work"
    repo = Repo.init(work_path)
    repo.create_remote("origin", str(origin_path))
    (work_path / "README.md").write_text("hi\n")
    repo.index.add(["README.md"])
    repo.index.commit("init")
    repo.create_tag("v0.1.5")
    repo.create_tag("v0.1.6")
    repo.git.push("--tags", "origin")

    assert release.read_latest_remote_tag_version(repo) == "0.1.6"


def test_ensure_version_order_compares_target_against_latest_remote_tag() -> None:
    release.ensure_version_order(
        "0.1.7",
        "0.1.7",
        "0.1.6",
        skip_update=False,
    )


def test_ensure_version_order_rejects_target_not_greater_than_remote_tag() -> None:
    with pytest.raises(SystemExit, match="Local branch version .* must be greater than latest remote tag"):
        release.ensure_version_order(
            "0.1.7",
            "0.1.6",
            "0.1.6",
            skip_update=False,
        )


def test_infer_version_from_version_branch(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path)
    (tmp_path / "README.md").write_text("hi\n")
    repo.index.add(["README.md"])
    repo.index.commit("init")
    repo.git.checkout("-b", "version/1.2.3")

    assert release.infer_version_from_branch(repo) == "1.2.3"


def test_infer_version_requires_version_branch(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path)
    (tmp_path / "README.md").write_text("hi\n")
    repo.index.add(["README.md"])
    repo.index.commit("init")
    repo.git.checkout("-b", "feature/foo")

    with pytest.raises(SystemExit, match="Provide --version or check out version/"):
        release.infer_version_from_branch(repo)
