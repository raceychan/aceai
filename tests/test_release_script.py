from pathlib import Path

import pytest

import scripts.release as release


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

