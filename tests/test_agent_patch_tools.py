import subprocess

import pytest

from agent_core.features.patch import apply_patch, preview_patch
from agent_core.features.repo import git_diff, git_status
from aceai.core import ToolExecutionError


def test_preview_patch_returns_diff_without_writing(tmp_path) -> None:
    target = tmp_path / "hello.txt"
    patch = """*** Begin Patch
*** Add File: hello.txt
+hello
*** End Patch"""

    result = preview_patch(patch=patch, cwd=str(tmp_path))

    assert result.affected_paths == ["hello.txt"]
    assert result.changes[0].action == "add"
    assert "--- /dev/null" in result.diff
    assert "+hello" in result.diff
    assert not target.exists()


def test_apply_patch_add_update_delete_and_move(tmp_path) -> None:
    (tmp_path / "app.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    (tmp_path / "old.txt").write_text("remove me\n", encoding="utf-8")
    (tmp_path / "name.txt").write_text("old name\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Add File: created.txt
+created
*** Update File: app.py
@@
 def greet():
-    return 'hi'
+    return 'hello'
*** Delete File: old.txt
*** Update File: name.txt
*** Move to: renamed.txt
@@
-old name
+new name
*** End Patch"""

    result = apply_patch(patch=patch, cwd=str(tmp_path))

    assert result.applied is True
    assert result.affected_paths == [
        "app.py",
        "created.txt",
        "name.txt",
        "old.txt",
        "renamed.txt",
    ]
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created\n"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == (
        "def greet():\n    return 'hello'\n"
    )
    assert not (tmp_path / "old.txt").exists()
    assert not (tmp_path / "name.txt").exists()
    assert (tmp_path / "renamed.txt").read_text(encoding="utf-8") == "new name\n"


def test_patch_rejects_absolute_path(tmp_path) -> None:
    patch = f"""*** Begin Patch
*** Add File: {tmp_path / "bad.txt"}
+bad
*** End Patch"""

    with pytest.raises(ToolExecutionError, match="Patch paths must be relative"):
        preview_patch(patch=patch, cwd=str(tmp_path))


def test_patch_rejects_cwd_escape(tmp_path) -> None:
    patch = """*** Begin Patch
*** Add File: ../bad.txt
+bad
*** End Patch"""

    with pytest.raises(ToolExecutionError, match="Patch path escapes cwd"):
        preview_patch(patch=patch, cwd=str(tmp_path))


def test_patch_update_requires_matching_context(tmp_path) -> None:
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Update File: app.py
@@
-print('missing')
+print('hello')
*** End Patch"""

    with pytest.raises(ToolExecutionError, match="Patch context not found"):
        preview_patch(patch=patch, cwd=str(tmp_path))


def test_git_status_and_diff_tools(tmp_path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "aceai@example.com")
    _git(tmp_path, "config", "user.name", "AceAI")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "init")
    tracked.write_text("two\n", encoding="utf-8")

    status = git_status(cwd=str(tmp_path))
    diff = git_diff(cwd=str(tmp_path), paths=["tracked.txt"])

    assert "tracked.txt" in status.status
    assert diff.paths == ["tracked.txt"]
    assert "-one" in diff.unstaged
    assert "+two" in diff.unstaged


def _git(cwd, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
