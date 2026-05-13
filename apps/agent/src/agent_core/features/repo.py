import subprocess
from pathlib import Path

from msgspec import Struct, field

from aceai.core.executor import ToolExecutionError
from aceai.core.tools import Annotated, spec, tool


class GitStatus(Struct, frozen=True, kw_only=True):
    cwd: str
    status: str


class GitDiff(Struct, frozen=True, kw_only=True):
    cwd: str
    paths: list[str] = field(default_factory=list[str])
    staged: str
    unstaged: str
    diff: str


@tool(tags=["dev"])
def git_status(
    cwd: Annotated[str, spec(description="Git workspace directory")] = ".",
) -> GitStatus:
    """Return git branch and working-tree status."""
    root = Path(cwd).expanduser().resolve()
    completed = _git(root, ["status", "--short", "--branch"])
    return GitStatus(cwd=str(root), status=completed.stdout)


@tool(tags=["dev"])
def git_diff(
    cwd: Annotated[str, spec(description="Git workspace directory")] = ".",
    paths: Annotated[list[str], spec(description="Optional relative paths to diff")] = [],
) -> GitDiff:
    """Return staged and unstaged git diffs."""
    root = Path(cwd).expanduser().resolve()
    resolved_paths = [_relative_git_path(root, path) for path in paths]
    staged = _git(root, ["diff", "--cached", "--", *resolved_paths]).stdout
    unstaged = _git(root, ["diff", "--", *resolved_paths]).stdout
    return GitDiff(
        cwd=str(root),
        paths=resolved_paths,
        staged=staged,
        unstaged=unstaged,
        diff=staged + unstaged,
    )


def _git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    if completed.returncode != 0:
        message = completed.stderr or completed.stdout
        raise ToolExecutionError(message)
    return completed


def _relative_git_path(root: Path, path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        raise ToolExecutionError("Git diff paths must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ToolExecutionError("Git diff path escapes cwd")
    return resolved.relative_to(root).as_posix()
