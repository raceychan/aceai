import subprocess
from pathlib import Path
from typing import Any

from msgspec import Struct

from aceai.core.tools import Annotated, Tool, spec, tool


class DirectoryEntry(Struct, frozen=True, kw_only=True):
    name: str
    path: str
    kind: str


class DirectoryListing(Struct, frozen=True, kw_only=True):
    path: str
    entries: list[DirectoryEntry]


class TextFile(Struct, frozen=True, kw_only=True):
    path: str
    content: str


class FileWriteResult(Struct, frozen=True, kw_only=True):
    path: str
    bytes_written: int


class TextReplacementResult(Struct, frozen=True, kw_only=True):
    path: str
    replacements: int


class CommandResult(Struct, frozen=True, kw_only=True):
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str


class SearchResult(Struct, frozen=True, kw_only=True):
    query: str
    path: str
    exit_code: int
    matches: str
    errors: str


@tool(tags=["agent_app", "filesystem"])
def list_directory(
    path: Annotated[str, spec(description="Directory path to list")],
) -> DirectoryListing:
    """List direct children of a directory."""
    root = Path(path).expanduser()
    entries: list[DirectoryEntry] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            kind = "directory"
        elif child.is_file():
            kind = "file"
        else:
            kind = "other"
        entries.append(
            DirectoryEntry(
                name=child.name,
                path=str(child),
                kind=kind,
            )
        )
    return DirectoryListing(path=str(root), entries=entries)


@tool(tags=["agent_app", "filesystem"])
def read_text_file(
    path: Annotated[str, spec(description="UTF-8 text file path to read")],
) -> TextFile:
    """Read a UTF-8 text file exactly as stored on disk."""
    target = Path(path).expanduser()
    return TextFile(path=str(target), content=target.read_text(encoding="utf-8"))


@tool(tags=["agent_app", "filesystem"], max_calls_per_run=8)
def write_text_file(
    path: Annotated[str, spec(description="UTF-8 text file path to write")],
    content: Annotated[str, spec(description="Complete file content to write")],
) -> FileWriteResult:
    """Write complete UTF-8 text content to a file."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = target.write_text(content, encoding="utf-8")
    return FileWriteResult(path=str(target), bytes_written=bytes_written)


@tool(tags=["agent_app", "filesystem"], max_calls_per_run=12)
def replace_text_in_file(
    path: Annotated[str, spec(description="UTF-8 text file path to edit")],
    old_text: Annotated[str, spec(description="Exact text to replace")],
    new_text: Annotated[str, spec(description="Replacement text")],
) -> TextReplacementResult:
    """Replace exact text in a UTF-8 file."""
    target = Path(path).expanduser()
    content = target.read_text(encoding="utf-8")
    if old_text not in content:
        raise ValueError("old_text was not found in file")
    updated = content.replace(old_text, new_text)
    target.write_text(updated, encoding="utf-8")
    return TextReplacementResult(
        path=str(target),
        replacements=content.count(old_text),
    )


@tool(tags=["agent_app", "shell"], max_calls_per_run=8)
def run_shell_command(
    command: Annotated[str, spec(description="Shell command to execute")],
    cwd: Annotated[str, spec(description="Working directory for the command")] = ".",
    timeout_seconds: Annotated[int, spec(description="Command timeout in seconds")] = 120,
) -> CommandResult:
    """Run a shell command and return stdout, stderr, and exit code."""
    completed = subprocess.run(
        command,
        cwd=Path(cwd).expanduser(),
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return CommandResult(
        command=command,
        cwd=cwd,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@tool(tags=["agent_app", "search"], max_calls_per_run=12)
def search_text(
    query: Annotated[str, spec(description="ripgrep search pattern")],
    path: Annotated[str, spec(description="File or directory path to search")] = ".",
) -> SearchResult:
    """Search text with ripgrep and return line-numbered matches."""
    completed = subprocess.run(
        ["rg", "--line-number", "--column", query, path],
        capture_output=True,
        text=True,
        check=False,
    )
    return SearchResult(
        query=query,
        path=path,
        exit_code=completed.returncode,
        matches=completed.stdout,
        errors=completed.stderr,
    )


def default_agent_tools() -> list[Tool[Any, Any]]:
    return [
        list_directory,
        read_text_file,
        write_text_file,
        replace_text_in_file,
        run_shell_command,
        search_text,
    ]
