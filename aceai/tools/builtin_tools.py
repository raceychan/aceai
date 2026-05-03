import subprocess
from pathlib import Path
from typing import Any

from msgspec import Struct

from ._tool_sig import Annotated, spec
from .tool import Tool
from .tool import tool


class DirEntry(Struct, frozen=True, kw_only=True):
    name: str
    path: str
    kind: str


class SearchMatch(Struct, frozen=True, kw_only=True):
    path: str
    line_number: int
    line: str


class EditResult(Struct, frozen=True, kw_only=True):
    path: str
    replacements: int


class CommandResult(Struct, frozen=True, kw_only=True):
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str


@tool
def read_text_file(
    path: Annotated[str, spec(description="Path to a UTF-8 text file to read")],
) -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


@tool
def list_dir(
    path: Annotated[str, spec(description="Directory path to list")],
) -> list[DirEntry]:
    """List direct children of a directory."""
    entries: list[DirEntry] = []
    for child in sorted(Path(path).iterdir()):
        kind = "directory" if child.is_dir() else "file"
        entries.append(DirEntry(name=child.name, path=str(child), kind=kind))
    return entries


@tool
def search_text(
    query: Annotated[str, spec(description="Literal text to search for")],
    path: Annotated[str, spec(description="Directory tree or file to search")] = ".",
) -> list[SearchMatch]:
    """Search UTF-8 text files for a literal string."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    matches: list[SearchMatch] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        for index, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(
                    SearchMatch(path=str(file), line_number=index, line=line)
                )
    return matches


@tool
def edit_text_file(
    path: Annotated[str, spec(description="Path to the UTF-8 text file to edit")],
    old_string: Annotated[str, spec(description="Existing text to replace")],
    new_string: Annotated[str, spec(description="Replacement text")],
) -> EditResult:
    """Replace exactly one occurrence of old_string in a UTF-8 text file."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count != 1:
        raise ValueError(f"old_string must appear exactly once, found {count}")
    file.write_text(text.replace(old_string, new_string), encoding="utf-8")
    return EditResult(path=str(file), replacements=1)


@tool
def run_command(
    command: Annotated[
        list[str],
        spec(description="Command argv to execute, e.g. ['uv', 'run', 'pytest']"),
    ],
    cwd: Annotated[str, spec(description="Working directory for the command")] = ".",
    timeout_seconds: Annotated[
        int,
        spec(description="Command timeout in seconds", gt=0),
    ] = 60,
) -> CommandResult:
    """Run a command and return stdout, stderr, and exit code."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandResult(
        command=command,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

BUILTIN_TOOLS: list[Tool[Any, Any]] = [
    read_text_file,
    list_dir,
    search_text,
    edit_text_file,
    run_command,
]
