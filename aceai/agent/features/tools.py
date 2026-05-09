import subprocess
from pathlib import Path
from typing import Any

from msgspec import Struct

from aceai.core.executor import ToolExecutionError
from aceai.agent.permissions import ToolPermission
from aceai.core.tools import Annotated, Tool, ToolMeta, spec, tool

from .patch import apply_patch, preview_patch
from .repo import git_diff, git_status


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


@tool(tags=["dev"])
def list_directory(
    path: Annotated[str, spec(description="Directory path to list")],
) -> DirectoryListing:
    """List direct children of a directory."""
    root = Path(path).expanduser()
    entries: list[DirectoryEntry] = []
    try:
        children = sorted(root.iterdir())
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    for child in children:
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


@tool(tags=["dev"])
def read_text_file(
    path: Annotated[str, spec(description="UTF-8 text file path to read")],
) -> TextFile:
    """Read a UTF-8 text file exactly as stored on disk."""
    target = Path(path).expanduser()
    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except UnicodeError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return TextFile(path=str(target), content=content)


@tool(
    tags=["dev"],
    require_approval=True,
    approval_policy="filesystem_write",
)
def write_text_file(
    path: Annotated[str, spec(description="UTF-8 text file path to write")],
    content: Annotated[str, spec(description="Complete file content to write")],
) -> FileWriteResult:
    """Write complete UTF-8 text content to a file."""
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return FileWriteResult(path=str(target), bytes_written=bytes_written)


@tool(
    tags=["dev"],
    require_approval=True,
    approval_policy="filesystem_write",
)
def replace_text_in_file(
    path: Annotated[str, spec(description="UTF-8 text file path to edit")],
    old_text: Annotated[str, spec(description="Exact text to replace")],
    new_text: Annotated[str, spec(description="Replacement text")],
) -> TextReplacementResult:
    """Replace exact text in a UTF-8 file."""
    target = Path(path).expanduser()
    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except UnicodeError as exc:
        raise ToolExecutionError(str(exc)) from exc
    if old_text not in content:
        raise ToolExecutionError("old_text was not found in file")
    updated = content.replace(old_text, new_text)
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return TextReplacementResult(
        path=str(target),
        replacements=content.count(old_text),
    )


@tool(
    tags=["dev"],
    require_approval=True,
    approval_policy="shell_command",
)
def run_shell_command(
    command: Annotated[str, spec(description="Shell command to execute")],
    cwd: Annotated[str, spec(description="Working directory for the command")] = ".",
    timeout_seconds: Annotated[int, spec(description="Command timeout in seconds")] = 120,
) -> CommandResult:
    """Run a shell command and return stdout, stderr, and exit code."""
    try:
        completed = subprocess.run(
            command,
            cwd=Path(cwd).expanduser(),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(str(exc)) from exc
    return CommandResult(
        command=command,
        cwd=cwd,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@tool(tags=["dev"])
def search_text(
    query: Annotated[str, spec(description="ripgrep search pattern")],
    path: Annotated[str, spec(description="File or directory path to search")] = ".",
) -> SearchResult:
    """Search text with ripgrep and return line-numbered matches."""
    try:
        completed = subprocess.run(
            ["rg", "--line-number", "--column", query, path],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return SearchResult(
        query=query,
        path=path,
        exit_code=completed.returncode,
        matches=completed.stdout,
        errors=completed.stderr,
    )


def default_agent_tools(
    tool_permissions: dict[str, ToolPermission] | None = None,
    tool_enabled: dict[str, bool] | None = None,
    tool_max_calls: dict[str, int] | None = None,
) -> list[Tool[Any, Any]]:
    tools = [
        list_directory,
        read_text_file,
        write_text_file,
        replace_text_in_file,
        preview_patch,
        apply_patch,
        git_status,
        git_diff,
        run_shell_command,
        search_text,
    ]
    if tool_permissions is None and tool_enabled is None and tool_max_calls is None:
        return tools
    return _configure_tools(
        tools,
        tool_permissions or {},
        tool_enabled or {},
        tool_max_calls or {},
    )


def _configure_tools(
    tools: list[Tool[Any, Any]],
    tool_permissions: dict[str, ToolPermission],
    tool_enabled: dict[str, bool],
    tool_max_calls: dict[str, int],
) -> list[Tool[Any, Any]]:
    configured_tools: list[Tool[Any, Any]] = []
    for configured_tool in tools:
        if configured_tool.name in tool_enabled and not tool_enabled[configured_tool.name]:
            continue
        permission = tool_permissions.get(configured_tool.name)
        has_permission_override = permission is not None
        has_max_call_override = configured_tool.name in tool_max_calls
        if not has_permission_override and not has_max_call_override:
            configured_tools.append(configured_tool)
            continue
        require_approval = (
            permission == "ask"
            if has_permission_override
            else configured_tool.metadata.require_approval
        )
        max_calls_per_run = (
            tool_max_calls[configured_tool.name]
            if has_max_call_override
            else configured_tool.metadata.max_calls_per_run
        )
        configured_tools.append(
            configured_tool.with_metadata(
                ToolMeta(
                    description=configured_tool.metadata.description,
                    max_calls_per_run=max_calls_per_run,
                    record_in_history=configured_tool.metadata.record_in_history,
                    require_approval=require_approval,
                    approval_policy=configured_tool.metadata.approval_policy,
                    tags=configured_tool.metadata.tags,
                )
            )
        )
    return configured_tools
