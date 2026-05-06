import difflib
from pathlib import Path
from typing import Literal

from msgspec import Struct, field

from aceai.core.executor import ToolExecutionError
from aceai.core.tools import Annotated, spec, tool


PatchActionKind = Literal["add", "delete", "update", "move"]


class PatchFileChange(Struct, frozen=True, kw_only=True):
    path: str
    action: PatchActionKind
    diff: str


class PatchPreview(Struct, frozen=True, kw_only=True):
    cwd: str
    affected_paths: list[str]
    changes: list[PatchFileChange]
    diff: str


class PatchApplyResult(Struct, frozen=True, kw_only=True):
    cwd: str
    affected_paths: list[str]
    changes: list[PatchFileChange]
    diff: str
    applied: bool


class _HunkLine(Struct, frozen=True, kw_only=True):
    marker: Literal["context", "delete", "add"]
    text: str


class _Hunk(Struct, frozen=True, kw_only=True):
    lines: list[_HunkLine]


class _PatchOperation(Struct, frozen=True, kw_only=True):
    action: PatchActionKind
    path: str
    hunks: list[_Hunk] = field(default_factory=list[_Hunk])
    add_lines: list[str] = field(default_factory=list[str])
    move_path: str = ""


class _ResolvedOperation(Struct, frozen=True, kw_only=True):
    operation: _PatchOperation
    path: Path
    move_path: Path | None = None


class _Simulation(Struct, frozen=True, kw_only=True):
    cwd: Path
    affected_paths: list[str]
    changes: list[PatchFileChange]
    writes: dict[Path, str]
    deletes: list[Path]

    @property
    def diff(self) -> str:
        return "".join(change.diff for change in self.changes)


@tool(tags=["agent_app", "filesystem", "patch"])
def preview_patch(
    patch: Annotated[str, spec(description="Complete Codex-style patch text")],
    cwd: Annotated[str, spec(description="Workspace directory for relative patch paths")] = ".",
) -> PatchPreview:
    """Preview a Codex-style patch without changing files."""
    simulation = _simulate_patch(patch, cwd)
    return PatchPreview(
        cwd=str(simulation.cwd),
        affected_paths=simulation.affected_paths,
        changes=simulation.changes,
        diff=simulation.diff,
    )


@tool(
    tags=["agent_app", "filesystem", "patch"],
    require_approval=True,
    approval_policy="filesystem_patch",
)
def apply_patch(
    patch: Annotated[str, spec(description="Complete Codex-style patch text")],
    cwd: Annotated[str, spec(description="Workspace directory for relative patch paths")] = ".",
) -> PatchApplyResult:
    """Apply a reviewed Codex-style patch to files under cwd."""
    simulation = _simulate_patch(patch, cwd)
    for path, content in simulation.writes.items():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(str(exc)) from exc
    for path in simulation.deletes:
        try:
            path.unlink()
        except OSError as exc:
            raise ToolExecutionError(str(exc)) from exc
    return PatchApplyResult(
        cwd=str(simulation.cwd),
        affected_paths=simulation.affected_paths,
        changes=simulation.changes,
        diff=simulation.diff,
        applied=True,
    )


def _simulate_patch(patch: str, cwd: str) -> _Simulation:
    root = Path(cwd).expanduser().resolve()
    operations = _parse_patch(patch)
    resolved = [_resolve_operation(root, operation) for operation in operations]
    writes: dict[Path, str] = {}
    deletes: list[Path] = []
    changes: list[PatchFileChange] = []

    for item in resolved:
        operation = item.operation
        source_path = item.path
        target_path = item.move_path or source_path
        if operation.action == "add":
            if source_path.exists() or source_path in writes:
                raise ToolExecutionError(f"Cannot add existing file: {operation.path}")
            old_text = ""
            new_text = _join_lines(operation.add_lines)
            writes[source_path] = new_text
            changes.append(
                _file_change(source_path, root, "add", old_text, new_text, from_file="/dev/null")
            )
        elif operation.action == "delete":
            old_text = _read_required(source_path)
            deletes.append(source_path)
            changes.append(
                _file_change(source_path, root, "delete", old_text, "", to_file="/dev/null")
            )
        elif operation.action in ("update", "move"):
            old_text = _read_required(source_path)
            new_text = _apply_hunks(old_text, operation.hunks, operation.path)
            writes[target_path] = new_text
            if target_path != source_path:
                deletes.append(source_path)
            changes.append(
                _file_change(target_path, root, operation.action, old_text, new_text)
            )
    affected_paths = _affected_paths(root, writes, deletes)
    return _Simulation(
        cwd=root,
        affected_paths=affected_paths,
        changes=changes,
        writes=writes,
        deletes=deletes,
    )


def _parse_patch(patch: str) -> list[_PatchOperation]:
    lines = patch.split("\n")
    if not lines or lines[0] != "*** Begin Patch":
        raise ToolExecutionError("Patch must start with *** Begin Patch")
    if lines[-1] == "":
        lines = lines[:-1]
    if not lines or lines[-1] != "*** End Patch":
        raise ToolExecutionError("Patch must end with *** End Patch")
    operations: list[_PatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            operation, index = _parse_add(lines, index)
        elif line.startswith("*** Delete File: "):
            operation, index = _parse_delete(lines, index)
        elif line.startswith("*** Update File: "):
            operation, index = _parse_update(lines, index)
        else:
            raise ToolExecutionError(f"Unsupported patch line: {line}")
        operations.append(operation)
    if not operations:
        raise ToolExecutionError("Patch must contain at least one file operation")
    return operations


def _parse_add(lines: list[str], index: int) -> tuple[_PatchOperation, int]:
    path = lines[index].removeprefix("*** Add File: ")
    index += 1
    add_lines: list[str] = []
    while index < len(lines) and not lines[index].startswith("***"):
        line = lines[index]
        if not line.startswith("+"):
            raise ToolExecutionError("Add file lines must start with +")
        add_lines.append(line.removeprefix("+"))
        index += 1
    return _PatchOperation(action="add", path=path, add_lines=add_lines), index


def _parse_delete(lines: list[str], index: int) -> tuple[_PatchOperation, int]:
    path = lines[index].removeprefix("*** Delete File: ")
    return _PatchOperation(action="delete", path=path), index + 1


def _parse_update(lines: list[str], index: int) -> tuple[_PatchOperation, int]:
    path = lines[index].removeprefix("*** Update File: ")
    index += 1
    move_path = ""
    action: PatchActionKind = "update"
    if lines[index].startswith("*** Move to: "):
        move_path = lines[index].removeprefix("*** Move to: ")
        action = "move"
        index += 1
    hunks: list[_Hunk] = []
    while index < len(lines) and not lines[index].startswith("***"):
        if not lines[index].startswith("@@"):
            raise ToolExecutionError("Update file hunks must start with @@")
        hunk, index = _parse_hunk(lines, index + 1)
        hunks.append(hunk)
    if not hunks:
        raise ToolExecutionError("Update file operation must include at least one hunk")
    return (
        _PatchOperation(action=action, path=path, hunks=hunks, move_path=move_path),
        index,
    )


def _parse_hunk(lines: list[str], index: int) -> tuple[_Hunk, int]:
    hunk_lines: list[_HunkLine] = []
    while index < len(lines):
        line = lines[index]
        if line.startswith("@@") or line.startswith("***"):
            break
        if line.startswith(" "):
            hunk_lines.append(_HunkLine(marker="context", text=line.removeprefix(" ")))
        elif line.startswith("-"):
            hunk_lines.append(_HunkLine(marker="delete", text=line.removeprefix("-")))
        elif line.startswith("+"):
            hunk_lines.append(_HunkLine(marker="add", text=line.removeprefix("+")))
        else:
            raise ToolExecutionError("Hunk lines must start with space, -, or +")
        index += 1
    if not hunk_lines:
        raise ToolExecutionError("Patch hunk cannot be empty")
    return _Hunk(lines=hunk_lines), index


def _resolve_operation(root: Path, operation: _PatchOperation) -> _ResolvedOperation:
    path = _resolve_patch_path(root, operation.path)
    move_path = None
    if operation.move_path:
        move_path = _resolve_patch_path(root, operation.move_path)
    return _ResolvedOperation(operation=operation, path=path, move_path=move_path)


def _resolve_patch_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        raise ToolExecutionError("Patch paths must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ToolExecutionError("Patch path escapes cwd")
    return resolved


def _read_required(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except UnicodeError as exc:
        raise ToolExecutionError(str(exc)) from exc


def _apply_hunks(content: str, hunks: list[_Hunk], patch_path: str) -> str:
    original = content.split("\n")
    if original and original[-1] == "":
        original = original[:-1]
    updated = list(original)
    cursor = 0
    for hunk in hunks:
        before = [
            line.text
            for line in hunk.lines
            if line.marker == "context" or line.marker == "delete"
        ]
        after = [
            line.text
            for line in hunk.lines
            if line.marker == "context" or line.marker == "add"
        ]
        match_index = _find_block(updated, before, cursor)
        if match_index < 0:
            raise ToolExecutionError(f"Patch context not found in {patch_path}")
        updated[match_index : match_index + len(before)] = after
        cursor = match_index + len(after)
    return _join_lines(updated)


def _find_block(lines: list[str], block: list[str], start: int) -> int:
    if not block:
        return start
    index = start
    last = len(lines) - len(block)
    while index <= last:
        if lines[index : index + len(block)] == block:
            return index
        index += 1
    return -1


def _join_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _file_change(
    path: Path,
    root: Path,
    action: PatchActionKind,
    old_text: str,
    new_text: str,
    *,
    from_file: str = "",
    to_file: str = "",
) -> PatchFileChange:
    rel = path.relative_to(root).as_posix()
    before_name = from_file or f"a/{rel}"
    after_name = to_file or f"b/{rel}"
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=before_name,
            tofile=after_name,
        )
    )
    return PatchFileChange(path=rel, action=action, diff=diff)


def _affected_paths(root: Path, writes: dict[Path, str], deletes: list[Path]) -> list[str]:
    paths = {path.relative_to(root).as_posix() for path in writes}
    paths.update(path.relative_to(root).as_posix() for path in deletes)
    return sorted(paths)
