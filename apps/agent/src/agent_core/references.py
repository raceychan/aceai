"""Shared app-layer helpers for inline reference candidates."""

import os
from pathlib import Path

from msgspec import Struct
from rapidfuzz import fuzz

from agent_core.memory.ideas import Idea

REFERENCE_IGNORED_DIRS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "wheels",
}


class ReferenceCandidate(Struct, frozen=True, kw_only=True):
    kind: str
    value: str
    label: str
    description: str
    idea_id: str | None = None


def reference_candidates(
    *,
    root: Path,
    ideas: list[Idea],
    query: str,
    kind: str,
    limit: int,
    score_cutoff: float = 50,
) -> tuple[ReferenceCandidate, ...]:
    query, kind = normalize_reference_query(query, kind)
    candidates: list[ReferenceCandidate] = []
    if kind == "all" or kind == "file":
        candidates.extend(file_reference_candidates(root))
    if kind == "all" or kind == "idea":
        candidates.extend(idea_reference_candidates(ideas))
    if query == "":
        return tuple(candidates[:limit])
    ranked = sorted(
        (
            (candidate, _reference_score(query, candidate), index)
            for index, candidate in enumerate(candidates)
        ),
        key=lambda match: (-match[1], match[2]),
    )
    return tuple(
        candidate
        for candidate, score, _index in ranked
        if score >= score_cutoff
    )[:limit]


def normalize_reference_query(query: str, kind: str) -> tuple[str, str]:
    if query.startswith("@"):
        query = query[1:]
    if query.startswith("idea:"):
        return query.removeprefix("idea:"), "idea"
    return query, kind


def file_reference_candidates(root: Path) -> tuple[ReferenceCandidate, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        ReferenceCandidate(
            kind="file",
            value="@" + path.relative_to(root).as_posix(),
            label=path.relative_to(root).as_posix(),
            description="file",
        )
        for path in iter_reference_file_candidates(root)
    )


def idea_reference_candidates(ideas: list[Idea]) -> tuple[ReferenceCandidate, ...]:
    return tuple(
        ReferenceCandidate(
            kind="idea",
            value=f"@idea:{index}",
            label=f"@idea:{index}",
            description=reference_idea_description(idea, limit=120),
            idea_id=idea.idea_id,
        )
        for index, idea in enumerate(ideas, start=1)
    )


def reference_idea_description(idea: Idea, *, limit: int) -> str:
    first_line = idea.content.splitlines()[0] if idea.content.splitlines() else "idea"
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 3] + "..."


def iter_reference_file_candidates(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname for dirname in dirnames if dirname not in REFERENCE_IGNORED_DIRS
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if not path.is_file():
                continue
            yield path


def _reference_score(query: str, candidate: ReferenceCandidate) -> float:
    normalized_query = query.casefold()
    return max(
        fuzz.WRatio(normalized_query, candidate.value.removeprefix("@").casefold()),
        fuzz.WRatio(normalized_query, candidate.label.casefold()),
        fuzz.WRatio(normalized_query, candidate.description.casefold()),
    )
