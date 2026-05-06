from datetime import datetime, timezone
from pathlib import Path

from msgspec import Struct


class Idea(Struct, frozen=True, kw_only=True):
    created_at: datetime
    workspace: str
    content: str
    source_session_id: str | None = None


class IdeaStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_ideas_path()

    def capture(
        self,
        content: str,
        *,
        workspace: Path | None = None,
        source_session_id: str | None = None,
    ) -> Idea:
        idea = Idea(
            created_at=datetime.now(timezone.utc),
            workspace=str((workspace or Path.cwd()).resolve()),
            content=content,
            source_session_id=source_session_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            if self.path.stat().st_size == 0:
                file.write("# AceAI Ideas\n\n")
            file.write(_idea_to_markdown(idea))
        return idea

    def list_recent(
        self,
        *,
        workspace: Path | None = None,
        limit: int = 20,
    ) -> list[Idea]:
        workspace_text = str((workspace or Path.cwd()).resolve())
        ideas = [
            idea
            for idea in self._read_all()
            if idea.workspace == workspace_text
        ]
        ideas.sort(key=lambda idea: idea.created_at, reverse=True)
        return ideas[:limit]

    def search(
        self,
        query: str,
        *,
        workspace: Path | None = None,
        limit: int = 20,
    ) -> list[Idea]:
        workspace_text = str((workspace or Path.cwd()).resolve())
        ideas = [
            idea
            for idea in self._read_all()
            if idea.workspace == workspace_text and query in idea.content
        ]
        ideas.sort(key=lambda idea: idea.created_at, reverse=True)
        return ideas[:limit]

    def delete_recent(self, index: int, *, workspace: Path | None = None) -> Idea:
        if index < 1:
            raise IndexError("Idea index must be one-based")
        ideas = self.list_recent(workspace=workspace)
        if index > len(ideas):
            raise IndexError("Idea index is out of range")
        idea = ideas[index - 1]
        remaining = [stored for stored in self._read_all() if stored != idea]
        self._write_all(remaining)
        return idea

    def _read_all(self) -> list[Idea]:
        if not self.path.exists():
            return []
        return _ideas_from_markdown(self.path.read_text(encoding="utf-8"))

    def _write_all(self, ideas: list[Idea]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            file.write("# AceAI Ideas\n\n")
            for idea in ideas:
                file.write(_idea_to_markdown(idea))


def default_ideas_path() -> Path:
    return Path.home() / ".aceai" / "memory" / "ideas.md"


def _idea_to_markdown(idea: Idea) -> str:
    source_session_id = idea.source_session_id
    if source_session_id is None:
        source_session_id = ""
    return (
        f"## {idea.created_at.isoformat()}\n"
        f"workspace: {idea.workspace}\n"
        f"source_session_id: {source_session_id}\n\n"
        f"{idea.content}\n\n"
    )


def _ideas_from_markdown(text: str) -> list[Idea]:
    ideas: list[Idea] = []
    current_created_at = ""
    current_workspace = ""
    current_source_session_id: str | None = None
    current_content: list[str] = []
    in_content = False
    for line in text.splitlines():
        if line.startswith("## "):
            if current_created_at != "":
                ideas.append(
                    _idea_from_parts(
                        current_created_at,
                        current_workspace,
                        current_source_session_id,
                        current_content,
                    )
                )
            current_created_at = line.removeprefix("## ")
            current_workspace = ""
            current_source_session_id = None
            current_content = []
            in_content = False
        elif line.startswith("workspace: ") and not in_content:
            current_workspace = line.removeprefix("workspace: ")
        elif line.startswith("source_session_id: ") and not in_content:
            value = line.removeprefix("source_session_id: ")
            current_source_session_id = value if value != "" else None
        elif current_created_at != "":
            if not in_content and line == "":
                in_content = True
            elif in_content:
                current_content.append(line)
    if current_created_at != "":
        ideas.append(
            _idea_from_parts(
                current_created_at,
                current_workspace,
                current_source_session_id,
                current_content,
            )
        )
    return ideas


def _idea_from_parts(
    created_at: str,
    workspace: str,
    source_session_id: str | None,
    content_lines: list[str],
) -> Idea:
    normalized_content_lines = content_lines
    if normalized_content_lines and normalized_content_lines[-1] == "":
        normalized_content_lines = normalized_content_lines[:-1]
    return Idea(
        created_at=datetime.fromisoformat(created_at),
        workspace=workspace,
        source_session_id=source_session_id,
        content="\n".join(normalized_content_lines),
    )
