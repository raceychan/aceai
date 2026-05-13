from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from msgspec import Struct
from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import delete as sql_delete
from sqlalchemy import insert as sql_insert
from sqlalchemy import select as sql_select
from sqlalchemy import update as sql_update
from sqlalchemy import inspect as sql_inspect
from sqlalchemy.engine import RowMapping
from typing_extensions import Self

from agent_core.project import ProjectMetadata, ProjectStore
from aceai.core.helpers.string import uuid_str


class Idea(Struct, frozen=True, kw_only=True):
    idea_id: str
    created_at: datetime
    project_id: str
    project_name: str
    workspace: str
    content: str
    source_session_id: str | None = None

    @classmethod
    def from_row(cls, row: RowMapping) -> Self:
        created_at = row["created_at"]
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return cls(
            idea_id=row["idea_id"],
            created_at=created_at,
            project_id=row["project_id"],
            project_name=row["project_name"],
            workspace=row["workspace"],
            content=row["content"],
            source_session_id=row["source_session_id"],
        )


class IdeaRenderer(Protocol):
    def render(self, ideas: list[Idea]) -> str: ...


class MarkdownIdeaRenderer:
    def render(self, ideas: list[Idea]) -> str:
        lines = ["# AceAI Ideas", ""]
        for idea in ideas:
            lines.extend(self.render_idea_lines(idea))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def render_idea_lines(self, idea: Idea) -> list[str]:
        source_session_id = idea.source_session_id
        if source_session_id is None:
            source_session_id = ""
        return [
            f"## {idea.created_at.isoformat()}",
            f"idea_id: {idea.idea_id}",
            f"project_id: {idea.project_id}",
            f"project: {idea.project_name}",
            f"workspace: {idea.workspace}",
            f"source_session_id: {source_session_id}",
            "",
            idea.content,
        ]


_metadata = MetaData()
_ideas_table = Table(
    "ideas",
    _metadata,
    Column("idea_id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("project_name", String, nullable=False),
    Column("workspace", String, nullable=False),
    Column("source_session_id", String, nullable=True, index=True),
    Column("content", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
)


class IdeaStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_ideas_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.path}")
        _metadata.create_all(self.engine)
        self._backfill_missing_project()
        self._import_legacy_markdown()

    def capture(
        self,
        content: str,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
        source_session_id: str | None = None,
    ) -> Idea:
        selected_project = project or self._resolve_project(workspace)
        idea = Idea(
            idea_id=uuid_str(),
            created_at=datetime.now(timezone.utc),
            project_id=selected_project.project_id,
            project_name=selected_project.name,
            workspace=str((workspace or Path.cwd()).resolve()),
            content=content,
            source_session_id=source_session_id,
        )
        with self.engine.begin() as conn:
            conn.execute(
                sql_insert(_ideas_table).values(
                    idea_id=idea.idea_id,
                    created_at=idea.created_at,
                    project_id=idea.project_id,
                    project_name=idea.project_name,
                    workspace=idea.workspace,
                    source_session_id=idea.source_session_id,
                    content=idea.content,
                )
            )
        return idea

    def list_recent(
        self,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
        limit: int = 20,
    ) -> list[Idea]:
        query = sql_select(_ideas_table).order_by(_ideas_table.c.created_at.asc())
        if project is not None or workspace is not None:
            selected_project = project or self._resolve_project(workspace)
            query = query.where(_ideas_table.c.project_id == selected_project.project_id)
        query = query.limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        return [Idea.from_row(row) for row in rows]

    def list_for_display(
        self,
        *,
        current_project: ProjectMetadata,
        limit: int = 20,
    ) -> list[Idea]:
        query = sql_select(_ideas_table).order_by(_ideas_table.c.created_at.asc())
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        ideas = [Idea.from_row(row) for row in rows]
        ideas.sort(
            key=lambda idea: (
                0 if idea.project_id == current_project.project_id else 1,
                idea.project_name,
                idea.created_at,
            )
        )
        return ideas[:limit]

    def search(
        self,
        query: str,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
        limit: int = 20,
    ) -> list[Idea]:
        selected_project = project or self._resolve_project(workspace)
        statement = (
            sql_select(_ideas_table)
            .where(_ideas_table.c.project_id == selected_project.project_id)
            .where(_ideas_table.c.content.contains(query))
            .order_by(_ideas_table.c.created_at.asc())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(statement).mappings().fetchall()
        return [Idea.from_row(row) for row in rows]

    def delete_recent(
        self,
        index: int,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
    ) -> Idea:
        if index < 1:
            raise IndexError("Idea index must be one-based")
        ideas = self.list_recent(project=project, workspace=workspace)
        if index > len(ideas):
            raise IndexError("Idea index is out of range")
        idea = ideas[index - 1]
        with self.engine.begin() as conn:
            conn.execute(
                sql_delete(_ideas_table).where(_ideas_table.c.idea_id == idea.idea_id)
            )
        return idea

    def delete_displayed(
        self,
        index: int,
        *,
        current_project: ProjectMetadata,
    ) -> Idea:
        if index < 1:
            raise IndexError("Idea index must be one-based")
        ideas = self.list_for_display(current_project=current_project)
        if index > len(ideas):
            raise IndexError("Idea index is out of range")
        idea = ideas[index - 1]
        with self.engine.begin() as conn:
            conn.execute(
                sql_delete(_ideas_table).where(_ideas_table.c.idea_id == idea.idea_id)
            )
        return idea

    def update_recent(
        self,
        index: int,
        content: str,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
    ) -> Idea:
        if index < 1:
            raise IndexError("Idea index must be one-based")
        ideas = self.list_recent(project=project, workspace=workspace)
        if index > len(ideas):
            raise IndexError("Idea index is out of range")
        idea = ideas[index - 1]
        updated = Idea(
            idea_id=idea.idea_id,
            created_at=idea.created_at,
            project_id=idea.project_id,
            project_name=idea.project_name,
            workspace=idea.workspace,
            source_session_id=idea.source_session_id,
            content=content,
        )
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_ideas_table)
                .where(_ideas_table.c.idea_id == idea.idea_id)
                .values(content=content)
            )
        return updated

    def update_displayed(
        self,
        index: int,
        content: str,
        *,
        current_project: ProjectMetadata,
    ) -> Idea:
        if index < 1:
            raise IndexError("Idea index must be one-based")
        ideas = self.list_for_display(current_project=current_project)
        if index > len(ideas):
            raise IndexError("Idea index is out of range")
        idea = ideas[index - 1]
        updated = Idea(
            idea_id=idea.idea_id,
            created_at=idea.created_at,
            project_id=idea.project_id,
            project_name=idea.project_name,
            workspace=idea.workspace,
            source_session_id=idea.source_session_id,
            content=content,
        )
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_ideas_table)
                .where(_ideas_table.c.idea_id == idea.idea_id)
                .values(content=content)
            )
        return updated

    def render_markdown(
        self,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
        limit: int = 20,
    ) -> str:
        return self.render(
            MarkdownIdeaRenderer(),
            project=project,
            workspace=workspace,
            limit=limit,
        )

    def render(
        self,
        renderer: IdeaRenderer,
        *,
        project: ProjectMetadata | None = None,
        workspace: Path | None = None,
        limit: int = 20,
    ) -> str:
        return renderer.render(
            self.list_recent(project=project, workspace=workspace, limit=limit)
        )

    def _resolve_project(self, workspace: Path | None) -> ProjectMetadata:
        return ProjectStore(self.path.parent / "projects").resolve_project(workspace)

    def _backfill_missing_project(self) -> None:
        column_names = {
            column["name"] for column in sql_inspect(self.engine).get_columns("ideas")
        }
        if "project_id" not in column_names or "project_name" not in column_names:
            return
        project = self._resolve_project(None)
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_ideas_table)
                .where(_ideas_table.c.project_id == "")
                .values(project_id=project.project_id)
            )
            conn.execute(
                sql_update(_ideas_table)
                .where(_ideas_table.c.project_name == "")
                .values(project_name=project.name)
            )

    def _import_legacy_markdown(self) -> None:
        legacy_path = self.path.with_name("ideas.md")
        if not legacy_path.exists():
            return
        query = sql_select(_ideas_table.c.idea_id).limit(1)
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is not None:
            return
        project = self._resolve_project(None)
        ideas = _legacy_ideas_from_markdown(
            legacy_path.read_text(encoding="utf-8"),
            project=project,
        )
        with self.engine.begin() as conn:
            for idea in ideas:
                conn.execute(
                    sql_insert(_ideas_table).values(
                        idea_id=idea.idea_id,
                        created_at=idea.created_at,
                        project_id=idea.project_id,
                        project_name=idea.project_name,
                        workspace=idea.workspace,
                        source_session_id=idea.source_session_id,
                        content=idea.content,
                    )
                )


def default_ideas_path() -> Path:
    return Path.home() / ".aceai" / "memory" / "ideas.sqlite3"


def ideas_to_markdown(ideas: list[Idea]) -> str:
    return MarkdownIdeaRenderer().render(ideas)


def _legacy_ideas_from_markdown(text: str, *, project: ProjectMetadata) -> list[Idea]:
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
                    _legacy_idea_from_parts(
                        current_created_at,
                        current_workspace,
                        current_source_session_id,
                        current_content,
                        project=project,
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
            _legacy_idea_from_parts(
                current_created_at,
                current_workspace,
                current_source_session_id,
                current_content,
                project=project,
            )
        )
    return ideas


def _legacy_idea_from_parts(
    created_at: str,
    workspace: str,
    source_session_id: str | None,
    content_lines: list[str],
    *,
    project: ProjectMetadata,
) -> Idea:
    stored_content_lines = content_lines
    if stored_content_lines and stored_content_lines[-1] == "":
        stored_content_lines = stored_content_lines[:-1]
    return Idea(
        idea_id=uuid_str(),
        created_at=datetime.fromisoformat(created_at),
        project_id=project.project_id,
        project_name=project.name,
        workspace=workspace,
        source_session_id=source_session_id,
        content="\n".join(stored_content_lines),
    )
