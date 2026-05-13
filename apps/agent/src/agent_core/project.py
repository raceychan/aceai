from datetime import datetime, timezone
from pathlib import Path

from msgspec import Struct
from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import insert as sql_insert
from sqlalchemy import select as sql_select
from sqlalchemy import update as sql_update
from sqlalchemy.engine import RowMapping
from typing_extensions import Self

from aceai.core.helpers.string import uuid_str


class ProjectMetadata(Struct, frozen=True, kw_only=True):
    project_id: str
    name: str
    root_path: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: RowMapping) -> Self:
        return cls(
            project_id=row["project_id"],
            name=row["name"],
            root_path=row["root_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


_metadata = MetaData()
_projects_table = Table(
    "projects",
    _metadata,
    Column("project_id", String, primary_key=True),
    Column("name", String, nullable=False, index=True),
    Column("root_path", String, nullable=False, unique=True, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, index=True),
)


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_project_root()
        self.db_path = self.root / "projects.sqlite3"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.root.mkdir(parents=True, exist_ok=True)
        _metadata.create_all(self.engine)

    def resolve_project(self, project_dir: Path | None = None) -> ProjectMetadata:
        root_path = str((project_dir or Path.cwd()).resolve())
        query = sql_select(_projects_table).where(
            _projects_table.c.root_path == root_path
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is not None:
            return ProjectMetadata.from_row(row)
        now = _utc_now()
        project = ProjectMetadata(
            project_id=uuid_str(),
            name=Path(root_path).name,
            root_path=root_path,
            created_at=now,
            updated_at=now,
        )
        with self.engine.begin() as conn:
            conn.execute(
                sql_insert(_projects_table).values(
                    project_id=project.project_id,
                    name=project.name,
                    root_path=project.root_path,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                )
            )
        return project

    def get_project(self, project_id: str) -> ProjectMetadata:
        query = sql_select(_projects_table).where(
            _projects_table.c.project_id == project_id
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().fetchone()
        if row is None:
            raise KeyError(project_id)
        return ProjectMetadata.from_row(row)

    def list_projects(self) -> list[ProjectMetadata]:
        query = sql_select(_projects_table).order_by(
            _projects_table.c.updated_at.desc()
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().fetchall()
        return [ProjectMetadata.from_row(row) for row in rows]

    def touch_project(self, project_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                sql_update(_projects_table)
                .where(_projects_table.c.project_id == project_id)
                .values(updated_at=_utc_now())
            )


def default_project_root() -> Path:
    return Path.home() / ".aceai" / "projects"


def default_project() -> ProjectMetadata:
    return ProjectStore().resolve_project()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
