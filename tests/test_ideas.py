from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy import insert as sql_insert

from aceai.agent.ideas import Idea, IdeaStore, ideas_to_markdown
from aceai.agent.project import ProjectMetadata


def test_idea_store_persists_structured_ideas_and_renders_markdown(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    ideas_path = tmp_path / "ideas.sqlite3"
    store = IdeaStore(ideas_path)

    first = store.capture("fix resume default session", workspace=workspace)
    store.capture("hidden other repo idea", workspace=other_workspace)
    second = store.capture("add idea command", workspace=workspace)

    ideas = store.list_recent(workspace=workspace)
    markdown = ideas_to_markdown(ideas)

    assert ideas_path.exists()
    assert markdown.startswith("# AceAI Ideas\n\n")
    assert f"## {first.created_at.isoformat()}" in markdown
    assert f"idea_id: {first.idea_id}" in markdown
    assert f"project_id: {first.project_id}" in markdown
    assert "project: repo" in markdown
    assert "workspace: " + str(workspace.resolve()) in markdown
    assert "source_session_id: \n\nfix resume default session" in markdown

    assert [idea.created_at for idea in ideas] == [first.created_at, second.created_at]
    assert [idea.content for idea in ideas] == [
        "fix resume default session",
        "add idea command",
    ]


def test_idea_store_filters_by_project_id(tmp_path) -> None:
    ioa = tmp_path / "ioa"
    other_ioa = tmp_path / "work" / "ioa"
    ioa.mkdir()
    other_ioa.mkdir(parents=True)
    store = IdeaStore(tmp_path / "ideas.sqlite3")
    first = store.capture("first ioa idea", workspace=ioa)
    store.capture("other ioa idea", workspace=other_ioa)

    ideas = store.list_recent(workspace=ioa)

    assert [idea.project_id for idea in ideas] == [first.project_id]
    assert [idea.content for idea in ideas] == ["first ioa idea"]


def test_idea_store_lists_all_ideas_with_current_project_first(tmp_path) -> None:
    ioa = tmp_path / "ioa"
    aceai = tmp_path / "aceai"
    ioa.mkdir()
    aceai.mkdir()
    store = IdeaStore(tmp_path / "ideas.sqlite3")
    aceai_idea = store.capture("aceai idea", workspace=aceai)
    ioa_idea = store.capture("ioa idea", workspace=ioa)

    ideas = store.list_for_display(current_project=_project_from_idea(ioa_idea))

    assert [idea.content for idea in ideas] == ["ioa idea", "aceai idea"]
    assert ideas[0].project_id == ioa_idea.project_id
    assert ideas[1].project_id == aceai_idea.project_id


def test_idea_store_backfills_empty_project_to_current_project(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    engine = create_engine(f"sqlite:///{ideas_path}")
    metadata = MetaData()
    ideas_table = Table(
        "ideas",
        metadata,
        Column("idea_id", String, primary_key=True),
        Column("project_id", String, nullable=False),
        Column("project_name", String, nullable=False),
        Column("workspace", String, nullable=False),
        Column("source_session_id", String, nullable=True),
        Column("content", String, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            sql_insert(ideas_table).values(
                idea_id="idea-1",
                project_id="",
                project_name="",
                workspace=str(tmp_path),
                source_session_id=None,
                content="old idea",
                created_at=datetime.now(timezone.utc),
            )
        )

    store = IdeaStore(ideas_path)
    idea = store.list_recent()[0]

    assert idea.project_id != ""
    assert idea.project_name == "aceai"


def test_idea_store_imports_legacy_markdown_as_current_project(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    legacy_path = tmp_path / "ideas.md"
    legacy_path.write_text(
        "# AceAI Ideas\n\n"
        "## 2026-05-07T20:12:06+00:00\n"
        f"workspace: {tmp_path}\n"
        "source_session_id: session-1\n\n"
        "legacy idea\n\n",
        encoding="utf-8",
    )

    store = IdeaStore(ideas_path)
    idea = store.list_recent()[0]

    assert idea.content == "legacy idea"
    assert idea.source_session_id == "session-1"
    assert idea.project_id != ""
    assert idea.project_name == "aceai"


def test_idea_store_searches_by_keyword_within_workspace(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.sqlite3")
    store.capture("fix resume default session", workspace=workspace)
    store.capture("add idea command", workspace=workspace)

    ideas = store.search("resume", workspace=workspace)

    assert [idea.content for idea in ideas] == ["fix resume default session"]


def test_idea_store_deletes_recent_idea_by_one_based_workspace_index(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.sqlite3")
    first = store.capture("first idea", workspace=workspace)
    store.capture("second idea", workspace=workspace)

    deleted = store.delete_recent(1, workspace=workspace)

    assert deleted.content == "first idea"
    assert [idea.content for idea in store.list_recent(workspace=workspace)] == [
        "second idea"
    ]
    assert first.content not in store.render_markdown(workspace=workspace)


def test_idea_store_updates_fifo_idea_by_one_based_workspace_index(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.sqlite3")
    first = store.capture("first idea", workspace=workspace)
    store.capture("second idea", workspace=workspace)

    updated = store.update_recent(1, "edited first idea", workspace=workspace)

    assert updated.created_at == first.created_at
    assert [idea.content for idea in store.list_recent(workspace=workspace)] == [
        "edited first idea",
        "second idea",
    ]


def test_idea_store_renders_with_custom_renderer(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.sqlite3")
    store.capture("first idea", workspace=workspace)
    store.capture("second idea", workspace=workspace)

    rendered = store.render(ContentOnlyRenderer(), workspace=workspace)

    assert rendered == "first idea|second idea"


class ContentOnlyRenderer:
    def render(self, ideas: list[Idea]) -> str:
        return "|".join(idea.content for idea in ideas)


def _project_from_idea(idea: Idea) -> ProjectMetadata:
    return ProjectMetadata(
        project_id=idea.project_id,
        name=idea.project_name,
        root_path=idea.workspace,
        created_at=idea.created_at,
        updated_at=idea.created_at,
    )
