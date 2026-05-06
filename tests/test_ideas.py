from aceai.agent.ideas import IdeaStore


def test_idea_store_appends_markdown_and_lists_current_workspace(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    ideas_path = tmp_path / "ideas.md"
    store = IdeaStore(ideas_path)

    first = store.capture("fix resume default session", workspace=workspace)
    store.capture("hidden other repo idea", workspace=other_workspace)
    second = store.capture("add idea command", workspace=workspace)

    markdown = ideas_path.read_text(encoding="utf-8")
    assert markdown.startswith("# AceAI Ideas\n\n")
    assert f"## {first.created_at.isoformat()}" in markdown
    assert "workspace: " + str(workspace.resolve()) in markdown
    assert "source_session_id: \n\nfix resume default session" in markdown

    ideas = store.list_recent(workspace=workspace)

    assert [idea.created_at for idea in ideas] == [first.created_at, second.created_at]
    assert [idea.content for idea in ideas] == [
        "fix resume default session",
        "add idea command",
    ]


def test_idea_store_searches_by_keyword_within_workspace(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.md")
    store.capture("fix resume default session", workspace=workspace)
    store.capture("add idea command", workspace=workspace)

    ideas = store.search("resume", workspace=workspace)

    assert [idea.content for idea in ideas] == ["fix resume default session"]


def test_idea_store_deletes_recent_idea_by_one_based_workspace_index(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.md")
    first = store.capture("first idea", workspace=workspace)
    store.capture("second idea", workspace=workspace)

    deleted = store.delete_recent(1, workspace=workspace)

    assert deleted.content == "first idea"
    assert [idea.content for idea in store.list_recent(workspace=workspace)] == [
        "second idea"
    ]
    assert first.content not in (tmp_path / "ideas.md").read_text(encoding="utf-8")


def test_idea_store_updates_fifo_idea_by_one_based_workspace_index(tmp_path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IdeaStore(tmp_path / "ideas.md")
    first = store.capture("first idea", workspace=workspace)
    store.capture("second idea", workspace=workspace)

    updated = store.update_recent(1, "edited first idea", workspace=workspace)

    assert updated.created_at == first.created_at
    assert [idea.content for idea in store.list_recent(workspace=workspace)] == [
        "edited first idea",
        "second idea",
    ]
