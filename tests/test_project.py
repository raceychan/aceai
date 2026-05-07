from aceai.agent.project import ProjectStore


def test_project_store_resolves_current_directory_to_stable_project(tmp_path) -> None:
    store = ProjectStore(tmp_path / "projects")
    workspace = tmp_path / "ioa"
    workspace.mkdir()

    first = store.resolve_project(workspace)
    second = store.resolve_project(workspace)

    assert first.project_id == second.project_id
    assert first.name == "ioa"
    assert first.root_path == str(workspace.resolve())


def test_project_store_lists_projects_by_recent_update(tmp_path) -> None:
    store = ProjectStore(tmp_path / "projects")
    first_workspace = tmp_path / "ioa"
    second_workspace = tmp_path / "aceai"
    first_workspace.mkdir()
    second_workspace.mkdir()

    first = store.resolve_project(first_workspace)
    second = store.resolve_project(second_workspace)
    store.touch_project(first.project_id)

    projects = store.list_projects()

    assert [project.project_id for project in projects] == [
        first.project_id,
        second.project_id,
    ]
