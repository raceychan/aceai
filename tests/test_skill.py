from pathlib import Path

import pytest

from aceai.llm.errors import AceAIConfigurationError
from aceai.core.skills import (
    DuplicateSkillError,
    Skill,
    SkillFileFormatError,
    SkillLoader,
    SkillLoadingError,
    SkillMetaError,
    SkillRegistry,
)


def write_skill(root: Path, name: str, description: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


def test_skill_loads_metadata_without_body(tmp_path: Path) -> None:
    skill_dir = write_skill(
        tmp_path,
        "spreadsheets",
        "Use for spreadsheet files.",
        "# Full instructions",
    )

    skill = Skill(skill_dir)

    assert skill.name == "spreadsheets"
    assert skill.description == "Use for spreadsheet files."
    assert skill.path == skill_dir
    assert skill.skill_file == skill_dir / "SKILL.md"


def test_skill_reads_instructions_lazily(tmp_path: Path) -> None:
    skill_dir = write_skill(
        tmp_path,
        "documents",
        "Use for document files.",
        "# Documents\nRead this only after activation.",
    )
    skill = Skill(skill_dir)

    assert skill.read_instructions() == "# Documents\nRead this only after activation."


def test_skill_loader_rejects_missing_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Bad", encoding="utf-8")

    with pytest.raises(SkillFileFormatError, match="must start"):
        Skill(skill_dir).metadata


def test_skill_loading_errors_are_configuration_errors() -> None:
    assert issubclass(SkillLoadingError, AceAIConfigurationError)
    assert issubclass(SkillFileFormatError, SkillLoadingError)
    assert issubclass(SkillMetaError, SkillLoadingError)
    assert issubclass(DuplicateSkillError, SkillLoadingError)


def test_skill_rejects_missing_required_metadata(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: bad",
                "---",
                "# Bad",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillMetaError, match="description"):
        Skill(skill_dir).metadata


def test_skill_requires_string_metadata_fields(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: 123",
                "description: Bad skill.",
                "---",
                "# Bad",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillMetaError, match="must be a string"):
        Skill(skill_dir).metadata


def test_skill_loader_loads_parent_directory(tmp_path: Path) -> None:
    write_skill(tmp_path, "spreadsheets", "Use for spreadsheet files.", "# Sheets")
    write_skill(tmp_path, "documents", "Use for document files.", "# Docs")
    (tmp_path / "not-a-skill").mkdir()

    registry = SkillLoader(tmp_path).load_skills()

    assert [skill.name for skill in registry.get_skills()] == [
        "documents",
        "spreadsheets",
    ]


def test_skill_registry_loads_skill_directories(tmp_path: Path) -> None:
    write_skill(tmp_path, "spreadsheets", "Use for spreadsheet files.", "# Sheets")
    write_skill(tmp_path, "documents", "Use for document files.", "# Docs")
    (tmp_path / "not-a-skill").mkdir()

    registry = SkillRegistry()
    registry.load_dir(tmp_path)

    assert [skill.name for skill in registry.get_skills()] == [
        "documents",
        "spreadsheets",
    ]
    assert registry.get("documents").description == "Use for document files."


def test_skill_registry_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    first = write_skill(tmp_path, "first", "First skill.", "# First")
    second = write_skill(tmp_path, "second", "Second skill.", "# Second")
    second_file = second / "SKILL.md"
    second_file.write_text(
        "\n".join(
            [
                "---",
                "name: first",
                "description: Duplicate name.",
                "---",
                "# Duplicate",
            ]
        ),
        encoding="utf-8",
    )

    registry = SkillRegistry()
    registry.register(Skill(first))

    with pytest.raises(DuplicateSkillError, match="duplicated"):
        registry.register(Skill(second))


def test_skill_registry_parses_and_resolves_mentions(tmp_path: Path) -> None:
    write_skill(tmp_path, "spreadsheets", "Use for spreadsheet files.", "# Sheets")
    write_skill(tmp_path, "documents", "Use for document files.", "# Docs")
    registry = SkillRegistry()
    registry.load_dir(tmp_path)

    assert registry.parse_skill_mentions(
        "$spreadsheets analyze this with $documents, then $spreadsheets."
    ) == ["spreadsheets", "documents"]

    assert [skill.name for skill in registry.resolve_mentions("$documents now")] == [
        "documents"
    ]
