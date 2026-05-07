import re
from html import escape
from pathlib import Path
from typing import Any, Callable, Literal, cast

from msgspec import Struct
import yaml

from aceai.llm.errors import AceAIConfigurationError
from aceai.core.tools import Annotated, spec, tool


class SkillLoadingError(AceAIConfigurationError):
    """Raised when a skill cannot be loaded from disk."""


class SkillFileFormatError(SkillLoadingError):
    """Raised when SKILL.md does not follow the required file structure."""


class SkillMetaError(SkillLoadingError):
    """Raised when skill frontmatter metadata is invalid."""


class DuplicateSkillError(SkillLoadingError):
    """Raised when two loaded skills declare the same name."""


class SkillMeta(Struct, frozen=True, kw_only=True):
    name: str
    description: str

    @classmethod
    def from_frontmatter(cls, frontmatter: str) -> "SkillMeta":
        fields = yaml.safe_load(frontmatter)
        if not isinstance(fields, dict):
            raise SkillMetaError("Skill frontmatter must be a mapping")
        fields = cast(dict[str, Any], fields)
        if "name" not in fields:
            raise SkillMetaError("Skill frontmatter must define 'name'")
        if "description" not in fields:
            raise SkillMetaError("Skill frontmatter must define 'description'")
        for k, v in fields.items():
            if not isinstance(v, str):
                raise SkillMetaError(f"Skill frontmatter field {k!r} must be a string")
            if v == "":
                raise SkillMetaError(f"Skill frontmatter field {k!r} cannot be empty")
        return cls(name=fields["name"], description=fields["description"])


class SkillListItem(Struct, frozen=True, kw_only=True):
    name: str
    description: str
    location: str


class SkillsListResult(Struct, frozen=True, kw_only=True):
    skills: list[SkillListItem]
    hint: str


class SkillViewResult(Struct, frozen=True, kw_only=True):
    name: str
    content: str
    location: str
    skill_dir: str
    references_dir: str
    scripts_dir: str
    assets_dir: str


class Skill:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.skill_file = self.path / "SKILL.md"
        frontmatter, instruction = self.split_skill_file()
        self._metadata: SkillMeta  = SkillMeta.from_frontmatter(frontmatter)
        self._instruction = instruction

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def scripts_dir(self) -> Path:
        return self.path / "scripts"

    @property
    def references_dir(self) -> Path:
        return self.path / "references"

    @property
    def assets_dir(self) -> Path:
        return self.path / "assets"

    @property
    def metadata(self) -> SkillMeta:
        return self._metadata

    @property
    def instruction(self) -> str:
        return self._instruction

    def read_instructions(self) -> str:
        return self._instruction

    def read_file(self, file_path: str) -> str:
        target = (self.path / file_path).resolve()
        skill_root = self.path.resolve()
        target.relative_to(skill_root)
        return target.read_text(encoding="utf-8")

    def split_skill_file(self) -> tuple[str, str]:
        text = self.skill_file.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise SkillFileFormatError(
                f"Skill file {self.skill_file} must start with YAML frontmatter"
            )
        parts = text.split("---\n", 2)
        if len(parts) != 3:
            raise SkillFileFormatError(
                f"Skill file {self.skill_file} must contain closing frontmatter"
            )
        return parts[1], parts[2]

class SkillLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def resolve_paths(
        cls,
        skill_path: str | Path | Literal["auto", "disable"],
    ) -> list[Path]:
        global_skills = Path.home() / ".aceai" / "skills"
        if skill_path == "auto":
            return [global_skills, Path.cwd() / ".agents" / "skills"]
        if skill_path == "disable":
            return []
        return [global_skills, Path(skill_path).expanduser()]

    @classmethod
    def load_registry(
        cls,
        skill_path: str | Path | Literal["auto", "disable"],
        loader_factory: Callable[[str], "SkillLoader"] | None = None,
        extra_skill_paths: tuple[Path, ...] = (),
    ) -> "SkillRegistry":
        registry = SkillRegistry()
        if skill_path == "disable":
            return registry
        for path in cls.resolve_paths(skill_path):
            loader = cls(str(path)) if loader_factory is None else loader_factory(str(path))
            registry.register(*loader.load_skills().get_skills())
        for path in extra_skill_paths:
            loader = cls(str(path)) if loader_factory is None else loader_factory(str(path))
            registry.register_missing(*loader.load_skills().get_skills())
        return registry

    def load_skills(self) -> "SkillRegistry":
        registry = SkillRegistry()
        if not self.path.exists():
            return registry
        if not self.path.is_dir():
            raise SkillLoadingError(f"Skill path {self.path} must be a directory")
        for child in sorted(self.path.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.exists():
                continue
            skill = Skill(child)
            skill.metadata
            registry.register(skill)
        return registry


class SkillRegistry:
    def __init__(self, *skills: Skill) -> None:
        self._skills: dict[str, Skill] = {}
        if skills:
            self.register(*skills)

    @property
    def skills(self) -> dict[str, Skill]:
        return self._skills

    def register(self, *skills: Skill) -> None:
        for skill in skills:
            if skill.name in self._skills:
                raise DuplicateSkillError(f"Skill {skill.name!r} is duplicated")
            self._skills[skill.name] = skill

    def register_missing(self, *skills: Skill) -> None:
        for skill in skills:
            if skill.name in self._skills:
                continue
            self._skills[skill.name] = skill

    def load_dir(self, path: Path) -> None:
        self.register(*SkillLoader(path).load_skills().get_skills())

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def get_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def select(self, names: tuple[str, ...]) -> "SkillRegistry":
        selected = SkillRegistry()
        selected_names = set(names)
        for name in names:
            self.get(name)
        selected.register(
            *[
                skill
                for skill in self.get_skills()
                if skill.name in selected_names
            ]
        )
        return selected

    def parse_skill_mentions(self, text: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"\$([A-Za-z0-9][A-Za-z0-9_-]*)", text):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def resolve_mentions(self, text: str) -> list[Skill]:
        return [self.get(name) for name in self.parse_skill_mentions(text)]

    def as_tools(self):
        @tool
        def skills_list() -> SkillsListResult:
            """List available skills. Use skill_view(name) to load full instructions."""
            return SkillsListResult(
                skills=[
                    SkillListItem(
                        name=skill.name,
                        description=skill.description,
                        location=str(skill.skill_file),
                    )
                    for skill in self.get_skills()
                ],
                hint="Use skill_view(name) to load a skill's full instructions.",
            )

        @tool
        def skill_view(
            name: Annotated[str, spec(description="Skill name to load")],
            file_path: Annotated[
                str,
                spec(
                    description=(
                        "Optional file path within the skill directory, such as "
                        "references/api.md, scripts/helper.py, or assets/template.json"
                    )
                ),
            ] = "",
        ) -> SkillViewResult:
            """Load full skill instructions, or a supporting file inside the skill."""
            skill = self.get(name)
            content = (
                skill.read_file(file_path) if file_path else skill.read_instructions()
            )
            return SkillViewResult(
                name=skill.name,
                content=content,
                location=str(
                    skill.skill_file if file_path == "" else skill.path / file_path
                ),
                skill_dir=str(skill.path),
                references_dir=str(skill.references_dir),
                scripts_dir=str(skill.scripts_dir),
                assets_dir=str(skill.assets_dir),
            )

        return [skills_list, skill_view]


def format_skills_for_prompt(registry: SkillRegistry) -> str:
    skills = registry.get_skills()
    if not skills:
        return ""

    lines = [
        "",
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the skill_view tool to load a skill's full instructions when the task matches its description.",
        "When a skill references relative paths, resolve them against that skill directory.",
        "",
        "<available_skills>",
    ]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape(skill.name)}</name>")
        lines.append(f"    <description>{escape(skill.description)}</description>")
        lines.append(f"    <location>{escape(str(skill.skill_file))}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)
