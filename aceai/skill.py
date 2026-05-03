import re
from pathlib import Path
from typing import Any

from msgspec import Struct
import yaml

from aceai.errors import AceAIConfigurationError


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


class Skill:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.skill_file = path / "SKILL.md"
        self._metadata: SkillMeta | None = None

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
        if self._metadata is None:
            self._metadata = self.load_metadata()
        return self._metadata

    def load_metadata(self) -> SkillMeta:
        frontmatter, _body = self.split_skill_file()
        fields = self.parse_frontmatter(frontmatter)
        return SkillMeta(
            name=fields["name"],
            description=fields["description"],
        )

    def read_instructions(self) -> str:
        _frontmatter, body = self.split_skill_file()
        return body

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

    def parse_frontmatter(self, frontmatter: str) -> dict[str, str]:
        fields = yaml.safe_load(frontmatter)
        if fields is None:
            raise SkillMetaError("Skill frontmatter cannot be empty")
        if not isinstance(fields, dict):
            raise SkillMetaError("Skill frontmatter must be a mapping")

        if "name" not in fields:
            raise SkillMetaError("Skill frontmatter must define 'name'")
        if "description" not in fields:
            raise SkillMetaError("Skill frontmatter must define 'description'")
        return {
            "name": self.require_string_field(fields, "name"),
            "description": self.require_string_field(fields, "description"),
        }

    def require_string_field(self, fields: dict[Any, Any], key: str) -> str:
        value = fields[key]
        if not isinstance(value, str):
            raise SkillMetaError(f"Skill frontmatter field {key!r} must be a string")
        if value == "":
            raise SkillMetaError(f"Skill frontmatter field {key!r} cannot be empty")
        return value


class SkillLoader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load_skills(self) -> "SkillRegistry":
        registry = SkillRegistry()
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

    def load_dir(self, path: Path) -> None:
        self.register(*SkillLoader(path).load_skills().get_skills())

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def get_skills(self) -> list[Skill]:
        return list(self._skills.values())

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
