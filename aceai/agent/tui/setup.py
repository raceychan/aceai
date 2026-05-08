"""Provider setup screen for the AceAI TUI."""

from datetime import datetime
import os
from pathlib import Path
from typing import Callable, Generic, TypeVar, cast

from rich import box
from rich.console import Group
from rich.cells import cell_len, set_cell_size
from rich.panel import Panel
from rich.text import Text
from msgspec import field
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from aceai.agent.ideas import Idea
from aceai.agent.provider_catalog import (
    api_key_env,
    default_model,
    reasoning_effort_options,
    supported_models,
    supported_provider_names,
    supports_reasoning_effort,
)
from aceai.agent.ace_agent import ACE_AGENT_BUILTIN_SKILL_PATHS, ACE_AGENT_SKILL_PATH
from aceai.agent.config import config_schema
from aceai.core.context_manager import CompressThreshold, ContextCompressionPolicy
from aceai.agent.permissions import TOOL_PERMISSION_OPTIONS, ToolPermission
from aceai.agent.session import SessionMetadata, SessionStore
from aceai.core.skills import Skill, SkillLoader, SkillLoadingError, SkillRegistry
from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

from aceai.agent.config import AgentAppConfig, ReasoningLevel
from aceai.agent.config import save_config
from aceai.agent.cost import format_usd
from .metadata import MetadataSection, _metadata_renderables
from .session_display import session_display_title

PanelListItem = TypeVar("PanelListItem")
PanelListRenderer = Callable[[list[PanelListItem], int], list[Text | Panel]]
IDEA_PREVIEW_LINES = 2
IDEA_PREVIEW_WIDTH = 112
REASONING_LEVEL_LABELS: dict[ReasoningLevel, str] = {
    "auto": "auto",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


class ConfigSelection(Record, kw_only=True):
    provider: str
    model: OpenAIModel
    default_model: OpenAIModel
    api_key: str
    skills: str
    skill_selection_mode: str = "all"
    enabled_skills: tuple[str, ...] = ()
    tool_permissions: dict[str, ToolPermission] = field(
        default_factory=dict[str, ToolPermission]
    )
    tool_enabled: dict[str, bool] = field(default_factory=dict[str, bool])
    tool_max_calls: dict[str, int] = field(default_factory=dict[str, int])
    compress_threshold: CompressThreshold = "100%"
    reasoning_level: ReasoningLevel = "auto"


class SkillConfigItem(Record, kw_only=True):
    name: str
    description: str
    location: str
    builtin: bool = False
    source: str = "global"


class ToolPermissionItem(Record, kw_only=True):
    name: str
    description: str
    permission: ToolPermission
    enabled: bool = True
    max_calls_per_run: int | None = None


IdeaSaveHandler = Callable[[int, str], list[Idea]]
IdeaCaptureHandler = Callable[[str], list[Idea]]
IdeaDeleteHandler = Callable[[int], list[Idea]]


def _candidate_text(candidates: tuple[str, ...], highlighted: int) -> Text:
    text = Text()
    for index, candidate in enumerate(candidates):
        style = "reverse" if index == highlighted else ""
        marker = "> " if index == highlighted else "  "
        text.append(f"{marker}{candidate}", style=style)
        if index < len(candidates) - 1:
            text.append("\n")
    return text


def _matching_candidates(candidates: tuple[str, ...], value: str) -> tuple[str, ...]:
    if value == "":
        return ()
    if value in candidates:
        return ()
    return tuple(candidate for candidate in candidates if candidate.startswith(value))


def _highlight_for_value(candidates: tuple[str, ...], value: str) -> int:
    if not candidates:
        return 0
    if value in candidates:
        return candidates.index(value)
    for index, candidate in enumerate(candidates):
        if candidate.startswith(value):
            return index
    return 0


def _move_highlight(candidates: tuple[str, ...], highlighted: int, delta: int) -> int:
    if not candidates:
        return 0
    return (highlighted + delta) % len(candidates)


def _masked_api_key(api_key: str) -> str:
    if api_key == "":
        return ""
    return f"*****************{api_key[-4:]}"


def _api_key_value_from_input(value: str, api_key: str) -> str:
    if value == _masked_api_key(api_key):
        return api_key
    return value


def _compress_threshold_input_value(value: CompressThreshold) -> str:
    return f"{value}"


def _compress_threshold_from_input(value: str) -> CompressThreshold:
    if value.endswith("%"):
        ContextCompressionPolicy(value)
        return value
    if value.count(".") == 1:
        threshold = float(value)
        ContextCompressionPolicy(threshold)
        return threshold
    if value.isdecimal():
        threshold = int(value)
        ContextCompressionPolicy(threshold)
        return threshold
    raise ValueError("compress_threshold must be a percentage, ratio, or token count")


def _skill_config_items(registry: SkillRegistry) -> tuple[SkillConfigItem, ...]:
    return tuple(
        SkillConfigItem(
            name=skill.name,
            description=skill.description,
            location=str(skill.skill_file),
            builtin=_is_builtin_skill_location(skill.skill_file),
            source=_skill_source(skill.skill_file),
        )
        for skill in registry.get_skills()
    )


def _is_builtin_skill_location(skill_file: Path) -> bool:
    return _skill_source(skill_file) == "aceai builtin"


def _skill_source(skill_file: Path) -> str:
    resolved = skill_file.resolve()
    if any(
        resolved.is_relative_to(builtin_path.resolve())
        for builtin_path in ACE_AGENT_BUILTIN_SKILL_PATHS
    ):
        return "aceai builtin"
    if _is_under(skill_file, Path.cwd() / ".agents" / "skills"):
        return "project"
    if _is_under(skill_file, Path.home() / ".aceai" / "skills"):
        return "global"
    return "project"


def _is_under(path: Path, root: Path) -> bool:
    absolute_path = path.expanduser().absolute()
    absolute_root = root.expanduser().absolute()
    return absolute_path.is_relative_to(absolute_root)


def _skill_checkboxes(
    skill_items: tuple[SkillConfigItem, ...],
    checked_items: tuple[SkillConfigItem, ...],
):
    if not skill_items:
        return (Static("No skills available", id="skills-empty"),)
    checked_names = {item.name for item in checked_items}
    controls: list[Container] = []
    for index, item in enumerate(skill_items):
        controls.append(
            Container(
                Checkbox(
                    item.name,
                    value=item.name in checked_names,
                    id=f"skill-{index}",
                ),
                Static(
                    item.description,
                    classes="skill-description",
                    id=f"skill-description-{index}",
                ),
                Static(
                    _skill_source_label(item),
                    classes=f"skill-source skill-source-{item.source.replace(' ', '-')}",
                    id=f"skill-source-{index}",
                ),
                Static(
                    item.location,
                    classes="skill-location",
                    id=f"skill-location-{index}",
                ),
                classes="skill-entry",
            )
        )
    return tuple(controls)


def _skill_source_label(item: SkillConfigItem) -> str:
    return f"{item.source} skill"


def _project_skill_dir() -> Path:
    return Path.cwd() / ".agents" / "skills"


def _project_skill_link_paths() -> tuple[Path, ...]:
    skills_dir = _project_skill_dir()
    if not skills_dir.exists():
        return ()
    return tuple(child for child in sorted(skills_dir.iterdir()) if child.is_symlink())


def _skill_candidate_controls(
    skill_items: tuple[SkillConfigItem, ...],
) -> tuple[Static | Horizontal, ...]:
    if not skill_items:
        return (Static("No new skills found", id="skill-candidates-empty"),)
    rows: list[Horizontal] = []
    for index, item in enumerate(skill_items):
        rows.append(
            Horizontal(
                Static(
                    f"{item.name} - {item.location}",
                    classes="skill-candidate-value",
                    id=f"skill-candidate-{index}",
                ),
                Button(
                    "Load",
                    id=f"load-skill-{index}",
                    classes="load-skill",
                ),
                classes="skill-candidate-row",
                id=f"skill-candidate-row-{index}",
            )
        )
    return tuple(rows)


def _find_project_skill_dirs() -> tuple[Path, ...]:
    root = Path.cwd()
    found = tuple(
        sorted(
            skill_file.parent
            for skill_file in root.rglob("SKILL.md")
            if not skill_file.is_relative_to(_project_skill_dir())
        )
    )
    return found


def _skill_items_from_dirs(skill_dirs: tuple[Path, ...]) -> tuple[SkillConfigItem, ...]:
    registry = SkillRegistry()
    for skill_dir in skill_dirs:
        registry.register(Skill(skill_dir))
    return _skill_config_items(registry)


def _skill_item_from_dir(skill_dir: Path) -> SkillConfigItem:
    return _skill_config_items(SkillRegistry(Skill(skill_dir)))[0]


def _selected_skill_names(
    screen: ModalScreen[object],
    skill_items: tuple[SkillConfigItem, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for index, item in enumerate(skill_items):
        try:
            checked = screen.query_one(f"#skill-{index}", Checkbox).value
        except NoMatches:
            checked = False
        if checked:
            selected.append(item.name)
    return tuple(selected)


class ProviderSetupScreen(ModalScreen[AgentAppConfig]):
    """Collect provider settings before the first live agent run."""

    DEFAULT_CSS = """
    ProviderSetupScreen {
        align: center middle;
    }

    #setup-panel {
        width: 72;
        height: auto;
        border: round #88c0d0;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #setup-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #setup-error {
        color: #bf616a;
        height: 1;
    }

    #setup-divider {
        height: 1;
        margin: 1 0;
        border-top: solid #4c566a;
    }

    #setup-skills-list {
        height: auto;
    }

    .skill-entry {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .skill-description {
        color: #e5e9f0;
        margin-left: 3;
        height: auto;
    }

    .skill-source {
        color: #88c0d0;
        margin-left: 3;
        height: auto;
    }

    .skill-location {
        color: #a7b1c2;
        margin-left: 3;
        height: auto;
    }

    Input, Checkbox {
        background: #3b4252;
        color: #eceff4;
        border: round #88c0d0;
    }

    #setup-actions {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(self, *, default_model: OpenAIModel) -> None:
        super().__init__()
        self._default_model = default_model
        self._provider = "openai"
        self._skill_items = _skill_config_items(
            SkillLoader.load_registry(
                ACE_AGENT_SKILL_PATH,
                extra_skill_paths=ACE_AGENT_BUILTIN_SKILL_PATHS,
            )
        )
        self._provider_highlight = _highlight_for_value(
            supported_provider_names(),
            self._provider,
        )
        self._model_highlight = _highlight_for_value(
            supported_models(self._provider),
            self._default_model,
        )

    def compose(self) -> ComposeResult:
        with Container(id="setup-panel"):
            yield Label("AceAI configuration", id="setup-title")
            yield Label(_field_label("provider"))
            yield Input(
                value="openai",
                placeholder="Provider",
                id="provider",
            )
            yield Static(
                _candidate_text(
                    _matching_candidates(supported_provider_names(), "openai"),
                    self._provider_highlight,
                ),
                id="provider-options",
            )
            yield Label(_field_label("model"))
            yield Input(
                value=self._default_model,
                placeholder="Model",
                id="model",
            )
            yield Static(
                _candidate_text(
                    _matching_candidates(
                        supported_models("openai"), self._default_model
                    ),
                    self._model_highlight,
                ),
                id="model-options",
            )
            yield Label(_field_label("api_key"))
            yield Input(
                password=True,
                placeholder=api_key_env("openai"),
                id="api-key",
            )
            yield Static("", id="setup-divider")
            yield Label(_skills_field_label())
            with Container(id="setup-skills-list"):
                yield from _skill_checkboxes(self._skill_items, self._skill_items)
            yield Checkbox("Persist to .aceai/config.yml", id="persist")
            yield Static("", id="setup-error")
            with Horizontal(id="setup-actions"):
                yield Button("Start", variant="primary", id="start")
                yield Button("Quit", id="quit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
            return
        if event.button.id != "start":
            return
        api_key = self.query_one("#api-key", Input).value
        if api_key == "":
            self.query_one("#setup-error", Static).update("API key is required")
            return
        provider = self._selected_provider()
        model = self.query_one("#model", Input).value
        if model not in supported_models(provider):
            raise ValueError("Unsupported model")
        enabled_skills = self._selected_skill_names()
        config = AgentAppConfig(
            provider=provider,
            api_key=api_key,
            model=cast(OpenAIModel, model),
            default_model=cast(OpenAIModel, model),
            skills=ACE_AGENT_SKILL_PATH,
            skill_selection_mode="selected",
            enabled_skills=list(enabled_skills),
            api_keys={provider: api_key},
            compress_threshold="100%",
        )
        persist = self.query_one("#persist", Checkbox).value
        if persist:
            save_config(config)
        self.dismiss(config)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "provider":
            self._provider_highlight = 0
            self._refresh_provider_candidates()
            if (
                event.value in supported_provider_names()
                and event.value != self._provider_name
            ):
                self._select_provider(event.value)
            return
        if event.input.id == "model":
            self._model_highlight = 0
            self._refresh_model_candidates()

    def on_key(self, event: Key) -> None:
        provider_input = self.query_one("#provider", Input)
        model_input = self.query_one("#model", Input)
        if provider_input.has_focus and event.key in ("up", "down", "tab"):
            candidates = self._provider_candidates()
            if not candidates:
                return
            event.prevent_default()
            event.stop()
            self._handle_candidate_key(
                event.key,
                input_widget=provider_input,
                candidates=candidates,
                field="provider",
            )
            return
        if model_input.has_focus and event.key in ("up", "down", "tab"):
            candidates = self._model_candidates()
            if not candidates:
                return
            event.prevent_default()
            event.stop()
            self._handle_candidate_key(
                event.key,
                input_widget=model_input,
                candidates=candidates,
                field="model",
            )

    def _handle_candidate_key(
        self,
        key: str,
        *,
        input_widget: Input,
        candidates: tuple[str, ...],
        field: str,
    ) -> None:
        if field == "provider":
            if key == "up":
                self._provider_highlight = _move_highlight(
                    candidates, self._provider_highlight, -1
                )
                self._refresh_provider_candidates()
                return
            if key == "down":
                self._provider_highlight = _move_highlight(
                    candidates, self._provider_highlight, 1
                )
                self._refresh_provider_candidates()
                return
            input_widget.value = candidates[self._provider_highlight]
            self._refresh_provider_candidates()
            return
        if key == "up":
            self._model_highlight = _move_highlight(
                candidates, self._model_highlight, -1
            )
            self._refresh_model_candidates()
            return
        if key == "down":
            self._model_highlight = _move_highlight(
                candidates, self._model_highlight, 1
            )
            self._refresh_model_candidates()
            return
        input_widget.value = candidates[self._model_highlight]
        self._refresh_model_candidates()

    def _select_provider(self, provider: str) -> None:
        self._provider = provider
        model_input = self.query_one("#model", Input)
        model_input.value = default_model(provider)
        self._model_highlight = _highlight_for_value(
            supported_models(provider),
            model_input.value,
        )
        self._refresh_model_candidates()
        self.query_one("#api-key", Input).placeholder = api_key_env(self._provider)

    def _provider_candidates(self) -> tuple[str, ...]:
        return _matching_candidates(
            supported_provider_names(),
            self.query_one("#provider", Input).value,
        )

    def _model_candidates(self) -> tuple[str, ...]:
        return _matching_candidates(
            supported_models(self._provider),
            self.query_one("#model", Input).value,
        )

    def _refresh_provider_candidates(self) -> None:
        self.query_one("#provider-options", Static).update(
            _candidate_text(self._provider_candidates(), self._provider_highlight)
        )

    def _refresh_model_candidates(self) -> None:
        self.query_one("#model-options", Static).update(
            _candidate_text(self._model_candidates(), self._model_highlight)
        )

    def _selected_provider(self) -> str:
        value = self.query_one("#provider", Input).value
        if value not in supported_provider_names():
            raise ValueError("Unsupported provider")
        return value

    def _selected_skill_names(self) -> tuple[str, ...]:
        return _selected_skill_names(self, self._skill_items)


class ConfigScreen(Screen[ConfigSelection | None]):
    """Collect runtime app configuration changes for future TUI runs."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    DEFAULT_CSS = """
    ConfigScreen {
        layout: vertical;
        background: #2e3440;
        color: #e5e9f0;
    }

    #config-panel {
        width: 100%;
        height: 1fr;
        padding: 1 3;
        background: #2e3440;
        color: #e5e9f0;
    }

    #config-tabs {
        height: 1fr;
    }

    #config-scroll, #tool-permissions-scroll, #stats-scroll {
        width: 100%;
        height: 1fr;
    }

    #stats-body {
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
    }

    #config-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #reasoning-level-row {
        height: auto;
    }

    #reasoning-level-row.hidden {
        display: none;
    }

    Input, Checkbox {
        background: #3b4252;
        color: #eceff4;
        border: round #88c0d0;
    }

    #config-error {
        color: #bf616a;
        height: 1;
    }

    #skill-search-error {
        color: #bf616a;
        height: auto;
        min-height: 1;
        margin-bottom: 1;
    }

    .config-divider {
        height: 1;
        margin: 1 0;
        border-top: solid #4c566a;
    }

    #skill-search-actions {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        align: center middle;
    }

    #search-skills {
        width: 20;
    }

    #skill-candidates {
        height: auto;
        margin-bottom: 1;
    }

    .skill-candidate-row {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        align: center middle;
    }

    .skill-candidate-value {
        width: 1fr;
        height: 3;
        content-align: left middle;
        color: #e5e9f0;
    }

    .load-skill {
        width: 10;
        margin-left: 1;
    }

    #config-skills-list {
        height: auto;
    }

    #tool-permissions-list {
        height: auto;
    }

    .tool-permission-table {
        width: 100%;
        height: auto;
        background: #343b49;
        border: round #4c566a;
    }

    .tool-permission-row {
        width: 100%;
        height: auto;
        min-height: 3;
        margin-bottom: 1;
        align: center middle;
    }

    .tool-permission-header {
        height: 2;
        min-height: 2;
        margin-bottom: 0;
        background: #3b4252;
        color: #a7b1c2;
        text-style: bold;
    }

    .tool-permission-cell {
        height: 100%;
        padding: 0 1;
        content-align: left middle;
    }

    .tool-enabled-cell {
        width: 7;
        padding: 0 0;
        content-align: center middle;
    }

    .tool-enabled-toggle {
        width: 5;
        height: 3;
        background: transparent;
        border: none;
        padding: 0 1;
        color: #a3be8c;
    }

    .tool-permission-name {
        width: 24;
        text-style: bold;
        color: #eceff4;
        content-align: left middle;
    }

    .tool-permission-policy-cell {
        width: 20;
        content-align: left middle;
    }

    .tool-permission-select {
        width: 19;
    }

    .tool-max-calls-cell {
        width: 18;
        content-align: left middle;
    }

    .tool-max-calls-input {
        width: 17;
    }

    .tool-permission-description {
        width: 1fr;
        height: auto;
        color: #a7b1c2;
        content-align: left top;
    }

    .tool-disabled {
        color: #6f7888;
        text-style: dim;
    }

    #tool-permission-actions {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        align: center middle;
    }

    #tool-permission-help {
        width: 1fr;
        color: #a7b1c2;
    }

    .skill-entry {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .skill-description {
        color: #e5e9f0;
        margin-left: 3;
        height: auto;
    }

    .skill-source {
        color: #88c0d0;
        margin-left: 3;
        height: auto;
    }

    .skill-location {
        color: #a7b1c2;
        margin-left: 3;
        height: auto;
    }

    #config-actions {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        *,
        provider_name: str,
        current_model: OpenAIModel,
        default_model: OpenAIModel,
        skills: str,
        api_keys: dict[str, str],
        skill_items: tuple[SkillConfigItem, ...] = (),
        skill_selection_mode: str = "all",
        enabled_skills: tuple[str, ...] = (),
        tool_permission_items: tuple[ToolPermissionItem, ...] = (),
        compress_threshold: CompressThreshold = "100%",
        reasoning_level: ReasoningLevel = "auto",
        stats_sections: list[MetadataSection] | None = None,
        initial_tab: str = "settings-tab",
    ) -> None:
        super().__init__()
        self._provider_name = provider_name
        self._current_model = current_model
        self._default_model = default_model
        self._skills = ACE_AGENT_SKILL_PATH if skills != "disable" else skills
        self._project_skill_links = _project_skill_link_paths()
        self._found_skill_dirs: tuple[Path, ...] = ()
        self._candidate_skill_dirs: tuple[Path, ...] = ()
        self._candidate_skill_items: tuple[SkillConfigItem, ...] = ()
        self._skill_items = skill_items
        self._skill_selection_mode = skill_selection_mode
        self._enabled_skills = enabled_skills
        self._tool_order = tuple(item.name for item in tool_permission_items)
        self._tool_descriptions = {
            item.name: item.description for item in tool_permission_items
        }
        self._tool_enabled = {item.name: item.enabled for item in tool_permission_items}
        self._tool_permissions = {
            item.name: item.permission for item in tool_permission_items
        }
        self._tool_max_calls = {
            item.name: item.max_calls_per_run
            for item in tool_permission_items
            if item.max_calls_per_run is not None
        }
        self._compress_threshold = compress_threshold
        self._reasoning_level = reasoning_level
        self._stats_sections = list(stats_sections or [])
        self._initial_tab = initial_tab
        self._sync_tool_order()
        self._api_keys = api_keys
        self._provider_highlight = _highlight_for_value(
            supported_provider_names(),
            self._provider_name,
        )
        self._model_highlight = _highlight_for_value(
            supported_models(self._provider_name),
            self._current_model,
        )

    def compose(self) -> ComposeResult:
        with Container(id="config-panel"):
            yield Label("AceAI configuration", id="config-title")
            with TabbedContent(initial=self._initial_tab, id="config-tabs"):
                with TabPane("Settings", id="settings-tab"):
                    with VerticalScroll(id="config-scroll"):
                        yield Label(_field_label("provider"))
                        yield Input(
                            value=self._provider_name,
                            placeholder="Provider",
                            id="provider",
                        )
                        yield Static(
                            _candidate_text(
                                _matching_candidates(
                                    supported_provider_names(), self._provider_name
                                ),
                                self._provider_highlight,
                            ),
                            id="provider-options",
                        )
                        yield Label(_field_label("model"))
                        yield Input(
                            value=self._current_model,
                            placeholder="Model",
                            id="model",
                        )
                        yield Static(
                            _candidate_text(
                                _matching_candidates(
                                    supported_models(self._provider_name),
                                    self._current_model,
                                ),
                                self._model_highlight,
                            ),
                            id="model-options",
                        )
                        with Container(
                            id="reasoning-level-row",
                            classes=self._reasoning_level_row_classes(),
                        ):
                            yield Label(_field_label("reasoning_level"))
                            yield Select(
                                _reasoning_level_options_for(
                                    self._provider_name,
                                    self._current_model,
                                ),
                                value=self._reasoning_level,
                                allow_blank=False,
                                id="reasoning-level",
                            )
                        yield Label(_field_label("api_key"))
                        yield Input(
                            value=_masked_api_key(
                                self._api_key_for_provider(self._provider_name)
                            ),
                            placeholder=api_key_env(self._provider_name),
                            id="api-key",
                        )
                        yield Static(
                            "",
                            classes="config-divider",
                            id="config-compression-divider",
                        )
                        yield Label(_field_label("compress_threshold"))
                        yield Input(
                            value=_compress_threshold_input_value(
                                self._compress_threshold
                            ),
                            placeholder="80%",
                            id="compress-threshold",
                        )
                        yield Static("", id="config-error")
                with TabPane("Tools", id="tool-permissions-tab"):
                    with VerticalScroll(id="tool-permissions-scroll"):
                        yield Label(_skills_field_label())
                        with Horizontal(id="skill-search-actions"):
                            yield Button(
                                "Search for skills",
                                id="search-skills",
                            )
                        yield Static("", id="skill-search-error")
                        yield Label("new skills")
                        with Container(id="skill-candidates"):
                            yield from _skill_candidate_controls(
                                self._candidate_skill_items
                            )
                        with Container(id="config-skills-list"):
                            yield from _skill_checkboxes(
                                self._skill_items,
                                self._checked_skill_items(),
                            )
                        yield Static(
                            "",
                            classes="config-divider",
                            id="config-tools-divider",
                        )
                        with Container(id="tool-permissions-list"):
                            yield from self._tool_permission_controls()
                with TabPane("Stats", id="stats-tab"):
                    with VerticalScroll(id="stats-scroll"):
                        stats_body = RichLog(
                            id="stats-body",
                            wrap=True,
                            auto_scroll=False,
                        )
                        for renderable in _metadata_renderables(self._stats_sections):
                            stats_body.write(renderable)
                        yield stats_body
            with Horizontal(id="config-actions"):
                yield Button("Apply", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "provider":
            self._provider_highlight = 0
            self._refresh_provider_candidates()
            if (
                event.value in supported_provider_names()
                and event.value != self._provider_name
            ):
                self._select_provider(event.value)
            return
        if event.input.id == "model":
            self._model_highlight = 0
            self._refresh_model_candidates()

    def on_key(self, event: Key) -> None:
        provider_input = self.query_one("#provider", Input)
        model_input = self.query_one("#model", Input)
        if provider_input.has_focus and event.key in ("up", "down", "tab"):
            candidates = self._provider_candidates()
            if not candidates:
                return
            event.prevent_default()
            event.stop()
            self._handle_candidate_key(
                event.key,
                input_widget=provider_input,
                candidates=candidates,
                field="provider",
            )
            return
        if model_input.has_focus and event.key in ("up", "down", "tab"):
            candidates = self._model_candidates()
            if not candidates:
                return
            event.prevent_default()
            event.stop()
            self._handle_candidate_key(
                event.key,
                input_widget=model_input,
                candidates=candidates,
                field="model",
            )

    def _handle_candidate_key(
        self,
        key: str,
        *,
        input_widget: Input,
        candidates: tuple[str, ...],
        field: str,
    ) -> None:
        if field == "provider":
            if key == "up":
                self._provider_highlight = _move_highlight(
                    candidates, self._provider_highlight, -1
                )
                self._refresh_provider_candidates()
                return
            if key == "down":
                self._provider_highlight = _move_highlight(
                    candidates, self._provider_highlight, 1
                )
                self._refresh_provider_candidates()
                return
            input_widget.value = candidates[self._provider_highlight]
            self._refresh_provider_candidates()
            return
        if key == "up":
            self._model_highlight = _move_highlight(
                candidates, self._model_highlight, -1
            )
            self._refresh_model_candidates()
            return
        if key == "down":
            self._model_highlight = _move_highlight(
                candidates, self._model_highlight, 1
            )
            self._refresh_model_candidates()
            return
        input_widget.value = candidates[self._model_highlight]
        self._refresh_model_candidates()

    def _select_provider(self, provider: str) -> None:
        self._provider_name = provider
        model_input = self.query_one("#model", Input)
        model_input.value = default_model(self._provider_name)
        self._model_highlight = _highlight_for_value(
            supported_models(self._provider_name),
            model_input.value,
        )
        self._refresh_model_candidates()
        self.query_one("#api-key", Input).placeholder = api_key_env(self._provider_name)
        self.query_one("#api-key", Input).value = _masked_api_key(
            self._api_key_for_provider(self._provider_name)
        )
        self._sync_reasoning_level_visibility()

    def _provider_candidates(self) -> tuple[str, ...]:
        return _matching_candidates(
            supported_provider_names(),
            self.query_one("#provider", Input).value,
        )

    def _model_candidates(self) -> tuple[str, ...]:
        return _matching_candidates(
            supported_models(self._provider_name),
            self.query_one("#model", Input).value,
        )

    def _refresh_provider_candidates(self) -> None:
        self.query_one("#provider-options", Static).update(
            _candidate_text(self._provider_candidates(), self._provider_highlight)
        )

    def _refresh_model_candidates(self) -> None:
        self.query_one("#model-options", Static).update(
            _candidate_text(self._model_candidates(), self._model_highlight)
        )
        model = self.query_one("#model", Input).value
        if model in supported_models(self._provider_name):
            self.query_one("#api-key", Input).value = _masked_api_key(
                self._api_key_for_provider(self._provider_name)
            )
        self._sync_reasoning_level_visibility()

    def _api_key_for_provider(self, provider: str) -> str:
        if provider in self._api_keys:
            return self._api_keys[provider]
        env_name = api_key_env(provider)
        if env_name in os.environ:
            return os.environ[env_name]
        return ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "allow-all-tools":
            self._set_all_tool_permissions("always")
            return
        if event.button.id == "disable-all-tools":
            self._set_all_tools_enabled(False)
            return
        if event.button.id == "enable-all-tools":
            self._set_all_tools_enabled(True)
            return
        if event.button.id == "search-skills":
            self._search_project_skills()
            return
        if event.button.id is not None and event.button.id.startswith("load-skill-"):
            index = int(event.button.id.removeprefix("load-skill-"))
            self._load_candidate_skill(index)
            return
        if event.button.id != "apply":
            return
        provider = self.query_one("#provider", Input).value
        model = self.query_one("#model", Input).value
        reasoning_level = self._selected_reasoning_level(provider, model)
        self._skills = ACE_AGENT_SKILL_PATH
        enabled_skills = self._selected_skill_names()
        stored_api_key = (
            self._api_key_for_provider(provider)
            if provider in supported_provider_names()
            else ""
        )
        api_key = _api_key_value_from_input(
            self.query_one("#api-key", Input).value,
            stored_api_key,
        )
        error = _config_selection_error(
            provider,
            model,
            api_key,
            self._skills,
            reasoning_level,
        )
        if error is not None:
            self.query_one("#config-error", Static).update(error)
            return
        try:
            self._sync_tool_settings_from_controls()
            compress_threshold = _compress_threshold_from_input(
                self.query_one("#compress-threshold", Input).value,
            )
        except ValueError as exc:
            self.query_one("#config-error", Static).update(str(exc))
            return
        self.dismiss(
            ConfigSelection(
                provider=provider,
                model=cast(OpenAIModel, model),
                default_model=cast(OpenAIModel, model),
                api_key=api_key,
                skills=self._skills,
                skill_selection_mode="selected",
                enabled_skills=enabled_skills,
                tool_permissions=dict(self._tool_permissions),
                tool_enabled=dict(self._tool_enabled),
                tool_max_calls=dict(self._tool_max_calls),
                compress_threshold=compress_threshold,
                reasoning_level=reasoning_level,
            )
        )

    def _selected_reasoning_level(
        self,
        provider: str,
        model: str,
    ) -> ReasoningLevel:
        if provider not in supported_provider_names():
            return "auto"
        if model not in supported_models(provider):
            return "auto"
        if not supports_reasoning_effort(provider, model):
            return "auto"
        reasoning_level = self.query_one("#reasoning-level", Select).value
        options = reasoning_effort_options(provider, model)
        if reasoning_level not in ("auto", *options):
            raise ValueError("Unsupported reasoning level")
        return reasoning_level

    def _sync_reasoning_level_visibility(self) -> None:
        row = self.query_one("#reasoning-level-row", Container)
        provider = self._provider_name
        model = self.query_one("#model", Input).value
        if supports_reasoning_effort(provider, model):
            select = self.query_one("#reasoning-level", Select)
            current_value = select.value
            select.set_options(_reasoning_level_options_for(provider, model))
            options = reasoning_effort_options(provider, model)
            if current_value in ("auto", *options):
                select.value = current_value
            elif self._reasoning_level in options:
                select.value = self._reasoning_level
            else:
                select.value = "auto"
            row.remove_class("hidden")
            return
        row.add_class("hidden")
        self.query_one("#reasoning-level", Select).value = "auto"

    def _reasoning_level_row_classes(self) -> str:
        if supports_reasoning_effort(self._provider_name, self._current_model):
            return ""
        return "hidden"

    def _checked_skill_items(self) -> tuple[SkillConfigItem, ...]:
        if self._skill_selection_mode == "all":
            return self._skill_items
        selected_names = set(self._enabled_skills)
        return tuple(
            item
            for item in self._skill_items
            if item.builtin or item.name in selected_names
        )

    def _selected_skill_names(self) -> tuple[str, ...]:
        return _selected_skill_names(self, self._skill_items)

    def _search_project_skills(self) -> None:
        try:
            self._found_skill_dirs = _find_project_skill_dirs()
            self._set_candidate_skills(self._found_skill_dirs)
        except (SkillLoadingError, OSError) as exc:
            self._set_skill_search_error(f"Skill search failed: {exc}")
            return
        self._set_skill_search_error("")
        self.run_worker(
            self._refresh_skill_candidate_controls(),
            group="skill-candidate-refresh",
            exclusive=True,
        )

    def _set_candidate_skills(self, skill_dirs: tuple[Path, ...]) -> None:
        loaded_names = {item.name for item in self._skill_items}
        loaded_names.update(path.name for path in self._project_skill_links)
        candidate_dirs: list[Path] = []
        candidate_items: list[SkillConfigItem] = []
        seen_names: set[str] = set()
        for skill_dir in skill_dirs:
            item = _skill_item_from_dir(skill_dir)
            if item.name in loaded_names or item.name in seen_names:
                continue
            seen_names.add(item.name)
            candidate_dirs.append(skill_dir)
            candidate_items.append(item)
        self._candidate_skill_dirs = tuple(candidate_dirs)
        self._candidate_skill_items = tuple(candidate_items)

    def _load_candidate_skill(self, index: int) -> None:
        try:
            self._save_skill_link(self._candidate_skill_dirs[index])
        except (IndexError, SkillLoadingError, OSError) as exc:
            self._set_skill_search_error(f"Skill search failed: {exc}")
            return
        self._project_skill_links = _project_skill_link_paths()
        self._try_reload_skill_items(ACE_AGENT_SKILL_PATH)
        self._skill_selection_mode = "all"
        self._enabled_skills = tuple(item.name for item in self._skill_items)
        self._set_candidate_skills(self._candidate_skill_dirs)
        self.run_worker(
            self._refresh_skill_candidate_controls(),
            group="skill-candidate-refresh",
            exclusive=True,
        )
        self.run_worker(
            self._refresh_skill_controls(),
            group="skill-refresh",
            exclusive=True,
        )

    def _save_found_skill_links(self) -> None:
        project_skill_dir = _project_skill_dir()
        project_skill_dir.mkdir(parents=True, exist_ok=True)
        for skill_dir in self._found_skill_dirs:
            self._save_skill_link(skill_dir)

    def _save_skill_link(self, skill_dir: Path) -> None:
        project_skill_dir = _project_skill_dir()
        project_skill_dir.mkdir(parents=True, exist_ok=True)
        skill = Skill(skill_dir)
        link_path = project_skill_dir / skill.name
        target = skill_dir.resolve()
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink() or link_path.resolve() != target:
                raise SkillLoadingError(
                    f"Project skill link {link_path} already exists"
                )
            return
        link_path.symlink_to(target, target_is_directory=True)

    def _try_reload_skill_items(self, skills: str) -> bool:
        try:
            self._reload_skill_items(skills)
        except (SkillLoadingError, OSError) as exc:
            self._set_skill_search_error(f"Skill search failed: {exc}")
            return False
        self._set_skill_search_error("")
        return True

    def _set_skill_search_error(self, message: str) -> None:
        self.query_one("#skill-search-error", Static).update(message)

    def _reload_skill_items(self, skills: str) -> None:
        registry = SkillLoader.load_registry(
            skills,
            extra_skill_paths=ACE_AGENT_BUILTIN_SKILL_PATHS,
        )
        checked_names = set(self._selected_skill_names())
        checked_names.update(item.name for item in self._checked_skill_items())
        self._skill_items = _skill_config_items(registry)
        self._skill_selection_mode = "selected"
        self._enabled_skills = tuple(
            item.name
            for item in self._skill_items
            if item.builtin or item.name in checked_names
        )

    async def _refresh_skill_candidate_controls(self) -> None:
        candidates = self.query_one("#skill-candidates", Container)
        await candidates.remove_children()
        await candidates.mount(*_skill_candidate_controls(self._candidate_skill_items))

    async def _refresh_skill_controls(self) -> None:
        skills_list = self.query_one("#config-skills-list", Container)
        await skills_list.remove_children()
        await skills_list.mount(
            *_skill_checkboxes(self._skill_items, self._checked_skill_items())
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id is None:
            return
        if event.select.id == "reasoning-level":
            if event.value not in ("auto", "low", "medium", "high", "max"):
                raise ValueError("Unsupported reasoning level")
            self._reasoning_level = cast(ReasoningLevel, event.value)
            return
        if not event.select.id.startswith("tool-permission-"):
            return
        if event.value not in TOOL_PERMISSION_OPTIONS:
            raise ValueError("Unsupported tool permission")
        tool_name = self._tool_control_names[event.select.id]
        self._tool_permissions[tool_name] = event.value

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id is None:
            return
        if not event.checkbox.id.startswith("tool-enabled-"):
            return
        self._sync_tool_settings_from_controls()
        tool_name = self._tool_enabled_control_names[event.checkbox.id]
        self._tool_enabled[tool_name] = event.value
        self._sync_tool_order()
        self.run_worker(
            self._refresh_tool_permission_controls(),
            group="tool-permission-refresh",
            exclusive=True,
        )

    def _tool_permission_controls(self) -> ComposeResult:
        with Horizontal(id="tool-permission-actions"):
            yield Static(
                "Tool access and per-run limits. Empty max calls means unlimited.",
                id="tool-permission-help",
            )
            yield Button("Enable all", id="enable-all-tools")
            yield Button("Allow all", id="allow-all-tools")
            yield Button("Disable all", id="disable-all-tools")
        with Container(id="tool-permission-table", classes="tool-permission-table"):
            yield from self._tool_permission_table_widgets()

    def _set_all_tool_permissions(self, permission: ToolPermission) -> None:
        for index, tool_name in enumerate(self._tool_names):
            self._tool_permissions[tool_name] = permission
            self.query_one(f"#tool-permission-{index}", Select).value = permission

    def _set_all_tools_enabled(self, enabled: bool) -> None:
        for tool_name in self._tool_names:
            self._tool_enabled[tool_name] = enabled
        self._sync_tool_order()
        self.run_worker(
            self._refresh_tool_permission_controls(),
            group="tool-permission-refresh",
            exclusive=True,
        )

    def _sync_tool_settings_from_controls(self) -> None:
        for index, tool_name in enumerate(self._tool_names):
            self._tool_enabled[tool_name] = self.query_one(
                f"#tool-enabled-{index}", Checkbox
            ).value
            value = self.query_one(f"#tool-permission-{index}", Select).value
            if value not in TOOL_PERMISSION_OPTIONS:
                raise ValueError("Unsupported tool permission")
            self._tool_permissions[tool_name] = value
            max_calls_value = self.query_one(f"#tool-max-calls-{index}", Input).value
            if max_calls_value == "":
                if tool_name in self._tool_max_calls:
                    del self._tool_max_calls[tool_name]
                continue
            if not max_calls_value.isdecimal():
                raise ValueError("Max calls must be empty or a positive integer")
            max_calls = int(max_calls_value)
            if max_calls < 1:
                raise ValueError("Max calls must be empty or a positive integer")
            self._tool_max_calls[tool_name] = max_calls

    def _tool_max_calls_value(self, tool_name: str) -> str:
        if tool_name not in self._tool_max_calls:
            return ""
        return f"{self._tool_max_calls[tool_name]}"

    def _sync_tool_order(self) -> None:
        self._tool_names = tuple(
            sorted(
                self._tool_order,
                key=lambda tool_name: not self._tool_enabled[tool_name],
            )
        )
        self._tool_control_names = {
            f"tool-permission-{index}": tool_name
            for index, tool_name in enumerate(self._tool_names)
        }
        self._tool_enabled_control_names = {
            f"tool-enabled-{index}": tool_name
            for index, tool_name in enumerate(self._tool_names)
        }

    async def _refresh_tool_permission_controls(self) -> None:
        table = self.query_one("#tool-permission-table", Container)
        await table.remove_children()
        await table.mount(*self._tool_permission_table_widgets())

    def _tool_permission_table_widgets(self) -> tuple[Horizontal, ...]:
        return (
            Horizontal(
                Static("On", classes="tool-permission-cell tool-enabled-cell"),
                Static("Tool", classes="tool-permission-cell tool-permission-name"),
                Static(
                    "Permission",
                    classes="tool-permission-cell tool-permission-policy-cell",
                ),
                Static(
                    "Max calls",
                    classes="tool-permission-cell tool-max-calls-cell",
                ),
                Static(
                    "Description",
                    classes="tool-permission-cell tool-permission-description",
                ),
                classes="tool-permission-row tool-permission-header",
            ),
            *self._tool_permission_rows(),
        )

    def _tool_permission_rows(self) -> tuple[Horizontal, ...]:
        rows: list[Horizontal] = []
        for index, tool_name in enumerate(self._tool_names):
            disabled_class = "" if self._tool_enabled[tool_name] else " tool-disabled"
            rows.append(
                Horizontal(
                    Checkbox(
                        "",
                        value=self._tool_enabled[tool_name],
                        id=f"tool-enabled-{index}",
                        classes=(
                            "tool-permission-cell "
                            "tool-enabled-cell "
                            f"tool-enabled-toggle{disabled_class}"
                        ),
                    ),
                    Static(
                        tool_name,
                        classes=f"tool-permission-cell tool-permission-name{disabled_class}",
                    ),
                    Select(
                        tuple((option, option) for option in TOOL_PERMISSION_OPTIONS),
                        value=self._tool_permissions[tool_name],
                        allow_blank=False,
                        id=f"tool-permission-{index}",
                        classes=(
                            "tool-permission-cell "
                            "tool-permission-policy-cell "
                            f"tool-permission-select{disabled_class}"
                        ),
                    ),
                    Input(
                        value=self._tool_max_calls_value(tool_name),
                        placeholder="no limit",
                        id=f"tool-max-calls-{index}",
                        classes=(
                            "tool-permission-cell "
                            "tool-max-calls-cell "
                            f"tool-max-calls-input{disabled_class}"
                        ),
                    ),
                    Static(
                        self._tool_descriptions[tool_name],
                        classes=(
                            "tool-permission-cell "
                            f"tool-permission-description{disabled_class}"
                        ),
                        id=f"tool-permission-description-{index}",
                    ),
                    classes=f"tool-permission-row{disabled_class}",
                    id=f"tool-row-{index}",
                )
            )
        return tuple(rows)


def _config_selection_error(
    provider: str,
    model: str,
    api_key: str,
    skills: str,
    reasoning_level: ReasoningLevel,
) -> str | None:
    if provider == "":
        return "Provider is required"
    if model == "":
        return "Model is required"
    if api_key == "":
        return "API key is required"
    if skills == "":
        return "Skills is required"
    if provider not in supported_provider_names():
        return "Unsupported provider"
    if model not in supported_models(provider):
        return "Unsupported model"
    if reasoning_level != "auto" and reasoning_level not in reasoning_effort_options(
        provider, model
    ):
        return "Reasoning level is unsupported for this model"
    return None


def _reasoning_level_options_for(
    provider: str,
    model: str,
) -> tuple[tuple[str, ReasoningLevel], ...]:
    options: list[tuple[str, ReasoningLevel]] = [("auto", "auto")]
    for option in reasoning_effort_options(provider, model):
        if option not in REASONING_LEVEL_LABELS:
            raise ValueError("Unsupported reasoning level")
        reasoning_level = cast(ReasoningLevel, option)
        options.append((REASONING_LEVEL_LABELS[reasoning_level], reasoning_level))
    return tuple(options)


def _field_label(name: str) -> str:
    for config_field in config_schema().fields:
        if config_field.name == name:
            marker = " *" if config_field.required else ""
            return f"{config_field.name}{marker}"
    raise ValueError("Unknown config field")


def _skills_field_label() -> str:
    for config_field in config_schema().fields:
        if config_field.name == "skills":
            marker = " *" if config_field.required else ""
            return f"skills for current model{marker}"
    raise ValueError("Unknown config field")


class SessionSelectScreen(ModalScreen[str | None]):
    """Select a saved session to resume in the TUI."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", priority=True),
        Binding("k", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down", priority=True),
        Binding("j", "cursor_down", "Down"),
        Binding("enter", "resume_session", "Resume"),
        Binding("d", "confirm_delete_session", "Delete", priority=True),
    ]

    DEFAULT_CSS = """
    SessionSelectScreen {
        align: center middle;
    }

    #session-panel {
        width: 92%;
        height: 88%;
        border: round #81a1c1;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #session-title {
        height: 1;
        color: #8fbcbb;
        text-style: bold;
    }

    #session-list-scroll {
        height: 1fr;
        margin-top: 1;
    }

    #session-status {
        height: 1;
        margin-top: 1;
        color: #9aa3b2;
    }
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        sessions: list[SessionMetadata],
        current_session_id: str | None,
    ) -> None:
        super().__init__()
        self._store = store
        self._sessions = sessions
        self._current_session_id = current_session_id

    def compose(self) -> ComposeResult:
        session_ids = {session.session_id for session in self._sessions}
        selected_session_id = (
            self._current_session_id
            if self._current_session_id in session_ids
            else self._sessions[0].session_id
        )
        with Container(id="session-panel"):
            yield Label(f"Sessions  {len(self._sessions)}", id="session-title")
            with VerticalScroll(id="session-list-scroll"):
                yield SessionListWidget(
                    sessions=self._sessions,
                    current_session_id=self._current_session_id,
                    selected_session_id=selected_session_id,
                    id="session-list",
                )
            yield Static(
                f"Enter resumes the highlighted session. Press d to delete. "
                f"Total cost: {format_usd(self._store.total_cost_usd())}.",
                id="session-status",
            )

    def on_mount(self) -> None:
        self.query_one("#session-list", SessionListWidget).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        self.query_one("#session-list", SessionListWidget).move_selection(-1)

    def action_cursor_down(self) -> None:
        self.query_one("#session-list", SessionListWidget).move_selection(1)

    def action_resume_session(self) -> None:
        session_id = self._selected_session_id()
        if session_id is None:
            self.query_one("#session-status", Static).update("Select a session row.")
            return
        self.dismiss(session_id)

    def action_confirm_delete_session(self) -> None:
        session_id = self._selected_session_id()
        if session_id is None:
            self.query_one("#session-status", Static).update("Select a session row.")
            return
        if session_id == self._current_session_id:
            self.query_one("#session-status", Static).update(
                "Switch to another session before deleting the current one."
            )
            return
        session = _session_by_id(self._sessions, session_id)
        self.app.push_screen(
            DeleteSessionConfirmScreen(session=session),
            lambda confirmed: self._delete_session_after_confirmation(
                confirmed,
                session_id,
            ),
        )

    def _delete_session_after_confirmation(
        self,
        confirmed: bool | None,
        session_id: str,
    ) -> None:
        if confirmed is not True:
            self.query_one("#session-list", SessionListWidget).focus()
            return
        self._store.delete_session(session_id)
        session_list = self.query_one("#session-list", SessionListWidget)
        selected_index = session_list.selected_index
        self._sessions = [
            session for session in self._sessions if session.session_id != session_id
        ]
        session_list.set_sessions(self._sessions, selected_index=selected_index)
        session_list.focus()
        self._update_title()
        self.query_one("#session-status", Static).update("Session deleted.")

    def _selected_session_id(self) -> str | None:
        return self.query_one("#session-list", SessionListWidget).selected_session_id()

    def _update_title(self) -> None:
        self.query_one("#session-title", Label).update(
            f"Sessions  {len(self._sessions)}"
        )


class SelectablePanelListWidget(Static, Generic[PanelListItem]):
    """Reusable panel-rendered list with one highlighted selection."""

    can_focus = True

    def __init__(
        self,
        *,
        items: list[PanelListItem],
        render_items: PanelListRenderer[PanelListItem],
        empty_message: str,
        selected_index: int = 0,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._items = items
        self._render_items = render_items
        self._empty_message = empty_message
        self.selected_index = self._clamp_index(selected_index)

    def on_mount(self) -> None:
        self._refresh_renderable()

    def set_items(
        self,
        items: list[PanelListItem],
        *,
        selected_index: int,
    ) -> None:
        self._items = items
        self.selected_index = self._clamp_index(selected_index)
        self._refresh_renderable()

    def move_selection(self, delta: int) -> None:
        if not self._items:
            return
        self.selected_index = self._clamp_index(self.selected_index + delta)
        self._refresh_renderable()

    def selected_item(self) -> PanelListItem | None:
        if not self._items:
            return None
        return self._items[self.selected_index]

    def items(self) -> list[PanelListItem]:
        return self._items

    def _clamp_index(self, index: int) -> int:
        if not self._items:
            return 0
        return max(0, min(index, len(self._items) - 1))

    def _refresh_renderable(self) -> None:
        if not self._items:
            self.update(
                Panel(
                    Text(self._empty_message, style="#d8dee9"),
                    box=box.ROUNDED,
                    border_style="#4c566a",
                    padding=(0, 1),
                )
            )
            return
        self.update(Group(*self._render_items(self._items, self.selected_index)))


class SessionListWidget(SelectablePanelListWidget[SessionMetadata]):
    """Panel-rendered session list."""

    def __init__(
        self,
        *,
        sessions: list[SessionMetadata],
        current_session_id: str | None,
        selected_session_id: str,
        id: str | None = None,
    ) -> None:
        self._current_session_id = current_session_id
        super().__init__(
            items=sessions,
            render_items=self._render_sessions,
            empty_message="No saved sessions yet.",
            selected_index=_session_index(sessions, selected_session_id),
            id=id,
        )

    def set_sessions(
        self, sessions: list[SessionMetadata], *, selected_index: int
    ) -> None:
        self.set_items(sessions, selected_index=selected_index)

    def selected_session_id(self) -> str | None:
        session = self.selected_item()
        if session is None:
            return None
        return session.session_id

    def sessions(self) -> list[SessionMetadata]:
        return self.items()

    def _render_sessions(
        self,
        sessions: list[SessionMetadata],
        selected_index: int,
    ) -> list[Text | Panel]:
        return _session_picker_renderables(
            sessions,
            current_session_id=self._current_session_id,
            selected_index=selected_index,
        )


class IdeaListWidget(SelectablePanelListWidget[Idea]):
    """Panel-rendered idea list."""

    def __init__(self, ideas: list[Idea], *, id: str | None = None) -> None:
        super().__init__(
            items=ideas,
            render_items=_idea_picker_renderables,
            empty_message="No saved ideas yet.",
            id=id,
        )

    def set_ideas(self, ideas: list[Idea], *, selected_index: int) -> None:
        self.set_items(ideas, selected_index=selected_index)

    def ideas(self) -> list[Idea]:
        return self.items()

    def selected_idea(self) -> Idea | None:
        return self.selected_item()


class IdeaPickerScreen(ModalScreen[Idea | None]):
    """Pick, reference, or edit saved ideas."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", priority=True),
        Binding("k", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down", priority=True),
        Binding("j", "cursor_down", "Down"),
        Binding("enter", "reference_idea", "Reference"),
        Binding("a", "add_idea", "Add"),
        Binding("e", "edit_idea", "Edit"),
        Binding("d", "delete_idea", "Delete"),
    ]

    DEFAULT_CSS = """
    IdeaPickerScreen {
        align: center middle;
    }

    #idea-panel {
        width: 92%;
        height: 88%;
        background: #2e3440;
        border: round #81a1c1;
        padding: 1 2;
    }

    #idea-title {
        height: 1;
        color: #8fbcbb;
        text-style: bold;
    }

    #idea-list-scroll {
        height: 1fr;
        margin-top: 1;
    }

    #idea-add-panel {
        height: 4;
        margin-top: 1;
    }

    #idea-add-panel.hidden {
        display: none;
    }

    #idea-add-input {
        height: 3;
        border: round #88c0d0;
        background: #3b4252;
        color: #eceff4;
    }

    #idea-status {
        height: 1;
        margin-top: 1;
        color: #9aa3b2;
    }
    """

    def __init__(
        self,
        *,
        ideas: list[Idea],
        capture_idea: IdeaCaptureHandler,
        save_idea: IdeaSaveHandler,
        delete_idea: IdeaDeleteHandler,
    ) -> None:
        super().__init__()
        self._ideas = ideas
        self._capture_idea = capture_idea
        self._save_idea = save_idea
        self._delete_idea = delete_idea

    def compose(self) -> ComposeResult:
        with Container(id="idea-panel"):
            yield Label(f"Ideas  {len(self._ideas)}", id="idea-title")
            with VerticalScroll(id="idea-list-scroll"):
                yield IdeaListWidget(self._ideas, id="idea-list")
            with Container(id="idea-add-panel", classes="hidden"):
                yield Input(
                    value="",
                    id="idea-add-input",
                )
            yield Static(
                "Enter references the highlighted idea. Press a to add, e to edit, d to delete.",
                id="idea-status",
            )

    def on_mount(self) -> None:
        self.query_one("#idea-list", IdeaListWidget).focus()

    def action_cancel(self) -> None:
        if self._add_panel_visible():
            self._hide_add_panel()
            return
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        self.query_one("#idea-list", IdeaListWidget).move_selection(-1)

    def action_cursor_down(self) -> None:
        self.query_one("#idea-list", IdeaListWidget).move_selection(1)

    def action_reference_idea(self) -> None:
        if not self._ideas:
            self.query_one("#idea-status", Static).update("No idea selected.")
            return
        self.dismiss(self._selected_idea())

    def action_edit_idea(self) -> None:
        if not self._ideas:
            self.query_one("#idea-status", Static).update("No idea selected.")
            return
        idea = self._selected_idea()
        self.app.push_screen(
            IdeaEditScreen(index=self._selected_index(), content=idea.content),
            self._after_edit,
        )

    def action_add_idea(self) -> None:
        self._show_add_panel()

    def action_save_new_idea(self) -> None:
        if not self._add_panel_visible():
            return
        content = self.query_one("#idea-add-input", Input).value
        self._ideas = self._capture_idea(content)
        idea_list = self.query_one("#idea-list", IdeaListWidget)
        idea_list.set_ideas(self._ideas, selected_index=len(self._ideas) - 1)
        self._update_title()
        self._hide_add_panel(status="Idea added.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "idea-add-input":
            return
        event.stop()
        self.action_save_new_idea()

    def action_delete_idea(self) -> None:
        if not self._ideas:
            self.query_one("#idea-status", Static).update("No idea selected.")
            return
        idea_list = self.query_one("#idea-list", IdeaListWidget)
        selected_row = idea_list.selected_index
        index = self._selected_index()
        self._ideas = self._delete_idea(index)
        next_row = min(selected_row, len(self._ideas) - 1)
        idea_list.set_ideas(self._ideas, selected_index=max(0, next_row))
        idea_list.focus()
        self._update_title()
        self.query_one("#idea-status", Static).update("Idea deleted.")

    def _after_edit(self, edited: tuple[int, str] | None) -> None:
        if edited is None:
            self.query_one("#idea-list", IdeaListWidget).focus()
            return
        index, content = edited
        self._ideas = self._save_idea(index, content)
        idea_list = self.query_one("#idea-list", IdeaListWidget)
        selected_row = min(index - 1, len(self._ideas) - 1)
        idea_list.set_ideas(self._ideas, selected_index=selected_row)
        idea_list.focus()
        self.query_one("#idea-status", Static).update("Idea updated.")

    def _show_add_panel(self) -> None:
        panel = self.query_one("#idea-add-panel", Container)
        panel.remove_class("hidden")
        add_input = self.query_one("#idea-add-input", Input)
        add_input.value = ""
        add_input.focus()
        self.query_one("#idea-status", Static).update("Enter saves. Escape cancels.")

    def _hide_add_panel(self, *, status: str | None = None) -> None:
        panel = self.query_one("#idea-add-panel", Container)
        panel.add_class("hidden")
        self.query_one("#idea-add-input", Input).value = ""
        self.query_one("#idea-list", IdeaListWidget).focus()
        if status is not None:
            self.query_one("#idea-status", Static).update(status)
            return
        self.query_one("#idea-status", Static).update(
            "Enter references the highlighted idea. Press a to add, e to edit, d to delete."
        )

    def _add_panel_visible(self) -> bool:
        return not self.query_one("#idea-add-panel", Container).has_class("hidden")

    def _update_title(self) -> None:
        self.query_one("#idea-title", Label).update(f"Ideas  {len(self._ideas)}")

    def _selected_index(self) -> int:
        return self.query_one("#idea-list", IdeaListWidget).selected_index + 1

    def _selected_idea(self) -> Idea:
        return self._ideas[self._selected_index() - 1]


class IdeaEditScreen(ModalScreen[tuple[int, str] | None]):
    """Edit a saved idea."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    DEFAULT_CSS = """
    IdeaEditScreen {
        align: center middle;
    }

    #idea-edit-panel {
        width: 82%;
        height: 64%;
        background: #2e3440;
        border: round #81a1c1;
        padding: 1 2;
    }

    #idea-edit-title {
        height: 1;
        color: #8fbcbb;
        text-style: bold;
    }

    #idea-editor {
        height: 1fr;
        margin-top: 1;
        border: round #88c0d0;
        background: #3b4252;
        color: #eceff4;
    }

    #idea-edit-actions {
        height: 3;
        margin-top: 1;
    }
    """

    def __init__(self, *, index: int, content: str) -> None:
        super().__init__()
        self._index = index
        self._content = content

    def compose(self) -> ComposeResult:
        with Container(id="idea-edit-panel"):
            yield Label(f"Edit idea {self._index}", id="idea-edit-title")
            yield TextArea(
                self._content,
                id="idea-editor",
                show_line_numbers=False,
                soft_wrap=True,
            )
            with Horizontal(id="idea-edit-actions"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#idea-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.action_cancel()
            return
        if event.button.id == "save":
            self.action_save()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        self.dismiss((self._index, self.query_one("#idea-editor", TextArea).text))


def _idea_title(idea: Idea) -> str:
    for line in idea.content.splitlines():
        if line != "":
            return line
    return ""


def _idea_body_lines(idea: Idea) -> list[str]:
    lines = idea.content.splitlines()
    body_lines: list[str] = []
    found_title = False
    for line in lines:
        if not found_title and line != "":
            found_title = True
            continue
        if found_title:
            body_lines.append(line)
    return body_lines


def _idea_picker_renderables(
    ideas: list[Idea],
    selected_index: int,
) -> list[Panel]:
    return [
        _idea_panel(
            idea,
            index=index,
            selected=index == selected_index,
        )
        for index, idea in enumerate(ideas)
    ]


def _idea_panel(idea: Idea, *, index: int, selected: bool) -> Panel:
    created_at = idea.created_at.strftime("%Y-%m-%d %H:%M")
    title = Text()
    marker = "> " if selected else "  "
    title.append(marker, style="#88c0d0" if selected else "#4c566a")
    title.append(f"{index + 1:>2}. ", style="bold #9aa3b2")
    title.append(_fixed_width(_idea_title(idea), width=48), style="bold #eceff4")
    title.append("  ")
    title.append(f"{idea.project_name}  ", style="#8fbcbb")
    title.append(created_at, style="#9aa3b2")
    return Panel(
        _idea_body_text(idea, expanded=selected),
        box=box.ROUNDED,
        title=title,
        title_align="left",
        border_style="#88c0d0" if selected else "#4c566a",
        style="on #3b4252" if selected else "",
        padding=(0, 1),
    )


def _idea_body_text(idea: Idea, *, expanded: bool) -> Text:
    body_lines = _idea_body_lines(idea)
    if expanded:
        display_lines = _at_least_preview_height(body_lines)
        body = "\n".join(display_lines)
        return Text(body if body != "" else " ", style="#d8dee9")
    preview = _idea_preview_lines(body_lines)
    text = Text()
    for index, line in enumerate(preview):
        if index > 0:
            text.append("\n")
        text.append(line, style="#d8dee9")
    return text


def _at_least_preview_height(lines: list[str]) -> list[str]:
    display_lines = [line if line != "" else " " for line in lines]
    while len(display_lines) < IDEA_PREVIEW_LINES:
        display_lines.append(" ")
    return display_lines


def _idea_preview_lines(body_lines: list[str]) -> list[str]:
    body = "\n".join(body_lines)
    if body == "":
        return [" ", " "]
    chunks = _wrap_preview_text(body, width=IDEA_PREVIEW_WIDTH)
    truncated = len(chunks) > IDEA_PREVIEW_LINES
    preview = chunks[:IDEA_PREVIEW_LINES]
    while len(preview) < IDEA_PREVIEW_LINES:
        preview.append(" ")
    if truncated:
        preview[-1] = _preview_more_line(preview[-1])
    return preview


def _wrap_preview_text(value: str, *, width: int) -> list[str]:
    wrapped: list[str] = []
    for source_line in value.splitlines():
        if source_line == "":
            wrapped.append(" ")
            continue
        line = source_line
        while line != "":
            if cell_len(line) <= width:
                wrapped.append(line)
                break
            wrapped.append(set_cell_size(line, width).rstrip())
            line = line[width:]
    return wrapped


def _preview_more_line(value: str) -> str:
    suffix = " ... more"
    return f"{set_cell_size(value, IDEA_PREVIEW_WIDTH - len(suffix)).rstrip()}{suffix}"


def _fixed_width(value: str, *, width: int) -> str:
    return set_cell_size(value, width)


def _delete_session_actions() -> Text:
    text = Text()
    text.append("Enter ", style="#9aa3b2")
    text.append("Delete", style="bold #bf616a")
    text.append("   ")
    text.append("Esc ", style="#9aa3b2")
    text.append("Cancel", style="#d8dee9")
    return text


class DeleteSessionConfirmScreen(ModalScreen[bool]):
    """Confirm deletion of a saved session."""

    BINDINGS = [
        Binding("enter", "confirm", "Delete"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    DeleteSessionConfirmScreen {
        align: center middle;
    }

    #delete-session-panel {
        width: 64;
        height: auto;
        border: round #bf616a;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #delete-session-title {
        text-style: bold;
        margin-bottom: 1;
        color: #bf616a;
    }

    #delete-session-message {
        margin-bottom: 1;
    }

    #delete-session-actions {
        height: 1;
        margin-top: 1;
        color: #9aa3b2;
    }
    """

    def __init__(self, *, session: SessionMetadata) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        with Container(id="delete-session-panel"):
            yield Label("Delete session?", id="delete-session-title")
            yield Static(
                f"{session_display_title(self._session.title)}\n"
                f"{self._session.session_id}",
                id="delete-session-message",
            )
            yield Static(_delete_session_actions(), id="delete-session-actions")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)


def _session_index(
    sessions: list[SessionMetadata],
    selected_session_id: str,
) -> int:
    for index, session in enumerate(sessions):
        if session.session_id == selected_session_id:
            return index
    return 0


def _session_picker_renderables(
    sessions: list[SessionMetadata],
    *,
    current_session_id: str | None,
    selected_index: int,
) -> list[Text | Panel]:
    renderables: list[Text | Panel] = []
    global_index = 0
    for _project_id, project_name, group_sessions in _session_groups(sessions):
        header = Text()
        header.append(project_name, style="bold #8fbcbb")
        header.append(f"  {len(group_sessions)}", style="#9aa3b2")
        renderables.append(header)
        for session in group_sessions:
            renderables.append(
                _session_panel(
                    session,
                    index=global_index,
                    selected=global_index == selected_index,
                    current=session.session_id == current_session_id,
                )
            )
            global_index += 1
    return renderables


def _session_panel(
    session: SessionMetadata,
    *,
    index: int,
    selected: bool,
    current: bool,
) -> Panel:
    title = Text()
    marker = "> " if selected else "  "
    title.append(marker, style="#88c0d0" if selected else "#4c566a")
    title.append(f"{index + 1:>2}. ", style="bold #9aa3b2")
    title.append(
        _fixed_width(session_display_title(session.title), width=58),
        style="bold #eceff4",
    )
    title.append("  ")
    title.append(session.project_name, style="#8fbcbb")
    title.append("  ")
    title.append(_session_second(session.updated_at), style="#9aa3b2")
    if current:
        title.append("  current", style="#88c0d0")
    body = Text()
    body.append("created ", style="#9aa3b2")
    body.append(_session_second(session.created_at), style="#d8dee9")
    body.append("  ")
    body.append(session.session_id, style="#9aa3b2")
    return Panel(
        body,
        box=box.ROUNDED,
        title=title,
        title_align="left",
        border_style="#88c0d0" if selected else "#4c566a",
        style="on #3b4252" if selected else "",
        padding=(0, 1),
    )


def _session_groups(
    sessions: list[SessionMetadata],
) -> list[tuple[str, str, list[SessionMetadata]]]:
    groups: list[tuple[str, str, list[SessionMetadata]]] = []
    project_indexes: dict[str, int] = {}
    for session in sessions:
        index = project_indexes.get(session.project_id)
        if index is None:
            project_indexes[session.project_id] = len(groups)
            groups.append((session.project_id, session.project_name, [session]))
            continue
        groups[index][2].append(session)
    return groups


def _session_by_id(
    sessions: list[SessionMetadata],
    session_id: str,
) -> SessionMetadata:
    for session in sessions:
        if session.session_id == session_id:
            return session
    raise ValueError(session_id)


def _session_second(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")
