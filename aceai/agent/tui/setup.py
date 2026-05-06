"""Provider setup screen for the AceAI TUI."""

from datetime import datetime
import os
from typing import cast

from rich.text import Text
from msgspec import field
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from aceai.agent.provider_catalog import (
    api_key_env,
    default_model,
    supported_models,
    supported_provider_names,
)
from aceai.agent.ace_agent import ACE_AGENT_SKILL_PATH
from aceai.agent.config import config_schema
from aceai.agent.permissions import TOOL_PERMISSION_OPTIONS, ToolPermission
from aceai.agent.session import SessionMetadata, SessionStore
from aceai.core.skills import SkillLoader, SkillRegistry
from aceai.llm.interface import Record
from aceai.llm.openai import OpenAIModel

from aceai.agent.config import AceAITUIConfig
from aceai.agent.config import save_config
from aceai.agent.cost import format_usd
from .session_display import session_display_title


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


class SkillConfigItem(Record, kw_only=True):
    name: str
    description: str
    location: str


class ToolPermissionItem(Record, kw_only=True):
    name: str
    description: str
    permission: ToolPermission
    enabled: bool = True
    max_calls_per_run: int | None = None


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


def _skill_config_items(registry: SkillRegistry) -> tuple[SkillConfigItem, ...]:
    return tuple(
        SkillConfigItem(
            name=skill.name,
            description=skill.description,
            location=str(skill.skill_file),
        )
        for skill in registry.get_skills()
    )


def _skill_checkboxes(
    skill_items: tuple[SkillConfigItem, ...],
    checked_items: tuple[SkillConfigItem, ...],
):
    if not skill_items:
        yield Static("No skills available", id="skills-empty")
        return
    checked_names = {item.name for item in checked_items}
    for index, item in enumerate(skill_items):
        with Container(classes="skill-entry"):
            yield Checkbox(
                item.name,
                value=item.name in checked_names,
                id=f"skill-{index}",
            )
            yield Static(
                item.description,
                classes="skill-description",
                id=f"skill-description-{index}",
            )
            yield Static(
                item.location,
                classes="skill-location",
                id=f"skill-location-{index}",
            )


def _selected_skill_names(
    screen: ModalScreen[object],
    skill_items: tuple[SkillConfigItem, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for index, item in enumerate(skill_items):
        if screen.query_one(f"#skill-{index}", Checkbox).value:
            selected.append(item.name)
    return tuple(selected)


class ProviderSetupScreen(ModalScreen[AceAITUIConfig]):
    """Collect provider settings before the first live agent run."""

    DEFAULT_CSS = """
    ProviderSetupScreen {
        align: center middle;
    }

    #setup-panel {
        width: 72;
        height: auto;
        border: solid #88c0d0;
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

    .skill-location {
        color: #a7b1c2;
        margin-left: 3;
        height: auto;
    }

    Input, Checkbox {
        background: #3b4252;
        color: #eceff4;
        border: solid #88c0d0;
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
            SkillLoader.load_registry(ACE_AGENT_SKILL_PATH)
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
                    _matching_candidates(supported_models("openai"), self._default_model),
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
        config = AceAITUIConfig(
            provider=provider,
            api_key=api_key,
            model=cast(OpenAIModel, model),
            default_model=cast(OpenAIModel, model),
            skills=ACE_AGENT_SKILL_PATH,
            skill_selection_mode="selected",
            enabled_skills=list(enabled_skills),
            api_keys={provider: api_key},
        )
        persist = self.query_one("#persist", Checkbox).value
        if persist:
            save_config(config)
        self.dismiss(config)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "provider":
            self._provider_highlight = 0
            self._refresh_provider_candidates()
            if event.value in supported_provider_names():
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

    #config-scroll, #tool-permissions-scroll {
        width: 100%;
        height: 1fr;
    }

    #config-title {
        text-style: bold;
        margin-bottom: 1;
    }

    Input, Checkbox {
        background: #3b4252;
        color: #eceff4;
        border: solid #88c0d0;
    }

    #config-error {
        color: #bf616a;
        height: 1;
    }

    .config-divider {
        height: 1;
        margin: 1 0;
        border-top: solid #4c566a;
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
        border: solid #4c566a;
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
    ) -> None:
        super().__init__()
        self._provider_name = provider_name
        self._current_model = current_model
        self._default_model = default_model
        self._skills = skills
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
            with TabbedContent(initial="settings-tab", id="config-tabs"):
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
                            id="config-skills-divider",
                        )
                        yield Label(_skills_field_label())
                        with Container(id="config-skills-list"):
                            yield from _skill_checkboxes(
                                self._skill_items,
                                self._checked_skill_items(),
                            )
                        yield Static("", id="config-error")
                with TabPane("Tools", id="tool-permissions-tab"):
                    with VerticalScroll(id="tool-permissions-scroll"):
                        with Container(id="tool-permissions-list"):
                            yield from self._tool_permission_controls()
            with Horizontal(id="config-actions"):
                yield Button("Apply", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "provider":
            self._provider_highlight = 0
            self._refresh_provider_candidates()
            if event.value in supported_provider_names():
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
        self.query_one("#api-key", Input).placeholder = (
            api_key_env(self._provider_name)
        )
        self.query_one("#api-key", Input).value = _masked_api_key(
            self._api_key_for_provider(self._provider_name)
        )

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
        if event.button.id != "apply":
            return
        provider = self.query_one("#provider", Input).value
        model = self.query_one("#model", Input).value
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
        )
        if error is not None:
            self.query_one("#config-error", Static).update(error)
            return
        try:
            self._sync_tool_settings_from_controls()
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
            )
        )

    def _checked_skill_items(self) -> tuple[SkillConfigItem, ...]:
        if self._skill_selection_mode == "all":
            return self._skill_items
        selected_names = set(self._enabled_skills)
        return tuple(item for item in self._skill_items if item.name in selected_names)

    def _selected_skill_names(self) -> tuple[str, ...]:
        return _selected_skill_names(self, self._skill_items)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id is None:
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
            sorted(self._tool_order, key=lambda tool_name: not self._tool_enabled[tool_name])
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
    return None


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


class SessionSelectScreen(ModalScreen[str]):
    """Select a saved session to resume in the TUI."""

    BINDINGS = [
        Binding("d", "confirm_delete_session", "Delete", priority=True),
    ]

    DEFAULT_CSS = """
    SessionSelectScreen {
        align: center middle;
    }

    #session-panel {
        width: 120;
        height: auto;
        border: solid #88c0d0;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #session-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #session-table {
        height: 14;
        background: #3b4252;
        color: #eceff4;
        border: solid #88c0d0;
    }

    #session-table > .datatable--header {
        text-style: bold;
        background: #2e3440;
        color: #88c0d0;
    }

    #session-table > .datatable--cursor {
        background: #007acc;
        color: #ffffff;
        text-style: bold;
    }

    #session-actions {
        height: auto;
        margin-top: 1;
    }

    #session-status {
        height: 1;
        margin-top: 1;
        color: #d8dee9;
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
        value = (
            self._current_session_id
            if self._current_session_id in session_ids
            else self._sessions[0].session_id
        )
        with Container(id="session-panel"):
            yield Label("Sessions", id="session-title")
            table = DataTable(id="session-table")
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.add_column("Current", width=7)
            table.add_column("Title", width=42)
            table.add_column("Updated", width=19)
            table.add_column("Created", width=19)
            table.add_column("Session ID", width=36)
            for session in self._sessions:
                table.add_row(
                    *_session_row_cells(
                        session,
                        current_session_id=self._current_session_id,
                    ),
                    key=session.session_id,
                )
            table.move_cursor(row=_session_row_index(self._sessions, value))
            yield table
            yield Static(
                f"Total cost: {format_usd(self._store.total_cost_usd())}. "
                "Press d to delete the highlighted session.",
                id="session-status",
            )
            with Horizontal(id="session-actions"):
                yield Button("Resume", variant="primary", id="resume")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#session-table", DataTable).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id != "resume":
            return
        session_id = self._selected_session_id()
        self.dismiss(session_id)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        session_id = event.row_key.value
        if type(session_id) is not str:
            raise TypeError("Selected session id must be str")
        self.dismiss(session_id)

    def action_confirm_delete_session(self) -> None:
        session_id = self._selected_session_id()
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
            self.query_one("#session-table", DataTable).focus()
            return
        self._store.delete_session(session_id)
        table = self.query_one("#session-table", DataTable)
        deleted_row = table.cursor_row
        table.remove_row(session_id)
        self._sessions = [
            session for session in self._sessions if session.session_id != session_id
        ]
        if table.row_count > 0:
            table.move_cursor(row=min(deleted_row, table.row_count - 1))
        table.focus()
        self.query_one("#session-status", Static).update("Session deleted.")

    def _selected_session_id(self) -> str:
        table = self.query_one("#session-table", DataTable)
        row = table.ordered_rows[table.cursor_row]
        session_id = row.key.value
        if type(session_id) is not str:
            raise TypeError("Selected session id must be str")
        return session_id


class DeleteSessionConfirmScreen(ModalScreen[bool]):
    """Confirm deletion of a saved session."""

    DEFAULT_CSS = """
    DeleteSessionConfirmScreen {
        align: center middle;
    }

    #delete-session-panel {
        width: 76;
        height: auto;
        border: solid #bf616a;
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
        height: auto;
        margin-top: 1;
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
            with Horizontal(id="delete-session-actions"):
                yield Button("Delete", variant="error", id="delete")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#delete", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        if event.button.id != "delete":
            return
        self.dismiss(True)


def _session_row_cells(
    session: SessionMetadata,
    *,
    current_session_id: str | None,
) -> tuple[str, str, str, str, str]:
    marker = "*" if session.session_id == current_session_id else ""
    return (
        marker,
        _fit_cell(session_display_title(session.title), 42),
        _session_second(session.updated_at),
        _session_second(session.created_at),
        session.session_id,
    )


def _session_row_index(sessions: list[SessionMetadata], session_id: str) -> int:
    for index, session in enumerate(sessions):
        if session.session_id == session_id:
            return index
    raise ValueError(session_id)


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


def _fit_cell(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return f"{value[: width - 3]}..."
