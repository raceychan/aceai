"""Provider setup screen for the AceAI TUI."""

from datetime import datetime
from typing import cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select, Static

from aceai.agent.session import SessionMetadata, SessionStore
from aceai.llm.openai import OpenAIModel

from .config import AceAITUIConfig, OPENAI_MODEL_OPTIONS, SUPPORTED_OPENAI_MODELS
from .config import save_config
from .session_display import session_display_title


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

    Input, Select, Checkbox {
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

    def compose(self) -> ComposeResult:
        with Container(id="setup-panel"):
            yield Label("AceAI provider setup", id="setup-title")
            yield Label("Provider")
            yield Select(
                [("OpenAI", "openai")],
                value="openai",
                allow_blank=False,
                id="provider",
            )
            yield Label("Model")
            yield Select(
                OPENAI_MODEL_OPTIONS,
                value=self._default_model,
                allow_blank=False,
                id="model",
            )
            yield Label("API key")
            yield Input(password=True, placeholder="OPENAI_API_KEY", id="api-key")
            yield Checkbox("Persist to ~/.aceai/config.yaml", id="persist")
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
        provider = self.query_one("#provider", Select).value
        model = self.query_one("#model", Select).value
        if provider != "openai":
            raise ValueError("Unsupported provider")
        if model not in SUPPORTED_OPENAI_MODELS:
            raise ValueError("Unsupported model")
        config = AceAITUIConfig(
            provider="openai",
            api_key=api_key,
            model=cast(OpenAIModel, model),
        )
        persist = self.query_one("#persist", Checkbox).value
        if persist:
            save_config(config)
        self.dismiss(config)


class ModelSelectScreen(ModalScreen[OpenAIModel]):
    """Select the runtime model for future TUI runs."""

    DEFAULT_CSS = """
    ModelSelectScreen {
        align: center middle;
    }

    #model-panel {
        width: 60;
        height: auto;
        border: solid #88c0d0;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #model-title {
        text-style: bold;
        margin-bottom: 1;
    }

    Select {
        background: #3b4252;
        color: #eceff4;
        border: solid #88c0d0;
    }

    #model-actions {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(self, *, current_model: OpenAIModel) -> None:
        super().__init__()
        self._current_model = current_model

    def compose(self) -> ComposeResult:
        with Container(id="model-panel"):
            yield Label("Model", id="model-title")
            yield Select(
                OPENAI_MODEL_OPTIONS,
                value=self._current_model,
                allow_blank=False,
                id="model",
            )
            with Horizontal(id="model-actions"):
                yield Button("Apply", variant="primary", id="apply")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id != "apply":
            return
        model = self.query_one("#model", Select).value
        if model not in SUPPORTED_OPENAI_MODELS:
            raise ValueError("Unsupported model")
        self.dismiss(cast(OpenAIModel, model))


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
            yield Static("Press d to delete the highlighted session.", id="session-status")
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
