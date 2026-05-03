"""Provider setup screen for the AceAI TUI."""

from typing import cast

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from aceai.llm.openai import OpenAIModel

from .config import AceAITUIConfig, save_config

OPENAI_MODEL_OPTIONS: list[tuple[str, OpenAIModel]] = [
    ("GPT-5.1", "gpt-5.1"),
    ("GPT-5o", "gpt-5o"),
    ("GPT-5o mini", "gpt-5o-mini"),
    ("GPT-4o", "gpt-4o"),
    ("GPT-4o mini", "gpt-4o-mini"),
    ("o3 large", "o3-large"),
    ("o4 mini", "o4-mini"),
]


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
        if model not in (
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5o",
            "gpt-5o-mini",
            "gpt-5.1",
            "o3-large",
            "o4-mini",
        ):
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
