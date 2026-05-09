"""Runtime metadata screen for the AceAI TUI."""

from msgspec import Struct, field
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, RichLog


class MetadataSection(Struct, frozen=True, kw_only=True):
    title: str
    lines: list[str] = field(default_factory=list[str])


class MetadataScreen(ModalScreen[None]):
    """Show runtime configuration, model, usage, skills, and tools."""

    DEFAULT_CSS = """
    MetadataScreen {
        align: center middle;
    }

    #metadata-panel {
        width: 100;
        height: 34;
        max-height: 34;
        border: round #88c0d0;
        padding: 1 2;
        background: #2e3440;
        color: #e5e9f0;
    }

    #metadata-body {
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
    }

    #metadata-actions {
        height: 3;
        margin-top: 1;
    }

    Button {
        width: auto;
        min-width: 10;
    }
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, sections: list[MetadataSection]) -> None:
        super().__init__()
        self._sections = sections

    def compose(self) -> ComposeResult:
        with Container(id="metadata-panel"):
            body = RichLog(id="metadata-body", wrap=True, auto_scroll=False)
            for renderable in _metadata_renderables(self._sections):
                body.write(renderable)
            yield body
            with Container(id="metadata-actions"):
                yield Button("Close", id="metadata-close")

    def on_mount(self) -> None:
        self.query_one("#metadata-body", RichLog).focus()

    def on_key(self, event: Key) -> None:
        body = self.query_one("#metadata-body", RichLog)
        if event.key == "up":
            body.scroll_up(animate=False)
            event.stop()
            return
        if event.key == "down":
            body.scroll_down(animate=False)
            event.stop()
            return
        if event.key == "pageup":
            body.scroll_page_up(animate=False)
            event.stop()
            return
        if event.key == "pagedown":
            body.scroll_page_down(animate=False)
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "metadata-close":
            self.dismiss(None)

    def action_dismiss(self) -> None:
        self.dismiss(None)


def _metadata_renderables(sections: list[MetadataSection]) -> list[RenderableType]:
    renderables: list[RenderableType] = []
    for section_index, section in enumerate(sections):
        if section_index > 0:
            renderables.append(Text(""))
        renderables.append(Text(section.title, style="bold #88c0d0"))
        if not section.lines:
            renderables.append(Text("  -", style="#d8dee9"))
            continue
        renderables.append(_section_table(section))
    return renderables


def _section_table(section: MetadataSection) -> Table:
    if section.title.endswith("Tool Calls"):
        return _tool_call_table(section)
    if section.title.startswith(("Skills", "Tools", "Hosted Tools")):
        return _list_table(section)
    return _key_value_table(section)


def _key_value_table(section: MetadataSection) -> Table:
    table = Table.grid(expand=True)
    table.add_column(width=18, style="#9aa3b2")
    table.add_column(ratio=1, style="#eceff4")
    for line in section.lines:
        key, value = _split_key_value(line)
        table.add_row(key, value)
    return table


def _list_table(section: MetadataSection) -> Table:
    table = Table.grid(expand=True)
    table.add_column(width=24, style="bold #eceff4")
    table.add_column(ratio=1, style="#d8dee9")
    table.add_column(ratio=1, style="dim #9aa3b2")
    for line in section.lines:
        name, detail, location = _split_list_line(line)
        table.add_row(name, detail, location)
    return table


def _tool_call_table(section: MetadataSection) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=1, min_width=24, style="bold #eceff4", overflow="fold")
    table.add_column(width=8, justify="right", style="#d8dee9")
    table.add_column(width=8, justify="right", style="#a3be8c")
    table.add_column(width=8, justify="right", style="#bf616a")
    table.add_row(
        Text("tool", style="#9aa3b2"),
        Text("calls", style="#9aa3b2"),
        Text("ok", style="#9aa3b2"),
        Text("failed", style="#9aa3b2"),
    )
    for line in section.lines:
        name, calls, succeeded, failed = _split_tool_call_stat(line)
        table.add_row(name, calls, succeeded, failed)
    return table


def _split_key_value(line: str) -> tuple[str, str]:
    if ": " not in line:
        return "", line
    key, value = line.split(": ", 1)
    return f"{key}:", value


def _split_list_line(line: str) -> tuple[str, str, str]:
    location = ""
    detail = line
    if line.endswith(")") and " (" in line:
        detail, location = line.rsplit(" (", 1)
        location = location[:-1]
    if ": " not in detail:
        return detail, "", location
    name, description = detail.split(": ", 1)
    return name, description, location


def _split_tool_call_stat(line: str) -> tuple[str, str, str, str]:
    name, values = line.rsplit(": ", 1)
    parts = values.split()
    if len(parts) != 6:
        raise ValueError("tool call stat line has unsupported format")
    if parts[0] != "calls" or parts[2] != "ok" or parts[4] != "failed":
        raise ValueError("tool call stat line has unsupported format")
    return name, parts[1], parts[3], parts[5]
