"""Top navigation strip for the AceAI TUI."""

from datetime import datetime

from textual.containers import Horizontal
from textual.events import Click
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static

DEBUG_ICON = "\uf188"
CONFIG_ICON = "\uf013"


class TopBarWidget(Horizontal):
    """Compact top bar with window actions and session navigation."""

    class QuitRequested(Message):
        pass

    class DebugRequested(Message):
        pass

    class ConfigRequested(Message):
        pass

    DEFAULT_CSS = """
    TopBarWidget {
        height: 1;
        background: #3b4252;
        color: #eceff4;
        layout: horizontal;
    }

    #topbar-quit {
        width: 3;
        height: 1;
        content-align: center middle;
        background: #bf616a;
        color: #eceff4;
        text-style: bold;
    }

    #topbar-title {
        width: 1fr;
        height: 1;
        content-align: center middle;
        background: #3b4252;
        color: #eceff4;
        text-style: bold;
    }

    #topbar-config, #topbar-debug {
        width: 3;
        height: 1;
        content-align: center middle;
        color: #eceff4;
        text-style: bold;
    }

    #topbar-config {
        background: #5e81ac;
    }

    #topbar-debug {
        background: #d08770;
    }

    #topbar-time {
        width: 10;
        height: 1;
        content-align: center middle;
        background: #3b4252;
        color: #eceff4;
        text-style: bold;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._title = "AceAI"
        self._timer: Timer | None = None

    def compose(self):
        yield Static("‹", id="topbar-quit")
        yield Static("", id="topbar-title")
        yield Static(CONFIG_ICON, id="topbar-config")
        yield Static(DEBUG_ICON, id="topbar-debug")
        yield Static("", id="topbar-time")

    def on_mount(self) -> None:
        self._timer = self.set_interval(1.0, self._render_topbar)
        self._render_topbar()

    def set_title(self, title: str) -> None:
        self._title = title
        self._render_topbar()

    def on_click(self, event: Click) -> None:
        control = event.control
        if control is None:
            return
        if control.id == "topbar-quit":
            event.stop()
            self.post_message(self.QuitRequested())
            return
        if control.id == "topbar-config":
            event.stop()
            self.post_message(self.ConfigRequested())
            return
        if control.id == "topbar-debug":
            event.stop()
            self.post_message(self.DebugRequested())

    def _render_topbar(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#topbar-title", Static).update(self._title)
        self.query_one("#topbar-time", Static).update(datetime.now().strftime("%H:%M:%S"))

    def on_unmount(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None:
            timer.stop()
