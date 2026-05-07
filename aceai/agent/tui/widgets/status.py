"""Status bar for the AceAI TUI."""

from time import perf_counter

from rich.table import Table
from rich.text import Text
from textual.events import Click
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static

from aceai.agent.cost import format_usd
from aceai.agent.provider_catalog import context_window_for_model_any_provider
from aceai.agent.tui.state import TUIRunStatus, TUIUsageState


class StatusBarWidget(Static):
    """Show persistent runtime state such as the selected model."""

    class MetadataRequested(Message):
        pass

    DEFAULT_CSS = """
    StatusBarWidget {
        height: 1;
        background: #3b4252;
        color: #eceff4;
        padding: 0 1;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("", id=id)
        self.current_text = ""
        self._model: str | None = None
        self._status: TUIRunStatus = "idle"
        self._usage = TUIUsageState()
        self._spinner_frame = 0
        self._spinner_timer: Timer | None = None
        self._notice_timer: Timer | None = None
        self._notice_queue: list[tuple[str, float, str]] = []
        self._notice_active = False
        self.current_style = ""
        self._left_text = ""
        self._right_text = ""
        self._run_started_at: float | None = None
        self._last_elapsed_seconds: float | None = None

    def set_status(
        self,
        *,
        model: str | None,
        status: TUIRunStatus,
        usage: TUIUsageState | None = None,
    ) -> None:
        self._model = model
        previous_status = self._status
        self._status = status
        self._usage = usage or TUIUsageState()
        self._sync_elapsed_timer(previous_status, status)
        self._sync_spinner_timer()
        if self._notice_active:
            return
        self._render_status()

    def _sync_elapsed_timer(
        self,
        previous_status: TUIRunStatus,
        next_status: TUIRunStatus,
    ) -> None:
        if next_status == "running":
            if previous_status != "running" or self._run_started_at is None:
                self._run_started_at = perf_counter()
                self._last_elapsed_seconds = None
            return
        if previous_status == "running" and self._run_started_at is not None:
            self._last_elapsed_seconds = perf_counter() - self._run_started_at
            self._run_started_at = None

    def show_notice(
        self,
        content: str,
        *,
        timeout: float = 3.0,
        style: str = "#a3be8c",
    ) -> None:
        self._notice_queue.append((content, timeout, style))
        if self._notice_active:
            return
        self._show_next_notice()

    def _show_next_notice(self) -> None:
        content, timeout, style = self._notice_queue.pop(0)
        self._notice_active = True
        self.current_text = content
        self.current_style = style
        self.update(Text(content, style=style))
        self._notice_timer = self.set_timer(timeout, self._clear_notice)

    def _clear_notice(self) -> None:
        self._notice_timer = None
        if self._notice_queue:
            self._show_next_notice()
            return
        self._notice_active = False
        self.current_style = ""
        self._render_status()

    def _sync_spinner_timer(self) -> None:
        if self._status == "running":
            if self._spinner_timer is None:
                self._spinner_timer = self.set_interval(0.15, self._tick_spinner)
            return
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        self._spinner_frame += 1
        if self._notice_active:
            return
        self._render_status()

    def _render_status(self) -> None:
        model_text = self._model if self._model is not None else "unconfigured"
        max_ctx = _max_context_window(self._model)
        left_text = f"{_status_symbol(self._status, self._spinner_frame)} model: {model_text}"
        right_text = _status_usage_text(
            self._usage,
            max_context_window=max_ctx,
            elapsed_seconds=self._elapsed_seconds(),
        )
        self.current_text = left_text if right_text == "" else f"{left_text}   {right_text}"
        if self._status == "suspended":
            self.current_text = (
                f"{self.current_text}   action: choose Approve or Reject"
            )
            right_text = f"{right_text}   action: choose Approve or Reject"

        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_column(justify="right")
        layout.add_row(Text(left_text), Text(right_text))
        self._left_text = left_text
        self._right_text = right_text
        self.update(layout)

    def on_click(self, event: Click) -> None:
        if self._notice_active:
            return
        width = max(1, self.size.width)
        if self._right_text != "" and event.x >= width - len(self._right_text) - 2:
            event.stop()
            self.post_message(self.MetadataRequested())

    def _elapsed_seconds(self) -> float | None:
        if self._run_started_at is not None:
            return perf_counter() - self._run_started_at
        return self._last_elapsed_seconds


def _usage_text(
    usage: TUIUsageState,
    *,
    max_context_window: int | None = None,
) -> str:
    context = _format_tokens(usage.current_context_tokens)
    ctx_pct = _context_window_pct(usage.current_context_tokens, max_context_window)
    cached = _format_tokens(usage.session_cached_input_tokens)
    session = _format_tokens(usage.session_total_tokens)
    input_tokens = _format_tokens(usage.session_input_tokens)
    output_tokens = _format_tokens(usage.session_output_tokens)
    cost = format_usd(usage.session_cost_usd)
    return (
        f"context: {context}{ctx_pct}   session: {session} "
        f"({input_tokens} in / {cached} cached / {output_tokens} out)   cost: {cost}"
    )


def _status_usage_text(
    usage: TUIUsageState,
    *,
    max_context_window: int | None = None,
    elapsed_seconds: float | None = None,
) -> str:
    parts: list[str] = []
    if usage.current_context_tokens is not None:
        ctx_pct = _context_window_pct(
            usage.current_context_tokens, max_context_window
        )
        parts.append(
            f"context: {_format_context_tokens(usage.current_context_tokens)}{ctx_pct}"
        )
    if usage.current_input_cache_hit_rate is not None:
        parts.append(f"cache rate: {_format_cache_rate(usage)}")
    if usage.session_cost_usd is not None and usage.session_cost_usd > 0:
        parts.append(f"cost: {_format_status_cost(usage.session_cost_usd)}")
    if elapsed_seconds is not None:
        parts.append(f"time: {_format_elapsed(elapsed_seconds)}")
    return "   ".join(parts)


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _format_context_tokens(value: int) -> str:
    return f"{value / 1000:.1f}k"


def _format_status_cost(value: float) -> str:
    return f"${value:.3f}"


def _format_elapsed(value: float) -> str:
    return f"{value:.1f}s"


def _format_cache_rate(usage: TUIUsageState) -> str:
    if usage.current_input_cache_hit_rate is None:
        return "-"
    return f"{usage.current_input_cache_hit_rate:.1%}"


def _max_context_window(model: str | None) -> int | None:
    if model is None:
        return None
    return context_window_for_model_any_provider(model)


def _context_window_pct(
    current_tokens: int | None,
    max_window: int | None,
) -> str:
    if current_tokens is None or max_window is None or max_window == 0:
        return ""
    pct = current_tokens / max_window * 100
    return f" ({pct:.0f}%)"


def _status_symbol(status: TUIRunStatus, frame: int) -> str:
    if status == "running":
        frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        return frames[frame % len(frames)]
    if status == "completed":
        return "✓"
    if status == "failed":
        return "!"
    if status == "suspended":
        return "?"
    return "·"
