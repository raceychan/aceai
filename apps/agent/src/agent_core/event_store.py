import json
from pathlib import Path
from typing import Protocol

from agent_core.session import EventLog, SessionEvent, SessionMetadata


class EventStore(Protocol):
    def create_event_log(self, session_id: str) -> str: ...

    def append_event(self, metadata: SessionMetadata, event: SessionEvent) -> None: ...

    def load_event_log(self, metadata: SessionMetadata) -> EventLog: ...

    def delete_event_log(self, metadata: SessionMetadata) -> None: ...


class JsonlEventStore:
    """JSONL-backed durable event storage for agent sessions."""

    def __init__(self, files_dir: Path) -> None:
        self.files_dir = files_dir
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def create_event_log(self, session_id: str) -> str:
        path = self.files_dir / f"{session_id}.events.jsonl"
        path.write_text("", encoding="utf-8")
        return path.name

    def append_event(self, metadata: SessionMetadata, event: SessionEvent) -> None:
        persisted_event = event.with_session_defaults(session_id=metadata.session_id)
        with Path(metadata.path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(persisted_event.as_json(), ensure_ascii=False))
            stream.write("\n")

    def load_event_log(self, metadata: SessionMetadata) -> EventLog:
        events: list[SessionEvent] = []
        for line in Path(metadata.path).read_text(encoding="utf-8").splitlines():
            if line == "":
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("Session event must be a mapping")
            events.append(SessionEvent.from_json(payload))
        return EventLog(events)

    def delete_event_log(self, metadata: SessionMetadata) -> None:
        Path(metadata.path).unlink()
