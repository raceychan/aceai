from aceai.agent.memory.context_checkpoint_store import ContextCheckpoint
from aceai.agent.session import EventLog, SessionEvent
from aceai.llm.models import LLMMessage


def build_context_history(
    *,
    event_log: EventLog,
    checkpoint: ContextCheckpoint | None,
) -> list[LLMMessage]:
    if checkpoint is None:
        return event_log.replay_llm_history()

    later_events = _events_after_checkpoint(
        event_log.events,
        included_event_id=checkpoint.included_event_id,
    )
    later_history = EventLog(later_events).replay_llm_history()
    return list(checkpoint.history) + later_history


def _events_after_checkpoint(
    events: list[SessionEvent],
    *,
    included_event_id: str,
) -> list[SessionEvent]:
    for index, event in enumerate(events):
        if event.event_id == included_event_id:
            return events[index + 1 :]
    raise ValueError("context checkpoint included_event_id is not in session events")
