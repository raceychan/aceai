# import asyncio
# from collections.abc import AsyncIterator
# from typing import Protocol

# from .events import AgentEvent


# class IEventBus(Protocol):
#     """Minimal event bus abstraction for agent step events."""

#     async def publish(self, event: AgentEvent) -> None:
#         """Broadcast a new event to all active subscribers."""

#     async def subscribe(self) -> AsyncIterator[AgentEvent]:
#         """Obtain an async iterator over future events."""
#         raise NotImplementedError

#     async def close(self) -> None:
#         """Release resources and signal subscribers to stop."""


# class InMemoryEventBus(IEventBus):
#     """Simple asyncio-backed bus that fans out events to subscribers."""

#     def __init__(self) -> None:
#         self._subscribers: set[asyncio.Queue[AgentEvent | None]] = set()
#         self._closed = False

#     async def publish(self, event: AgentEvent) -> None:
#         if self._closed or not self._subscribers:
#             return
#         queues = tuple(self._subscribers)
#         await asyncio.gather(*(queue.put(event) for queue in queues))

#     async def subscribe(self) -> AsyncIterator[AgentEvent]:
#         if self._closed:
#             raise RuntimeError("InMemoryEventBus is already closed")

#         queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
#         self._subscribers.add(queue)

#         async def iterator() -> AsyncIterator[AgentEvent]:
#             try:
#                 while True:
#                     event = await queue.get()
#                     if event is None:
#                         break
#                     yield event
#             finally:
#                 self._subscribers.discard(queue)

#         return iterator()

#     async def close(self) -> None:
#         if self._closed:
#             return
#         self._closed = True
#         if not self._subscribers:
#             return
#         queues = tuple(self._subscribers)
#         await asyncio.gather(*(queue.put(None) for queue in queues))
