from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")
Handler = Callable[[Any], Awaitable[None] | None]


class EventBus:
    """Small in-process event bus used to decouple transports from application logic."""

    def __init__(self) -> None:
        self._subscribers: dict[type[Any], list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None] | None]) -> None:
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Any) -> None:
        handlers = tuple(self._subscribers.get(type(event), ()))
        for handler in handlers:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
