from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class ConnectionManager:
    """Coordinates application startup and shutdown without coupling the UI to adapters."""

    def __init__(self) -> None:
        self._starters: list[Callable[[], Awaitable[None]]] = []
        self._stoppers: list[Callable[[], Awaitable[None]]] = []

    def register(self, start: Callable[[], Awaitable[None]], stop: Callable[[], Awaitable[None]]) -> None:
        self._starters.append(start)
        self._stoppers.append(stop)

    async def start_all(self) -> None:
        if self._starters:
            await asyncio.gather(*(start() for start in self._starters))

    async def stop_all(self) -> None:
        if self._stoppers:
            await asyncio.gather(*(stop() for stop in reversed(self._stoppers)), return_exceptions=True)
