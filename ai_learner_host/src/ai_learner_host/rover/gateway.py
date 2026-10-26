from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.domain.events import RoverConnectedEvent, RoverDisconnectedEvent
from ai_learner_host.domain.models import ValidatedPlan
from .protocol import ProtocolCodec


class RoverGateway:
    """Persistent bidirectional WebSocket adapter with one reader and one writer.

    All outbound messages are serialized through a single asyncio.Queue. This is the
    only class permitted to own the WebSocket object.
    """

    def __init__(
        self,
        event_bus: EventBus,
        *,
        reconnect_seconds: float = 3.0,
        auto_reconnect: bool = True,
    ) -> None:
        self.event_bus = event_bus
        self.reconnect_seconds = reconnect_seconds
        self.auto_reconnect = auto_reconnect
        self.uri = ""
        self._outbound: asyncio.Queue[str] = asyncio.Queue()
        self._runner: asyncio.Task[None] | None = None
        self._running = False
        self._connected = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    async def start(self, uri: str) -> None:
        self.uri = uri
        if self._runner and not self._runner.done():
            return
        self._running = True
        self._runner = asyncio.create_task(self._connection_loop(), name="rover-websocket")

    async def stop(self) -> None:
        self._running = False
        if self._runner:
            self._runner.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner
        self._runner = None
        self._connected.clear()

    async def reconnect(self, uri: str | None = None) -> None:
        await self.stop()
        await self.start(uri or self.uri)

    async def send_path(self, plan: ValidatedPlan) -> None:
        await self._outbound.put(ProtocolCodec.encode_path(plan))

    async def send_heading(self, deg: float) -> None:
        await self._outbound.put(ProtocolCodec.encode_heading(deg))

    async def set_telemetry(self, on: bool) -> None:
        await self._outbound.put(ProtocolCodec.encode_telemetry_switch(on))

    async def set_speed(self, base: int) -> None:
        await self._outbound.put(ProtocolCodec.encode_speed(base))

    async def set_pid(self, kp: float, ki: float, kd: float) -> None:
        await self._outbound.put(ProtocolCodec.encode_pid(kp, ki, kd))

    async def _connection_loop(self) -> None:
        while self._running:
            try:
                async with websockets.connect(
                    self.uri,
                    ping_interval=10,
                    ping_timeout=10,
                    close_timeout=2,
                    max_size=2**20,
                ) as websocket:
                    self._connected.set()
                    await self.event_bus.publish(RoverConnectedEvent(self.uri))
                    reader = asyncio.create_task(self._reader(websocket), name="rover-reader")
                    writer = asyncio.create_task(self._writer(websocket), name="rover-writer")
                    done, pending = await asyncio.wait(
                        {reader, writer}, return_when=asyncio.FIRST_EXCEPTION
                    )
                    for task in pending:
                        task.cancel()
                    for task in pending:
                        with suppress(asyncio.CancelledError):
                            await task
                    for task in done:
                        exception = task.exception()
                        if exception:
                            raise exception
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # transport boundary: publish rather than crash app
                self._connected.clear()
                await self.event_bus.publish(RoverDisconnectedEvent(str(exc)))
                if not self._running or not self.auto_reconnect:
                    return
                await asyncio.sleep(self.reconnect_seconds)
            else:
                self._connected.clear()
                await self.event_bus.publish(RoverDisconnectedEvent("connection closed"))
                if not self.auto_reconnect:
                    return
                await asyncio.sleep(self.reconnect_seconds)

    async def _reader(self, websocket: Any) -> None:
        try:
            async for raw in websocket:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                await self.event_bus.publish(ProtocolCodec.decode(raw))
        except ConnectionClosed as exc:
            raise RuntimeError(f"WebSocket closed: {exc.code}") from exc

    async def _writer(self, websocket: Any) -> None:
        try:
            while self._running:
                message = await self._outbound.get()
                await websocket.send(message)
        except ConnectionClosed as exc:
            raise RuntimeError(f"WebSocket closed during send: {exc.code}") from exc
