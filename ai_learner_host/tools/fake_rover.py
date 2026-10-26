"""Small development-only ESP32 rover simulator.

Run this separately, point Settings at ws://127.0.0.1:8765, and the host can exercise
its WebSocket/event flow without the physical rover. TURN commands issue three
req_heading messages to exercise the fast camera-heading path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

import websockets


class FakeRover:
    def __init__(self) -> None:
        self.telemetry_enabled = False
        self.heading_replies: asyncio.Queue[float] = asyncio.Queue()
        self.send_lock = asyncio.Lock()
        self.path_task: asyncio.Task | None = None

    async def send(self, ws, payload: dict) -> None:
        async with self.send_lock:
            await ws.send(json.dumps(payload, separators=(",", ":")))

    async def handle(self, ws) -> None:
        print("host connected")
        telemetry = asyncio.create_task(self._telemetry_loop(ws))
        try:
            async for raw in ws:
                message = json.loads(raw)
                print("RX", message)
                t = message.get("t")
                if t == "telem":
                    self.telemetry_enabled = bool(message.get("on"))
                elif t == "heading":
                    await self.heading_replies.put(float(message["deg"]))
                elif t == "path":
                    if self.path_task and not self.path_task.done():
                        await self.send(
                            ws,
                            {"t": "nack", "id": message.get("id", -1), "reason": "busy"},
                        )
                    else:
                        self.path_task = asyncio.create_task(self._execute_path(ws, message))
                elif t in {"speed", "pid"}:
                    pass
        finally:
            telemetry.cancel()
            if self.path_task:
                self.path_task.cancel()
            print("host disconnected")

    async def _execute_path(self, ws, message: dict) -> None:
        path_id = int(message["id"])
        commands = message.get("cmds", [])
        for wire_index, command in enumerate(commands):
            op = command[0]
            if op == "TURN":
                for _ in range(3):
                    await self.send(ws, {"t": "req_heading"})
                    try:
                        heading = await asyncio.wait_for(self.heading_replies.get(), timeout=1.5)
                        print("heading reply", heading)
                    except asyncio.TimeoutError:
                        print("heading reply timeout (expected while perception is placeholder)")
                    await asyncio.sleep(0.08)
            if op != "STOP":
                await asyncio.sleep(0.35)
            await self.send(ws, {"t": "ack", "of": "cmd", "id": path_id, "i": wire_index})
        await self.send(ws, {"t": "ack", "of": "path", "id": path_id})

    async def _telemetry_loop(self, ws) -> None:
        soc = 82.0
        while True:
            await asyncio.sleep(0.5)
            if not self.telemetry_enabled:
                continue
            soc = max(0.0, soc - 0.02)
            await self.send(
                ws,
                {
                    "t": "telem",
                    "vbat": round(7.7 + random.uniform(-0.04, 0.04), 2),
                    "soc": round(soc, 1),
                    "dist": round(35.0 + random.uniform(-8.0, 8.0), 1),
                },
            )


async def main(host: str, port: int) -> None:
    rover = FakeRover()
    async with websockets.serve(rover.handle, host, port):
        print(f"Fake rover listening on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        pass
