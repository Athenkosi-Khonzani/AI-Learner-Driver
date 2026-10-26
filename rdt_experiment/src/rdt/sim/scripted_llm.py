"""A deterministic stand-in planner/verifier for rehearsals and tests.

It parses the instruction, drives in a straight line to the target and ignores
obstacles, so it fails the obstacle tasks; that is useful for checking the
failure attribution. It never calls a network service.
"""

from __future__ import annotations

import asyncio
import json
import math
import re

from ai_learner_host.planning.llm_providers import StructuredLLMResult

COMPASS = {"north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0}
DIRECTION = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _json_after(label: str, text: str) -> dict | None:
    idx = text.find(label)
    if idx < 0:
        return None
    start = text.find("{", idx)
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : j + 1])
    return None


def _instruction(text: str) -> str:
    for label in ("USER INSTRUCTION:", "ORIGINAL USER INSTRUCTION:"):
        if label in text:
            return text.split(label, 1)[1].strip().splitlines()[0]
    return text


def target_from_instruction(instr: str, world: dict) -> tuple[float | None, float | None, float | None]:
    """Return (x, y, final_heading) implied by the instruction, any may be None."""
    lower = instr.lower()
    pose = world["rover"]["pose"]
    x, y, h = pose["x_mm"], pose["y_mm"], pose["heading_deg"]
    final = next((v for k, v in COMPASS.items() if f"face {k}" in lower), None)
    m = re.search(r"coordinate \((\d+(?:\.\d+)?) mm, (\d+(?:\.\d+)?) mm\)", lower)
    if m:
        return float(m.group(1)), float(m.group(2)), final
    m = re.search(r"(\d+) mm directly (north|south|east|west) of the (\w+) block", lower)
    if m:
        dist, direction, colour = float(m.group(1)), m.group(2), m.group(3)
        block = next((o for o in world.get("obstacles", []) if o["colour"] == colour), None)
        if block:
            dx, dy = DIRECTION[direction]
            return block["centre_mm"]["x"] + dx * dist, block["centre_mm"]["y"] + dy * dist, final
    m = re.search(r"turn (right|left) by (\d+) degrees, move forward (\d+) mm", lower)
    if m:
        turn = float(m.group(2)) * (1 if m.group(1) == "right" else -1)
        nh = math.radians(h + turn)
        d = float(m.group(3))
        return x + d * math.sin(nh), y + d * math.cos(nh), (h + turn) % 360
    m = re.search(r"move forward (\d+) mm", lower)
    if m:
        d = float(m.group(1))
        return x + d * math.sin(math.radians(h)), y + d * math.cos(math.radians(h)), None
    return None, None, final


def straight_line_plan(x, y, h, tx, ty, final) -> list[dict]:
    cmds: list[dict] = []
    heading = h
    if tx is not None:
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist > 5:
            bearing = math.degrees(math.atan2(dx, dy)) % 360
            turn = _wrap(bearing - heading)
            if abs(turn) >= 1:
                cmds.append({"op": "TURN", "value": round(turn, 1)})
            heading = bearing
            remaining = dist
            while remaining > 0.5:
                step = min(remaining, 2000.0)
                cmds.append({"op": "FWD", "value": round(step, 1)})
                remaining -= step
    if final is not None:
        turn = _wrap(final - heading)
        if abs(turn) >= 1:
            cmds.append({"op": "TURN", "value": round(turn, 1)})
    cmds.append({"op": "STOP", "value": None})
    return cmds


class ScriptedLLM:
    def __init__(self, latency_s: float = 0.3, tolerance_mm: float = 40.0) -> None:
        self.latency = latency_s
        self.tolerance = tolerance_mm
        self.provider, self.model = "scripted", "straight-line-v1"

    def readiness(self) -> tuple[bool, str]:
        return True, "Scripted rehearsal planner ready"

    async def generate_plan(self, system_prompt: str, user_prompt: str) -> StructuredLLMResult:
        await asyncio.sleep(self.latency)
        world = _json_after("WORLD STATE JSON:", user_prompt)
        instr = _instruction(user_prompt)
        pose = world["rover"]["pose"]
        tx, ty, final = target_from_instruction(instr, world)
        cmds = straight_line_plan(pose["x_mm"], pose["y_mm"], pose["heading_deg"], tx, ty, final)
        data = {"cmds": cmds}
        return StructuredLLMResult(data, json.dumps(data), self.provider, self.model)

    async def generate_verification(self, system_prompt: str, user_prompt: str) -> StructuredLLMResult:
        await asyncio.sleep(self.latency)
        before = _json_after("BEFORE WORLD STATE JSON:", user_prompt)
        after = _json_after("AFTER WORLD STATE JSON:", user_prompt)
        instr = _instruction(user_prompt)
        tx, ty, final = target_from_instruction(instr, before)
        pose = after["rover"]["pose"]
        if pose is None or tx is None:
            data = {"success": False, "message": "cannot judge", "cmds": []}
        else:
            err = math.hypot(pose["x_mm"] - tx, pose["y_mm"] - ty)
            herr = abs(_wrap(pose["heading_deg"] - final)) if final is not None else 0.0
            if err <= self.tolerance and herr <= 8:
                data = {"success": True, "message": f"within {err:.0f} mm", "cmds": []}
            else:
                cmds = straight_line_plan(pose["x_mm"], pose["y_mm"], pose["heading_deg"], tx, ty, final)
                data = {"success": False, "message": f"{err:.0f} mm from target", "cmds": cmds}
        return StructuredLLMResult(data, json.dumps(data), self.provider, self.model)
