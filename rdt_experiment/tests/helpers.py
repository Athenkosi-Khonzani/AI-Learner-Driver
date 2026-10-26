"""Hand-built host-style world states for scoring tests."""


def scene(x, y, h, blocks=(), arena=(1710.0, 1254.0), conf=0.95):
    obstacles = []
    for colour, bx, by, long_side, short_side, host_orient in blocks:
        import math
        a = math.radians(host_orient)
        ux, uy = math.cos(a) * long_side / 2, math.sin(a) * long_side / 2
        vx, vy = -math.sin(a) * short_side / 2, math.cos(a) * short_side / 2
        poly = [(bx + ux + vx, by + uy + vy), (bx - ux + vx, by - uy + vy), (bx - ux - vx, by - uy - vy), (bx + ux - vx, by + uy - vy)]
        obstacles.append({
            "id": f"{colour}_block_01", "type": "coloured_block", "colour": colour,
            "centre_mm": {"x": bx, "y": by}, "size_mm": {"long_side": long_side, "short_side": short_side},
            "orientation_deg": host_orient, "polygon_mm": [{"x": px, "y": py} for px, py in poly],
        })
    return {
        "schema": "ai_learner_world_state", "schema_version": "1.0",
        "arena": {"width_mm": arena[0], "height_mm": arena[1]},
        "rover": {"available": True, "pose_valid": True, "confidence": conf,
                  "footprint_mm": {"length": 155.5, "width": 121.3},
                  "pose": None if x is None else {"x_mm": x, "y_mm": y, "heading_deg": h}, "stale": False, "age_s": 0.0},
        "obstacles": obstacles,
    }


def record(task, w0, after, commands, *, nacks=(), path_ack=True, verification=False, trail=None, annotations=None,
           extra_rounds=()):
    rounds = [{
        "index": 0, "source": "planner", "revision": 0, "path_id": 1, "commands": commands,
        "start_world_key": "W0", "end_world_key": "after_0" if after is not None else None,
        "t_path_sent": 1.0, "t_path_ack": 5.0 if path_ack else None, "nacks": list(nacks),
        "command_acks": [], "verifier": None, "planner_latency_s": 2.0,
    }]
    worlds = {"W0": {"scene": w0}}
    if after is not None:
        worlds["after_0"] = {"scene": after}
    for i, (cmds, end_scene) in enumerate(extra_rounds, start=1):
        worlds[f"after_{i}"] = {"scene": end_scene}
        rounds.append({"index": i, "source": "verification", "revision": i, "path_id": i + 1, "commands": cmds,
                       "start_world_key": f"after_{i-1}", "end_world_key": f"after_{i}", "t_path_sent": 10.0 * i,
                       "t_path_ack": 10.0 * i + 3, "nacks": [], "command_acks": [], "verifier": None})
    return {
        "meta": {"instruction": task.instruction, "verification_enabled": verification},
        "status": "complete", "placement": {"ok": True}, "rounds": rounds, "worlds": worlds,
        "events": [], "trail": trail or [], "annotations": annotations or {}, "timing": {},
    }
