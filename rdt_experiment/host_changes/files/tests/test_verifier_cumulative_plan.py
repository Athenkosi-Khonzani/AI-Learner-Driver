import unittest

from ai_learner_host.application.event_bus import EventBus
from ai_learner_host.application.mission_controller import MissionController
from ai_learner_host.application.state_store import AppStateStore
from ai_learner_host.config import AppConfig
from ai_learner_host.domain.events import CommandAckEvent
from ai_learner_host.domain.models import DriveCommand, MissionRun, ValidatedPlan
from ai_learner_host.rover.execution_tracker import ExecutionTracker


class _Repo:
    async def save_command_ack(self, *a):
        pass


class CumulativePlanTests(unittest.IsolatedAsyncioTestCase):
    def controller(self):
        mc = MissionController(
            config=AppConfig(), event_bus=EventBus(), state_store=AppStateStore(), planner=None,
            perception=None, validator=None, verifier=None, rover=None,
            execution=ExecutionTracker(), repository=_Repo(),
        )
        mc.current_run = MissionRun(prompt="x")
        mc._verify_after_path = lambda: None  # not under test here
        return mc

    def plan(self, pid, *cmds):
        return ValidatedPlan(pid, [DriveCommand(op, v) for op, v in cmds] + [DriveCommand("STOP")])

    async def test_correction_round_sees_both_plans(self):
        mc = self.controller()
        first = self.plan(1, ("TURN", 90), ("FWD", 1105))
        mc._executed_rounds.append({"plan": first, "completed": True, "last_acked": 2})
        correction = self.plan(2, ("FWD", 89))
        mc._executed_rounds.append({"plan": correction, "completed": True, "last_acked": 1})
        cmds = [(c.op, c.value) for c in mc.cumulative_executed_plan().commands]
        self.assertEqual(cmds, [("TURN", 90), ("FWD", 1105), ("FWD", 89), ("STOP", None)])

    async def test_stopped_path_contributes_only_acknowledged_commands(self):
        mc = self.controller()
        plan = self.plan(3, ("FWD", 300), ("TURN", 90), ("FWD", 500))
        mc._executed_rounds.append({"plan": plan, "completed": False, "last_acked": -1})
        mc.current_run.current_path_id = 3
        await mc._on_command_ack(CommandAckEvent(3, 0))
        await mc._on_command_ack(CommandAckEvent(3, 1))
        cmds = [(c.op, c.value) for c in mc.cumulative_executed_plan().commands]
        self.assertEqual(cmds, [("FWD", 300), ("TURN", 90), ("STOP", None)])

    async def test_new_mission_starts_empty(self):
        mc = self.controller()
        mc._executed_rounds.append({"plan": self.plan(1, ("FWD", 10)), "completed": True, "last_acked": 1})
        mc._executed_rounds = []
        self.assertEqual([c.op for c in mc.cumulative_executed_plan().commands], ["STOP"])


if __name__ == "__main__":
    unittest.main()
