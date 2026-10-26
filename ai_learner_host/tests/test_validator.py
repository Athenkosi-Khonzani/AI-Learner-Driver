import unittest

from ai_learner_host.config import AppConfig
from ai_learner_host.domain.models import DriveCommand, PlanProposal
from ai_learner_host.planning.validator import PlanValidationError, PlanValidator


class PlanValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator(AppConfig(max_primitives=6, max_forward_mm=1000))

    def test_valid_plan(self):
        plan = self.validator.validate(
            PlanProposal([DriveCommand("FWD", 500), DriveCommand("TURN", 90), DriveCommand("STOP")])
        )
        self.assertEqual(plan.commands[0].display(), "FORWARD 500 mm")
        self.assertEqual(plan.commands[-1].op, "STOP")

    def test_requires_stop(self):
        with self.assertRaises(PlanValidationError):
            self.validator.validate(PlanProposal([DriveCommand("FWD", 100)]))

    def test_rejects_unknown_operation(self):
        with self.assertRaises(PlanValidationError):
            self.validator.validate(PlanProposal([DriveCommand("FLY", 10), DriveCommand("STOP")]))


if __name__ == "__main__":
    unittest.main()
