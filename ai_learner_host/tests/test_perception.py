from __future__ import annotations

import unittest

import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import AppConfig
from ai_learner_host.perception.service import (
    CalibrationService,
    ColourCalibrationService,
    PerceptionService,
)


class CalibrationServiceTests(unittest.TestCase):
    def test_world_frame_origin_is_bottom_left(self):
        service = CalibrationService()
        profile = service.compute(
            [[0, 0], [100, 0], [100, 100], [0, 100]],
            1000,
            1000,
            (100, 100, 3),
        )
        x1, y1 = service.pixel_to_world(0, 0, profile)
        x2, y2 = service.pixel_to_world(0, 100, profile)
        self.assertAlmostEqual(x1, 0.0, places=2)
        self.assertAlmostEqual(y1, 1000.0, places=2)
        self.assertAlmostEqual(x2, 0.0, places=2)
        self.assertAlmostEqual(y2, 0.0, places=2)


class ColourCalibrationTests(unittest.TestCase):
    def test_four_colour_profile_is_valid(self):
        service = ColourCalibrationService()

        def patch(h, s=220, v=230):
            return np.tile(np.array([[h, s, v]], dtype=np.uint8), (441, 1))

        samples = {
            "red": [patch(1), patch(178), patch(0), patch(2), patch(177)],
            "green": [patch(60) for _ in range(5)],
            "yellow": [patch(30) for _ in range(5)],
            "blue": [patch(100) for _ in range(5)],
        }
        profile = service.compute(samples)
        self.assertTrue(profile.valid)
        self.assertIn("blue", profile.colours)
        self.assertTrue(profile.colours["red"]["hue_ranges"])


class WorldMapTests(unittest.TestCase):
    def test_text_map_contains_rover_and_blocks(self):
        config = AppConfig()
        perception = PerceptionService(FrameBuffer(), config)
        scene = {
            "arena": {"width_mm": 1000.0, "height_mm": 1000.0},
            "rover": {
                "available": True,
                "pose_valid": True,
                "confidence": 1.0,
                "pose": {"x_mm": 500.0, "y_mm": 500.0, "heading_deg": 90.0},
            },
            "obstacles": [
                {
                    "id": "red_block_01",
                    "colour": "red",
                    "centre_mm": {"x": 100.0, "y": 100.0},
                }
            ],
        }
        text_map = perception._text_map(scene, cell_mm=100.0)
        self.assertIn("R", text_map)
        self.assertIn(">", text_map)
        self.assertIn("Origin: bottom-left", text_map)


if __name__ == "__main__":
    unittest.main()
