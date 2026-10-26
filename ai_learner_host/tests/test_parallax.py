import math
import unittest

import numpy as np

from ai_learner_host.camera.frame_buffer import FrameBuffer
from ai_learner_host.config import AppConfig
from ai_learner_host.perception.service import PerceptionService


class MarkerParallaxTests(unittest.TestCase):
    def service(self, H=0.0, h=0.0, nx=0.0, ny=0.0):
        cfg = AppConfig()
        cfg.parallax_camera_height_mm = H
        cfg.parallax_marker_height_mm = h
        cfg.parallax_nadir_x_mm = nx
        cfg.parallax_nadir_y_mm = ny
        return PerceptionService(FrameBuffer(), cfg)

    def test_disabled_by_default(self):
        p = np.array([900.0, 600.0])
        np.testing.assert_allclose(self.service().correct_marker_parallax(p), p)

    def test_inverts_the_projection_of_a_raised_marker(self):
        H, h, n = 1590.0, 55.0, np.array([1032.0, -828.0])
        svc = self.service(H, h, *n)
        for true in ([300.0, 250.0], [855.0, 627.0], [1400.0, 1000.0]):
            true = np.array(true)
            seen = n + (true - n) * H / (H - h)  # what the floor homography reports
            np.testing.assert_allclose(svc.correct_marker_parallax(seen), true, atol=1e-9)

    def test_magnitude_matches_tape_readings(self):
        # Tape (1482, 798) was reported by the camera as (1502, 864) before the correction.
        svc = self.service(1549.6, 55.0, 1032.0, -828.0)
        corrected = svc.correct_marker_parallax(np.array([1502.0, 864.0]))
        self.assertLess(math.dist(corrected, (1482.0, 798.0)), 10.0)

    def test_scaling_preserves_direction(self):
        svc = self.service(1590.0, 55.0, 1000.0, -800.0)
        a = svc.correct_marker_parallax(np.array([400.0, 300.0]))
        b = svc.correct_marker_parallax(np.array([400.0, 900.0]))
        self.assertAlmostEqual(a[0], b[0], places=6)  # a north-pointing segment stays north-pointing


if __name__ == "__main__":
    unittest.main()
