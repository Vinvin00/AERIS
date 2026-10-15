"""Camera-motion estimator on synthetic pans with known shifts; no real-footage claim."""
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from aeris.estimation.camera_motion import CameraMotion, apply_transform

W, H = 640, 360


def world(seed=0):
    noise = (np.random.default_rng(seed).random((900, 1400)) * 255).astype(np.uint8)
    blurred = cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 3), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def box(cx, cy, w=30, h=16):
    return SimpleNamespace(x_min=cx - w / 2, y_min=cy - h / 2, x_max=cx + w / 2, y_max=cy + h / 2)


class CameraMotionTests(unittest.TestCase):
    def test_pan_is_recovered_and_a_masked_aircraft_does_not_bias_it(self):
        scene, motion = world(), CameraMotion()
        for i in range(30):
            ox, oy = 100 + 6 * i, 200 - 2 * i                  # camera pans right and up
            image = scene[oy:oy + H, ox:ox + W].copy()
            ax, ay = 400 + 3 * i - ox, 400 - oy                # aircraft moves right in the scene
            b = box(ax, ay)
            cv2.rectangle(image, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), (255, 255, 255), -1)
            assert motion.update(image, [b])
        assert np.allclose(motion.to_ref[:2, 2], (6 * 29, -2 * 29), atol=1.5)
        assert np.allclose(motion.to_reference((ax, ay)), (400 + 3 * 29 - 100, 200), atol=1.5)
        assert np.allclose(motion.from_reference(motion.to_reference((12.0, 34.0))), (12.0, 34.0))
        assert motion.counters == {"frames": 30, "estimated": 29, "failed": 0}

    def test_featureless_frames_fail_and_keep_the_previous_transform(self):
        motion, blank = CameraMotion(), np.full((H, W, 3), 128, np.uint8)
        assert motion.update(blank) and not motion.update(blank)
        assert np.array_equal(motion.to_ref, np.eye(3)) and motion.counters["failed"] == 1

    def test_apply_transform(self):
        h = np.array([[2.0, 0, 5], [0, 2.0, -1], [0, 0, 1]])
        assert apply_transform(h, (1.0, 2.0)) == (7.0, 3.0)
