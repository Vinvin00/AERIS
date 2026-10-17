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


def sharp(seed, shape):
    """High-frequency texture: its corners outrank the blurred background's in goodFeaturesToTrack."""
    noise = (np.random.default_rng(seed).random(shape) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(noise, (0, 0), 0.8), cv2.COLOR_GRAY2BGR)


class CameraMotionMaskAndScaleTests(unittest.TestCase):
    def test_a_strongly_textured_moving_target_is_masked_out(self):
        """The target's corners dominate the frame; it moves against the pan. With its box masked
        the pan is recovered; without the box the fit follows the target. Run at 640x360 (no
        downscale) and at 1280x720 (fit on a 640 px copy, so the mask must be scaled too)."""
        for width, height, step in ((640, 360, 6), (1280, 720, 12)):
            scene = cv2.resize(world(), (1400 * width // 640, 900 * width // 640))
            th, tw = height * 5 // 9, width * 15 // 32
            target = sharp(5, (th, tw))

            def pan_estimate(masked):
                motion = CameraMotion()
                for i in range(20):
                    ox = 100 + step * i
                    image = scene[200:200 + height, ox:ox + width].copy()
                    tx, ty = width * 15 // 32 - 8 * i * width // 640, height // 5
                    image[ty:ty + th, tx:tx + tw] = target
                    b = SimpleNamespace(x_min=tx, y_min=ty, x_max=tx + tw, y_max=ty + th)
                    assert motion.update(image, [b] if masked else [])
                return motion.to_ref[0, 2]
            self.assertLess(abs(pan_estimate(True) - step * 19), 2 * width / 640, width)
            self.assertGreater(abs(pan_estimate(False) - step * 19), 20, width)

    def test_hd_frames_use_the_downscaled_fit_and_recover_rotation_and_zoom(self):
        """1280x720 frames (estimated on a 640 px copy): uneven pan, 1.5-3.5 deg rotation and 0.3 % zoom per frame.
        The chained transform must match the true frame->first-frame map at the frame corners."""
        big = cv2.resize(world(3), (2800, 1800), interpolation=cv2.INTER_LINEAR)
        cx, cy = 1400.0, 900.0

        def world_to_frame(i):                       # per-frame steps vary, so chain order matters
            a = cv2.getRotationMatrix2D((cx, cy), 1.5 * i + 1.0 * (i % 3), 1.003 ** i)
            a[:, 2] += (-(cx - 640) - 5 * i - 0.8 * i * i, -(cy - 360) + 6 * (i % 4))
            return np.vstack([a, [0, 0, 1]])
        motion = CameraMotion()
        for i in range(16):
            frame = cv2.warpAffine(big, world_to_frame(i)[:2], (1280, 720), flags=cv2.INTER_LINEAR)
            assert motion.update(frame)
        truth = world_to_frame(0) @ np.linalg.inv(world_to_frame(15))       # frame 15 -> frame 0
        corners = [(0, 0), (1279, 0), (0, 719), (1279, 719)]
        errors = [np.hypot(*np.subtract(motion.to_reference(c), apply_transform(truth, c))) for c in corners]
        assert max(errors) < 4.0, errors
