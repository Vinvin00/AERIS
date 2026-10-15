"""Analytic contract checks for surveyed-site bearings; no real-camera accuracy claim."""
from __future__ import annotations

import json
import math
import unittest

from aeris.estimation.camera import Calibration, CameraModel, FrameContext, GeometryError
from aeris.estimation.world_bearing import (FixedCameraPose, WorldBearingRecord,
                                            world_bearing_from_pixel)


CALIBRATION = {
    "camera_id": "fixed-1", "width": 200, "height": 200,
    "K": [[100, 0, 100], [0, 100, 100], [0, 0, 1]],
    "distortion_model": "none", "distortion_coefficients": [],
    "calibration_id": "lens-1", "quality": "analytic fixture", "lens": "fixture", "zoom": 1.0,
}
POSE = {
    "schema_version": 1, "pose_id": "survey-1", "camera_id": "fixed-1",
    "calibration_id": "lens-1", "reference_id": "survey-origin-1",
    "position_enu_m": [10.0, 20.0, 3.0],
    # camera right -> east, down -> down, forward -> north
    "rotation_camera_to_enu": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    "surveyed_at_utc": "2026-10-15T09:00:00Z", "fixed_mount": True,
}


class WorldBearingTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraModel(Calibration.from_dict(CALIBRATION))
        self.pose = FixedCameraPose.from_dict(POSE)
        self.kw = dict(camera=self.camera, pose=self.pose, sequence_id="s", track_id="a",
                       issue_time_s=2.0, frame_size=(200, 200), context=FrameContext(zoom=1.0),
                       camera_stable=True)

    def bearing(self, pixel, **changes):
        return world_bearing_from_pixel(pixel=pixel, **{**self.kw, **changes})

    def test_center_right_up_and_forecast_timing(self):
        center = self.bearing((100, 100))
        self.assertEqual(center.direction_enu, (0.0, 1.0, 0.0))
        self.assertEqual((center.azimuth_deg, center.elevation_deg), (0.0, 0.0))
        self.assertEqual(center.origin_enu_m, (10.0, 20.0, 3.0))
        self.assertIsNone(center.angular_uncertainty_rad)
        right = self.bearing((200, 100), horizon_s=0.5)
        self.assertEqual((right.kind, right.issue_time_s, right.target_time_s), ("forecast", 2.0, 2.5))
        self.assertAlmostEqual(right.azimuth_deg, 45.0)
        up = self.bearing((100, 50))
        self.assertAlmostEqual(up.elevation_deg, math.degrees(math.atan(0.5)))
        self.assertEqual(WorldBearingRecord.from_dict(json.loads(json.dumps(up.to_dict()))), up)
        self.assertEqual(FixedCameraPose.from_dict(json.loads(json.dumps(self.pose.to_dict()))), self.pose)

    def test_distorted_pixel_uses_camera_model_undistortion(self):
        distorted = Calibration.from_dict({**CALIBRATION, "distortion_model": "brown_conrady",
                                           "distortion_coefficients": [0.03, -0.01, 0.001, 0.002, 0.0]})
        camera = CameraModel(distorted)
        pixel = camera.ray_to_pixel((0.2, -0.1, 1.0))
        result = self.bearing(pixel, camera=camera)
        self.assertIsNone(result.availability_reason)
        self.assertAlmostEqual(result.azimuth_deg, math.degrees(math.atan2(0.2, 1.0)), places=7)
        self.assertAlmostEqual(result.elevation_deg,
                               math.degrees(math.atan2(0.1, math.hypot(0.2, 1.0))), places=7)

    def test_missing_and_mismatched_inputs_are_null_with_reason(self):
        cases = (
            {"camera": CameraModel(None)},
            {"pose": None},
            {"pose": FixedCameraPose.from_dict({**POSE, "calibration_id": "other"})},
            {"pose": FixedCameraPose.from_dict({**POSE, "fixed_mount": False})},
            {"camera_stable": None},
            {"camera_stable": False},
            {"camera_stable": 1},
            {"frame_size": (100, 200)},
            {"frame_size": None},
            {"context": None},
            {"context": FrameContext(zoom=None)},
            {"context": FrameContext(zoom=1.1)},
            {"context": FrameContext(zoom=1.0, camera_moved=True)},
        )
        for change in cases:
            with self.subTest(change=change):
                result = self.bearing((100, 100), **change)
                self.assertIsNone(result.direction_enu)
                self.assertIsNone(result.azimuth_deg)
                self.assertIsNone(result.origin_enu_m)
                self.assertTrue(result.availability_reason)
        self.assertIn("outside", self.bearing((250, 100)).availability_reason)
        self.assertIn("unassessed", self.bearing((100, 100), camera_stable=None,
                                                 context=FrameContext(zoom=1.0)).availability_reason)

    def test_pose_and_record_reject_invalid_geometry_and_time(self):
        bad_poses = (
            {"rotation_camera_to_enu": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"rotation_camera_to_enu": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"position_enu_m": [math.inf, 0, 0]},
            {"surveyed_at_utc": "2026-10-15T09:00:00"},
            {"fixed_mount": 1},
            {"schema_version": 2},
        )
        for change in bad_poses:
            with self.subTest(change=change), self.assertRaises(GeometryError):
                FixedCameraPose.from_dict({**POSE, **change})
        valid = self.bearing((100, 100)).to_dict()
        for change in ({"target_time_s": 3}, {"azimuth_deg": 180},
                       {"direction_enu": [0, 2, 0]}, {"availability_reason": "unknown"},
                       {"angular_uncertainty_rad": 0.1}):
            with self.subTest(change=change), self.assertRaises(GeometryError):
                WorldBearingRecord.from_dict({**valid, **change})
        with self.assertRaises(GeometryError):
            self.bearing((100, 100), horizon_s=-0.1)


if __name__ == "__main__":
    unittest.main()
