"""Camera geometry tests. Independent closed-form / NumPy forward models serve as truth."""
import dataclasses
import json
import math
import unittest
from pathlib import Path

import numpy as np

from aeris.data.contracts import BBox
from aeris.estimation.camera import (
    CameraModel, Calibration, FrameContext, GeometryError, ImageTransform,
    SizeChange,
)

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "tests" / "fixtures" / "simulation"
PINHOLE = json.loads((SIM / "input" / "camera.json").read_text())
DISTORTED = {**PINHOLE, "camera_id": "distorted", "width": 640, "height": 480,
             "K": [[500.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]],
             "distortion_model": "brown_conrady",
             "distortion_coefficients": [-0.2, 0.05, 0.001, -0.0008, 0.01], "quality": "synthetic"}


def forward_pixel(cal, ray):
    """Independent Brown-Conrady forward model (not OpenCV): ray -> distorted pixel."""
    x, y = ray[0] / ray[2], ray[1] / ray[2]
    k1, k2, p1, p2, k3 = (list(cal["distortion_coefficients"]) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return cal["K"][0][0] * xd + cal["K"][0][2], cal["K"][1][1] * yd + cal["K"][1][2]


def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


class ProjectionRoundTripTests(unittest.TestCase):
    def test_pinhole_rays_match_analytic_world_points(self):
        model = CameraModel(Calibration.from_dict(PINHOLE))
        for line in (SIM / "truth" / "projection.jsonl").read_text().splitlines():
            truth = json.loads(line)
            ray = model.pixel_to_ray(*truth["projected_center_xy_px"]).direction
            np.testing.assert_allclose(ray, unit(truth["world_xyz_m"]), atol=1e-12)

    def test_distortion_is_removed_before_the_ray_is_formed(self):
        model = CameraModel(Calibration.from_dict(DISTORTED))
        for x in np.linspace(-0.4, 0.4, 9):
            for y in np.linspace(-0.3, 0.3, 7):
                ray = unit([x, y, 1.0])
                pixel = forward_pixel(DISTORTED, ray)
                np.testing.assert_allclose(model.pixel_to_ray(*pixel).direction, ray, atol=1e-9)

    def test_ray_pixel_round_trip_with_distortion_incl_corners(self):
        model = CameraModel(Calibration.from_dict(DISTORTED))
        for u, v in [(1, 1), (639, 1), (1, 479), (639, 479), (320, 240), (100, 400), (555, 33.5)]:
            ray = model.pixel_to_ray(u, v).direction
            np.testing.assert_allclose(model.ray_to_pixel(ray), (u, v), atol=1e-6)

    def test_crop_resize_pad_transforms_round_trip(self):
        model = CameraModel(Calibration.from_dict(DISTORTED))
        transform = ImageTransform(crop_x=100, crop_y=60, scale_x=0.5, scale_y=0.75, pad_x=3, pad_y=5)
        for original in [(150.0, 100.0), (400.0, 300.0), (101.0, 61.0)]:
            processed = transform.from_original(*original)
            np.testing.assert_allclose(transform.to_original(*processed), original, atol=1e-12)
            direct = model.pixel_to_ray(*original).direction
            via = model.pixel_to_ray(*processed, transform).direction
            np.testing.assert_allclose(via, direct, atol=1e-12)
            np.testing.assert_allclose(model.ray_to_pixel(direct, transform), processed, atol=1e-6)

    def test_letterbox_matches_the_detector_inverse(self):
        # AERIS-006: 100x200 image, ratio 2.08, model box (208,104)-(416,208) == original (100,50)-(200,100)
        transform = ImageTransform.letterbox(2.08)
        np.testing.assert_allclose(transform.to_original(208, 104), (100, 50))
        np.testing.assert_allclose(transform.to_original(416, 208), (200, 100))

    def test_invalid_transform_and_inputs(self):
        for bad in ({"scale_x": 0}, {"scale_y": -1}, {"crop_x": float("nan")}):
            with self.assertRaises(GeometryError):
                ImageTransform(**bad)
        model = CameraModel(Calibration.from_dict(PINHOLE))
        with self.assertRaises(GeometryError):
            model.pixel_to_ray(float("nan"), 1)
        with self.assertRaises(GeometryError):
            model.ray_to_pixel((0, 0, -1))

    def test_pixel_outside_calibrated_frame_has_no_ray(self):
        ray = CameraModel(Calibration.from_dict(PINHOLE)).pixel_to_ray(70, 10)
        assert ray.direction is None and "outside" in ray.availability_reason["direction"]


class CalibrationValidationTests(unittest.TestCase):
    def test_bad_calibrations_rejected(self):
        for change in ({"K": [[0, 0, 32], [0, 100, 24], [0, 0, 1]]},
                       {"K": [[100, 0, 99], [0, 100, 24], [0, 0, 1]]},
                       {"K": [[100, 0.5, 32], [0, 100, 24], [0, 0, 1]]},
                       {"K": [[100, 0, 32], [0, 100, 24]]},
                       {"distortion_model": "fisheye"},
                       {"distortion_model": "none", "distortion_coefficients": [0.1]},
                       {"distortion_model": "brown_conrady", "distortion_coefficients": [0.1, 0.2]},
                       {"zoom": 0}, {"width": 0}, {"calibration_id": " "},
                       {"distortion_coefficients": [float("nan")], "distortion_model": "brown_conrady"}):
            with self.assertRaises(GeometryError, msg=str(change)):
                Calibration.from_dict({**PINHOLE, **change})
        with self.assertRaises(GeometryError):
            Calibration.from_dict({k: v for k, v in PINHOLE.items() if k != "zoom"})


class ExtentAndRateTests(unittest.TestCase):
    def setUp(self):
        self.model = CameraModel(Calibration.from_dict(PINHOLE))

    def test_extent_equals_edge_ray_angle_and_is_angular_not_metric(self):
        extent = self.model.bbox_angular_extent(BBox(22, 20, 42, 28))
        f, cx, cy = 100.0, 32.0, 24.0
        assert abs(extent.width_rad - (math.atan((42 - cx) / f) - math.atan((22 - cx) / f))) < 1e-12
        assert abs(extent.height_rad - (math.atan((28 - cy) / f) - math.atan((20 - cy) / f))) < 1e-12
        assert abs(extent.width_deg - math.degrees(extent.width_rad)) < 1e-12
        assert not any("meter" in f.name or f.name.endswith("_m") for f in dataclasses.fields(extent))

    def test_extent_is_independent_of_the_processing_transform(self):
        transform = ImageTransform(scale_x=2.0, scale_y=2.0)
        original = self.model.bbox_angular_extent(BBox(22, 20, 42, 28))
        processed = self.model.bbox_angular_extent(BBox(44, 40, 84, 56), transform)
        assert abs(original.width_rad - processed.width_rad) < 1e-12

    def test_angular_rate_is_camera_relative_rad_per_second(self):
        rate = self.model.angular_rate((32, 24), 1.0, (42, 24), 1.5)
        assert abs(rate.rate_rad_s - math.atan(0.1) / 0.5) < 1e-12
        assert abs(rate.azimuth_rate_rad_s - math.atan(0.1) / 0.5) < 1e-12 and abs(rate.elevation_rate_rad_s) < 1e-12
        assert abs(rate.rate_deg_s - math.degrees(rate.rate_rad_s)) < 1e-12
        down = self.model.angular_rate((32, 24), 0, (32, 34), 1)   # image y down = negative elevation
        assert down.elevation_rate_rad_s < 0
        for t1 in (1.0, 0.5):
            with self.assertRaises(GeometryError):
                self.model.angular_rate((32, 24), 1.0, (42, 24), t1)


class AvailabilityTests(unittest.TestCase):
    def test_pixel_only_mode_suppresses_angular_claims_but_keeps_pixel_outputs(self):
        model = CameraModel(None)
        assert model.pixel_to_ray(10, 10).direction is None
        assert "pixel-only" in model.pixel_to_ray(10, 10).availability_reason["angular"]
        extent = model.bbox_angular_extent(BBox(1, 1, 5, 5))
        assert extent.width_rad is None and extent.width_deg is None and extent.availability_reason
        rate = model.angular_rate((1, 1), 0, (2, 2), 1)
        assert rate.rate_rad_s is None and rate.rate_deg_s is None and rate.availability_reason
        assert model.image_heading(5, 0).heading_deg == 0.0
        samples = [(0.1 * i, 20 * math.exp(0.5 * 0.1 * i)) for i in range(6)]
        assert model.apparent_size_change(samples, basis="pixel_width").label == "expanding"
        angular = model.apparent_size_change(samples, basis="angular_width")
        assert angular.label == "uncertain" and angular.log_rate_per_s is None and angular.availability_reason

    def test_zoom_change_and_camera_motion_suppress_angular_claims(self):
        model = CameraModel(Calibration.from_dict(PINHOLE))
        ok = FrameContext(zoom=1.0)
        assert model.pixel_to_ray(32, 24, context=ok).direction is not None
        assert model.pixel_to_ray(32, 24, context=FrameContext()).direction is not None   # zoom unknown: assumed
        for bad in (FrameContext(zoom=2.0), FrameContext(camera_moved=True)):
            ray = model.pixel_to_ray(32, 24, context=bad)
            assert ray.direction is None and ray.availability_reason
            assert model.bbox_angular_extent(BBox(20, 20, 30, 30), context=bad).width_rad is None
            assert model.angular_rate((32, 24), 0, (33, 24), 1, context1=bad).rate_rad_s is None

    def test_size_change_is_uncertain_when_zoom_or_camera_changes(self):
        samples = [(0.1 * i, 20 * math.exp(0.5 * 0.1 * i)) for i in range(8)]
        model = CameraModel(Calibration.from_dict(PINHOLE))
        assert model.apparent_size_change(samples, contexts=[FrameContext()] * 8).label == "expanding"
        for contexts in ([FrameContext(zoom=1.0)] * 7 + [FrameContext(zoom=1.5)],
                         [FrameContext(zoom=1.0), FrameContext(camera_moved=True)]):
            for basis in ("angular_width", "pixel_width"):
                result = model.apparent_size_change(samples, basis=basis, contexts=contexts)
                assert result.label == "uncertain" and result.availability_reason, (basis, contexts)
        # zoom change inside a window is caught even in pixel-only mode
        result = CameraModel(None).apparent_size_change(
            samples, basis="pixel_width", contexts=[FrameContext(zoom=1.0), FrameContext(zoom=2.0)])
        assert result.label == "uncertain"


class HeadingAndSizeChangeTests(unittest.TestCase):
    def setUp(self):
        self.model = CameraModel(Calibration.from_dict(PINHOLE))

    def test_heading_convention(self):
        cases = {(10, 0): 0.0, (0, -10): 90.0, (-10, 0): 180.0, (0, 10): 270.0, (10, -10): 45.0, (10, 10): 315.0}
        for (vx, vy), expected in cases.items():
            assert abs(self.model.image_heading(vx, vy).heading_deg - expected) < 1e-9, (vx, vy)
        assert 0 <= self.model.image_heading(-1e-9 + 5, -1e-12).heading_deg < 360
        assert "not aircraft yaw" in self.model.image_heading(5, 0).convention

    def test_near_zero_speed_has_undefined_heading_with_reason(self):
        result = self.model.image_heading(0.3, -0.4)
        assert result.heading_deg is None and abs(result.speed_px_s - 0.5) < 1e-12
        assert "undefined" in result.availability_reason["heading_deg"]
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(GeometryError):
                self.model.image_heading(bad, 0)

    def test_size_change_labels_do_not_imply_range_or_approach(self):
        def series(rate, n=8):
            return [(0.1 * i, 0.05 * math.exp(rate * 0.1 * i)) for i in range(n)]
        assert self.model.apparent_size_change(series(0.8)).label == "expanding"
        assert self.model.apparent_size_change(series(-0.8)).label == "shrinking"
        stable = self.model.apparent_size_change(series(0.0))
        assert stable.label == "stable" and abs(stable.log_rate_per_s) < 1e-9
        result = self.model.apparent_size_change(series(0.8))
        assert "not range" in result.interpretation and "approach" in result.interpretation
        assert result.label not in ("approaching", "receding")
        assert not any("range" in f.name or "yaw" in f.name or f.name.endswith("_m")
                       for f in dataclasses.fields(SizeChange))

    def test_noisy_or_sparse_size_series_is_uncertain(self):
        rng = np.random.default_rng(5)
        noisy = [(0.1 * i, 0.05 * math.exp(rng.normal(0, 0.4))) for i in range(8)]
        assert self.model.apparent_size_change(noisy).label == "uncertain"
        short = self.model.apparent_size_change([(0, 0.05), (0.1, 0.06), (0.2, 0.07)])
        assert short.label == "uncertain" and short.availability_reason
        brief = self.model.apparent_size_change([(0.01 * i, 0.05 + 0.001 * i) for i in range(6)])
        assert brief.label == "uncertain"   # span below size_min_span_s
        for bad in ([(0, 0.0)] * 4, [(0, 1), (0, 2), (1, 3), (2, 4)]):
            with self.assertRaises(GeometryError):
                self.model.apparent_size_change(bad)
        with self.assertRaises(GeometryError):
            self.model.apparent_size_change(noisy, basis="range")

    def test_model_parameters_validated(self):
        for kw in ({"min_speed_px_s": 0}, {"size_min_samples": 2}, {"size_stable_log_rate_per_s": -1}):
            with self.assertRaises(GeometryError):
                CameraModel(None, **kw)
        with self.assertRaises(GeometryError):
            CameraModel({"K": 1})


if __name__ == "__main__":
    unittest.main()
