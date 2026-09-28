"""Analytic fixture tests (AERIS-004). SYNTHETIC scope: they check math and plumbing only.

Expected numbers below are computed by hand or by independent implementations (OpenCV,
closed-form geometry), not by calling the generator's own helpers.
"""
from dataclasses import replace
import functools
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aeris.data.contracts import (
    PREDICTION_HORIZONS_S, AnnotationRecord, DetectionRecord, FrameRecord, validate_sequence,
)
from aeris.data.validate import validate_dataset
from aeris.simulation import fixtures
from aeris.simulation.fixtures import (
    DEFAULT_SEED, IMAGE_SCENARIOS, PROJECTION_TIMES, SCOPE, build_fixture_set,
    build_image_space, build_projection, write_fixture_set,
)

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "tests/fixtures/simulation"
TRUTH_KEYS = {"track_id", "objects", "ignore", "occluded", "truncated", "velocity_px_s",
              "center_scene", "markers_scene", "rays", "range_scene", "angular_extent_rad",
              "angular_rate_rad_s", "states", "forecast_targets", "source_track_id"}
NOISE_BOUND = 2 * math.sqrt(3.0) * 0.5


def jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode().splitlines()]


def keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(item) for item in value)) if value else set()
    return set()


def iou(a, b) -> float:
    width = min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"])
    height = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
    if width <= 0 or height <= 0:
        return 0.0
    area = lambda box: (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])  # noqa: E731
    return width * height / (area(a) + area(b) - width * height)


class FixtureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.manifest = build_fixture_set(DEFAULT_SEED)
        cls.expected = json.loads(cls.files["image_space/truth/expected.json"])
        cls.sequences = {s["sequence_id"]: s for s in cls.expected["sequences"]}
        cls.truth = [AnnotationRecord.from_dict(r)
                     for r in jsonl(cls.files["image_space/truth/annotations.jsonl"])]
        cls.detections = [DetectionRecord.from_dict(r)
                          for r in jsonl(cls.files["image_space/inputs/detections.jsonl"])]
        cls.frames = [FrameRecord.from_dict(r)
                      for r in jsonl(cls.files["image_space/inputs/frames.jsonl"])]
        cls.camera = json.loads(cls.files["projection/inputs/camera.json"])
        cls.observations = jsonl(cls.files["projection/inputs/observations.jsonl"])
        cls.projection = json.loads(cls.files["projection/truth/truth.json"])

    def truth_boxes(self, sequence_id, frame_index):
        record = next(r for r in self.truth
                      if (r.frame.sequence_id, r.frame.frame_index) == (sequence_id, frame_index))
        return {item.track_id: item.bbox.to_dict() for item in record.objects}

    def track(self, sequence_id, track_id):
        return next(t for t in self.sequences[sequence_id]["tracks"] if t["track_id"] == track_id)


class ImageSpaceAnalyticTests(FixtureTestCase):
    def test_constant_motion_matches_hand_computed_centres(self):
        # t1: (20 + 96 t, 40 + 32 t), 16x8; t2: (116 - 96 t, 72 - 32 t), 12x12; t3 fixed.
        self.assertEqual(self.truth_boxes("sim-cv-crossing", 3), {
            "t1": {"x_min": 27.0, "y_min": 41.0, "x_max": 43.0, "y_max": 49.0},
            "t2": {"x_min": 95.0, "y_min": 61.0, "x_max": 107.0, "y_max": 73.0},
            "t3": {"x_min": 131.0, "y_min": 15.0, "x_max": 141.0, "y_max": 25.0}})
        self.assertEqual(self.truth_boxes("sim-cv-crossing", 14)["t1"],
                         {"x_min": 108.0, "y_min": 68.0, "x_max": 124.0, "y_max": 76.0})
        for sequence in self.expected["sequences"]:
            for track in sequence["tracks"]:
                motion, states = track["motion"], track["states"]
                for state in states:
                    t = state["timestamp_s"]
                    self.assertEqual(state["center_px"], [motion["center_px"][k]
                                                          + motion["velocity_px_s"][k] * t
                                                          for k in range(2)])
                for before, after in zip(states, states[1:]):
                    dt = after["timestamp_s"] - before["timestamp_s"]
                    self.assertEqual([after["center_px"][k] - before["center_px"][k] for k in range(2)],
                                     [motion["velocity_px_s"][k] * dt for k in range(2)])

    def test_variable_delta_times_and_dropped_frames(self):
        sequence = self.sequences["sim-cv-crossing"]
        self.assertEqual(sequence["delta_t_s"], [0.0625, 0.0625, 0.03125, 0.09375, 0.0625, 0.1875,
                                                 0.0625, 0.0625, 0.125, 0.0625, 0.0625, 0.125])
        self.assertEqual(sequence["frame_index_gaps"], [
            {"after_frame_index": 5, "next_frame_index": 8, "missing_frame_indices": [6, 7],
             "delta_t_s": 0.1875}])
        frames = [f for f in self.frames if f.sequence_id == "sim-cv-crossing"]
        self.assertEqual([f.frame_index for f in frames], [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14])
        self.assertTrue(all(f.timestamp_source == "provided" for f in self.frames))
        for sequence_id in ("sim-cv-crossing", "sim-empty-birth"):
            validate_sequence([r for r in self.truth if r.frame.sequence_id == sequence_id])
        self.assertEqual(self.sequences["sim-empty-birth"]["delta_t_s"], [0.125] * 7)

    def test_crossing_time_point_and_overlap(self):
        (event,) = self.sequences["sim-cv-crossing"]["pair_events"]
        # Centres meet when 20 + 96 t = 116 - 96 t and 40 + 32 t = 72 - 32 t: t = 0.5 at (68, 56).
        self.assertEqual((event["type"], event["track_ids"], event["crossing_time_s"],
                          event["crossing_center_px"], event["crossing_frame_index"]),
                         ("crossing", ["t1", "t2"], 0.5, [68.0, 56.0], 8))
        # Boxes overlap while |96 - 192 t| < 14 and |32 - 64 t| < 10: 82/192 < t < 110/192.
        self.assertEqual(event["box_overlap_span_s"], [82 / 192, 110 / 192])
        # t = 0.5: [60,76]x[52,60] and [62,74]x[50,62] share 12x8 = 96 of 128 + 144 - 96.
        # t = 0.5625: [66,82]x[54,62] and [56,68]x[48,60] share 2x6 = 12 of 260.
        self.assertEqual([(f["frame_index"], f["iou"]) for f in event["overlapping_frames"]],
                         [(8, 96 / 176), (9, 12 / 260)])
        boxes = self.truth_boxes("sim-cv-crossing", 8)
        self.assertEqual(iou(boxes["t1"], boxes["t2"]), 96 / 176)

    def test_visibility_gap_and_complete_empty_frames(self):
        for frame_index in (11, 12):
            self.assertNotIn("t2", self.truth_boxes("sim-cv-crossing", frame_index))
        for frame_index in (10, 13):
            self.assertIn("t2", self.truth_boxes("sim-cv-crossing", frame_index))
        t2 = self.track("sim-cv-crossing", "t2")
        hidden = [s for s in t2["states"] if not s["visible"]]
        self.assertEqual([(s["frame_index"], s["center_px"]) for s in hidden],
                         [(11, [44.0, 48.0]), (12, [38.0, 46.0])])
        self.assertEqual([(s["first_frame_index"], s["last_frame_index"]) for s in t2["visible_spans"]],
                         [(0, 10), (13, 14)])
        empty = self.sequences["sim-empty-birth"]
        self.assertEqual(empty["empty_frame_indices"], [0, 1, 6, 7])
        self.assertEqual(self.sequences["sim-cv-crossing"]["empty_frame_indices"], [])
        for record in self.truth:
            if record.frame.sequence_id == "sim-empty-birth":
                self.assertEqual(record.frame.annotation_status, "complete")
                self.assertEqual(record.is_negative, record.frame.frame_index in (0, 1, 6, 7))
        self.assertEqual(sum(1 for d in self.detections
                             if d.sequence_id == "sim-empty-birth" and d.frame_index in (0, 1, 6, 7)), 0)

    def test_heading_speed_and_stationary_unavailability(self):
        t1, t2, t3 = (self.track("sim-cv-crossing", name) for name in ("t1", "t2", "t3"))
        # Heading is atan2(-vy, vx): moving right and down (vy > 0) points below image right.
        self.assertAlmostEqual(t1["heading_deg"], 360 - math.degrees(math.atan(32 / 96)), places=8)
        self.assertAlmostEqual(t2["heading_deg"], 180 - math.degrees(math.atan(32 / 96)), places=8)
        self.assertEqual(t1["speed_px_s"], math.sqrt(96 ** 2 + 32 ** 2))
        self.assertIsNone(t3["heading_deg"])
        self.assertIn("heading is undefined", t3["availability_reason"]["heading_deg"])

    def test_detector_noise_misses_duplicates_and_false_positives_are_explicit(self):
        sequence = self.sequences["sim-cv-crossing"]
        self.assertEqual(sequence["missed_detections"], [{"frame_index": 4, "track_id": "t1"}])
        kinds = {(row["frame_index"], row["kind"], row["source_track_id"])
                 for row in sequence["detections"] if row["kind"] != "true_positive"}
        self.assertEqual(kinds, {(2, "false_positive", None), (9, "duplicate", "t3")})
        by_frame = {}
        for detection in self.detections:
            by_frame.setdefault((detection.sequence_id, detection.frame_index), []).append(detection)
        rows = self.expected["sequences"]
        for sequence in rows:
            for row in sequence["detections"]:
                detection = by_frame[(sequence["sequence_id"], row["frame_index"])][row["detection_index"]]
                if row["kind"] == "true_positive":
                    truth = self.truth_boxes(sequence["sequence_id"], row["frame_index"])[row["source_track_id"]]
                    box = detection.bbox.to_dict()
                    self.assertTrue(all(abs(box[k] - truth[k]) <= NOISE_BOUND for k in box))
                    self.assertGreater(iou(box, truth), 0.5)
                    self.assertTrue(0.7 <= detection.score < 0.95)
        truth_frame_4 = self.truth_boxes("sim-cv-crossing", 4)["t1"]
        self.assertFalse(any(iou(d.bbox.to_dict(), truth_frame_4) > 0
                             for d in by_frame[("sim-cv-crossing", 4)]))
        false_positive = by_frame[("sim-cv-crossing", 2)]
        self.assertTrue(any(all(iou(d.bbox.to_dict(), b) == 0
                                for b in self.truth_boxes("sim-cv-crossing", 2).values())
                            for d in false_positive))
        for group in by_frame.values():
            self.assertEqual([d.bbox.x_min for d in group], sorted(d.bbox.x_min for d in group))
        self.assertEqual(len(self.detections), 37 + 4 - 1 + 1 + 1)

    def test_forecast_targets_follow_the_analytic_trajectory(self):
        for sequence in self.expected["sequences"]:
            tracks = {t["track_id"]: t for t in sequence["tracks"]}
            visible = sum(s["visible"] for t in sequence["tracks"] for s in t["states"])
            targets = sequence["forecast_targets"]
            self.assertEqual(len(targets), visible * len(PREDICTION_HORIZONS_S))
            last = sequence["frame_times"][-1]["timestamp_s"]
            for target in targets:
                motion = tracks[target["track_id"]]["motion"]
                self.assertIn(target["horizon_s"], PREDICTION_HORIZONS_S)
                self.assertEqual(target["target_time_s"], target["issue_time_s"] + target["horizon_s"])
                for k in range(2):
                    self.assertAlmostEqual(target["center_px"][k], motion["center_px"][k]
                                           + motion["velocity_px_s"][k] * target["target_time_s"],
                                           delta=1e-9)
                self.assertEqual(target["target_within_sequence"], target["target_time_s"] <= last)
        first = next(t for t in self.sequences["sim-cv-crossing"]["forecast_targets"]
                     if (t["track_id"], t["issue_frame_index"], t["horizon_s"]) == ("t1", 0, 0.25))
        self.assertEqual(first["center_px"], [44.0, 48.0])


class ProjectionTests(FixtureTestCase):
    def opencv_camera(self):
        i, d = self.camera["intrinsics"], self.camera["distortion"]
        # The fixture uses continuous pixels (top-left pixel centre 0.5); OpenCV uses 0.0.
        matrix = np.array([[i["fx"], 0, i["cx"] - 0.5], [0, i["fy"], i["cy"] - 0.5], [0, 0, 1]])
        return matrix, np.array([d["coefficients"][k] for k in d["order"]])

    def test_camera_parameters_are_complete(self):
        camera = self.camera
        self.assertEqual(camera["original_resolution"], [640, 480])
        i = camera["intrinsics"]
        self.assertTrue(i["fx"] > 0 and i["fy"] > 0 and 0 < i["cx"] < 640 and 0 < i["cy"] < 480)
        self.assertEqual((camera["distortion"]["model"], camera["distortion"]["order"]),
                         ("opencv_brown_conrady", ["k1", "k2", "p1", "p2", "k3"]))
        self.assertEqual(set(camera["distortion"]["coefficients"]), {"k1", "k2", "p1", "p2", "k3"})
        for key in ("camera_id", "calibration_id", "lens", "zoom", "pixel_convention", "camera_frame"):
            self.assertTrue(camera[key].strip(), key)
        self.assertIsNone(camera["quality"]["reprojection_error_px"])
        self.assertTrue(camera["quality"]["availability_reason"]["reprojection_error_px"].strip())
        self.assertEqual({o["camera_id"] for o in self.observations}, {camera["camera_id"]})

    def test_projection_matches_opencv(self):
        matrix, distortion = self.opencv_camera()
        worst = 0.0
        for frame, observation in zip(self.projection["frames"], self.observations):
            names = sorted(observation["points_px"])
            points = np.array([frame["markers_scene"][n] for n in names], float)
            projected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)
            fixture = np.array([observation["points_px"][n] for n in names]) - 0.5
            worst = max(worst, np.abs(projected.reshape(-1, 2) - fixture).max())
        grid = self.projection["round_trip_points"]
        rays = np.array([[*p["normalized"], 1.0] for p in grid])
        projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, distortion)
        worst = max(worst, np.abs(projected.reshape(-1, 2) - (np.array([p["pixel"] for p in grid]) - 0.5)).max())
        self.assertLess(worst, 1e-9)
        self.assertTrue(all(0 <= p["pixel"][0] < 640 and 0 <= p["pixel"][1] < 480 for p in grid))

    def test_undistortion_recovers_truth_rays(self):
        matrix, distortion = self.opencv_camera()
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-15)
        for frame, observation in zip(self.projection["frames"], self.observations):
            names = sorted(observation["points_px"])
            pixels = np.array([observation["points_px"][n] for n in names]) - 0.5
            normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, distortion,
                                             None, None, None, criteria).reshape(-1, 2)
            for name, point in zip(names, normalized):
                ray = frame["rays"][name]
                self.assertAlmostEqual(sum(v * v for v in ray), 1.0, places=12)
                np.testing.assert_allclose(point, [ray[0] / ray[2], ray[1] / ray[2]], atol=1e-9)
        for point in self.projection["round_trip_points"]:
            x, y = point["normalized"]
            np.testing.assert_allclose(point["ray"], np.array([x, y, 1.0]) / math.sqrt(x * x + y * y + 1),
                                       atol=1e-15)

    def test_angular_truth_matches_closed_forms(self):
        frames = self.projection["frames"]
        for frame in frames:
            left, right = (np.array(frame["markers_scene"][n]) for n in ("left", "right"))
            a, b, c = np.linalg.norm(left), np.linalg.norm(right), np.linalg.norm(left - right)
            self.assertAlmostEqual(frame["angular_extent_rad"]["horizontal"],
                                   math.acos((a * a + b * b - c * c) / (2 * a * b)), delta=1e-9)
        reference = self.projection["reference_on_axis"]
        self.assertAlmostEqual(reference["angular_extent_rad"],
                               2 * math.atan(reference["half_span"] / reference["depth"]), delta=1e-11)
        # Line-of-sight rate by central differences of the centre direction.
        start, velocity = np.array([-400.0, -120.0, 3000.0]), np.array([150.0, 20.0, -400.0])
        step = 1e-4
        for frame in frames:
            t = frame["timestamp_s"]
            before, after = start + velocity * (t - step), start + velocity * (t + step)
            rate = math.atan2(np.linalg.norm(np.cross(before, after)), before @ after) / (2 * step)
            self.assertAlmostEqual(frame["angular_rate_rad_s"] / rate, 1.0, delta=1e-6)
        extents = [f["angular_extent_rad"]["horizontal"] for f in frames]
        self.assertEqual(extents, sorted(extents))
        self.assertTrue(all(f["range_rate_scene_per_s"] < 0 for f in frames))

    def test_bbox_and_model_input_transform(self):
        for observation in self.observations:
            box, points = observation["bbox"], observation["points_px"].values()
            self.assertEqual(box, {"x_min": min(p[0] for p in points), "y_min": min(p[1] for p in points),
                                   "x_max": max(p[0] for p in points), "y_max": max(p[1] for p in points)})
            self.assertTrue(0 <= box["x_min"] < box["x_max"] <= 640 and 0 <= box["y_min"] < box["y_max"] <= 480)
        transform = self.camera["model_input_transform"]
        self.assertEqual((transform["crop_xywh"], transform["resize_wh"]), ([64, 48, 512, 384], [256, 192]))
        checks = self.projection["transform_checks"]
        self.assertEqual([c["model_input_px"] for c in checks[:3]], [[0.0, 0.0], [256.0, 192.0], [128.0, 96.0]])
        for check in checks:
            u, v = check["original_px"]
            self.assertEqual(check["model_input_px"], [(u - 64) * 256 / 512, (v - 48) * 192 / 384])


class TruthSeparationTests(FixtureTestCase):
    def test_inputs_hold_no_truth_fields(self):
        for entry in self.manifest["artifacts"]:
            path, role = entry["path"], entry["role"]
            self.assertEqual(role, "evaluator_truth" if "/truth/" in path else "perception_input")
            if role == "perception_input" and not path.endswith(".pgm"):
                data = self.files[path]
                values = jsonl(data) if path.endswith(".jsonl") else [json.loads(data)]
                self.assertFalse(keys(values) & TRUTH_KEYS, path)
        self.assertEqual({tuple(sorted(o)) for o in self.observations}, {(
            "bbox", "camera_id", "frame_index", "points_px", "schema_version", "sequence_id",
            "timestamp_s")})
        self.assertTrue(self.expected["evaluator_only"] and self.projection["evaluator_only"])
        # Input frames are the same records the evaluator's annotations carry.
        self.assertEqual(self.frames, [record.frame for record in self.truth])

    def test_inputs_never_depend_on_later_frames(self):
        full = build_image_space(DEFAULT_SEED)
        for cut in (1, 4, 7, 10):
            truncated = tuple(replace(s, frames=s.frames[:cut]) for s in IMAGE_SCENARIOS)
            prefix = build_image_space(DEFAULT_SEED, truncated)
            for name in ("frames.jsonl", "detections.jsonl"):
                path = f"image_space/inputs/{name}"
                kept = {(r["sequence_id"], r["frame_index"]) for r in jsonl(prefix["image_space/inputs/frames.jsonl"])}
                expected = [r for r in jsonl(full[path]) if (r["sequence_id"], r["frame_index"]) in kept]
                self.assertEqual(jsonl(prefix[path]), expected, (cut, name))
            for path, data in prefix.items():
                if path.endswith(".pgm"):
                    self.assertEqual(data, full[path])
        observations = jsonl(build_projection()["projection/inputs/observations.jsonl"])
        for cut in (1, 3, 6):
            prefix = jsonl(build_projection(PROJECTION_TIMES[:cut])["projection/inputs/observations.jsonl"])
            self.assertEqual(prefix, observations[:cut])


class ReproducibilityTests(FixtureTestCase):
    def test_committed_fixtures_match_regeneration(self):
        committed = {p.relative_to(COMMITTED).as_posix() for p in COMMITTED.rglob("*") if p.is_file()}
        listed = {a["path"] for a in self.manifest["artifacts"] if a["committed"]}
        self.assertEqual(committed, listed | {"manifest.json"})
        self.assertFalse(any(path.endswith(".pgm") for path in committed))
        for path in listed:
            self.assertEqual((COMMITTED / path).read_bytes(), self.files[path], path)
        self.assertEqual(json.loads((COMMITTED / "manifest.json").read_text()), self.manifest)

    def test_materialized_images_match_manifest_and_validate(self):
        temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp)
        write_fixture_set(temp, images=True)
        for entry in self.manifest["artifacts"]:
            data = (temp / entry["path"]).read_bytes()
            self.assertEqual((hashlib.sha256(data).hexdigest(), len(data)), (entry["sha256"], entry["bytes"]))
        for frame in self.frames:
            image = temp / "image_space/inputs/images" / frame.image_path
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), frame.image_checksum)
            self.assertEqual(cv2.imread(str(image), cv2.IMREAD_GRAYSCALE).shape, (frame.height, frame.width))
        report = validate_dataset(temp / "image_space/truth")
        self.assertEqual((report["problems"], report["duplicate_images_within_split"]), ([], 0))
        stats = report["stats"]["synthetic"]
        self.assertEqual((stats["frames"], stats["negative_frames"], stats["objects"], stats["tracks"]),
                         (21, 4, 41, 4))
        self.assertEqual(stats["width_bins"], {"[0,16)": 28, "[16,32)": 13, "[32,64)": 0, "[64,inf)": 0})
        # Rendered pixels follow the truth box (pixel centres inside the half-open box).
        image = cv2.imread(str(temp / "image_space/inputs/images/sim-cv-crossing_0003.pgm"),
                           cv2.IMREAD_GRAYSCALE)
        self.assertTrue(np.all(image[41:49, 27:43] == 60))
        self.assertTrue(np.all(image[40, 27:43] != 60) and np.all(image[41:49, 26] != 60))
        # b1 at t = 0.25 spans y 74.5-81.5: rows 74-80 have centres inside, row 81 does not.
        image = cv2.imread(str(temp / "image_space/inputs/images/sim-empty-birth_0002.pgm"),
                           cv2.IMREAD_GRAYSCALE)
        self.assertTrue(np.all(image[74:81, 39:53] == 60))
        self.assertFalse(np.any(image[[73, 81], 39:53] == 60))
        # Hidden t2 at t = 0.75 (box 38-50 x 42-54) is not drawn.
        image = cv2.imread(str(temp / "image_space/inputs/images/sim-cv-crossing_0011.pgm"),
                           cv2.IMREAD_GRAYSCALE)
        self.assertFalse(np.any(image[42:54, 38:50] == 60))

    def test_seed_changes_only_detector_noise(self):
        other, manifest = build_fixture_set(DEFAULT_SEED + 1)
        changed = sorted(path for path in self.files if other[path] != self.files[path])
        self.assertIn("image_space/inputs/detections.jsonl", changed)
        self.assertTrue(set(changed) <= {"image_space/inputs/detections.jsonl",
                                         "image_space/truth/expected.json"}, changed)
        # Noise can reorder near-coincident boxes, which renumbers expected detection rows only.
        strip = lambda data: [{k: v for k, v in s.items() if k != "detections"}  # noqa: E731
                              for s in json.loads(data)["sequences"]]
        self.assertEqual(strip(other["image_space/truth/expected.json"]), strip(
            self.files["image_space/truth/expected.json"]))
        for seed, files in ((DEFAULT_SEED, self.files), (DEFAULT_SEED + 1, other)):
            rows = [(s["sequence_id"], r["frame_index"], r["kind"], r["source_track_id"])
                    for s in json.loads(files["image_space/truth/expected.json"])["sequences"]
                    for r in s["detections"]]
            self.assertEqual(len(rows), 42, seed)
        self.assertEqual(manifest["seed"], DEFAULT_SEED + 1)
        self.assertEqual(build_fixture_set(DEFAULT_SEED)[0], self.files)

    def test_output_does_not_depend_on_float_summation_order(self):
        # Python 3.12 changed built-in sum() over floats, which once made the committed
        # bytes differ between CI's 3.11 and 3.12 jobs. Reversing the order sum() adds in
        # must not change a single byte.
        def reversed_sum(values, start=0):
            return functools.reduce(operator.add, reversed(list(values)), start)

        with patch.object(fixtures, "sum", reversed_sum, create=True):
            files, manifest = build_fixture_set(DEFAULT_SEED)
        self.assertEqual((files, manifest), (self.files, self.manifest))

    def test_synthetic_scope_is_labeled(self):
        self.assertIn("SYNTHETIC", SCOPE)
        for value in (self.manifest["scope"], self.expected["scope"], self.camera["scope"],
                      self.projection["scope"]):
            self.assertEqual(value, SCOPE)
        self.assertIn(SCOPE, json.loads(self.files["image_space/truth/dataset.json"])["group_basis"])
        self.assertTrue(all(d.model_provenance.startswith("SYNTHETIC") for d in self.detections))
        self.assertTrue(all(f.sequence_id.startswith("sim-") for f in self.frames))
        self.assertEqual(self.manifest["seed"], DEFAULT_SEED)

    def test_cli_writes_the_fixture_set(self):
        temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp)
        result = subprocess.run([sys.executable, "-m", "aeris.simulation.fixtures", str(temp)],
                                capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((temp / "manifest.json").read_bytes(), (COMMITTED / "manifest.json").read_bytes())
        self.assertFalse(list(temp.rglob("*.pgm")))


if __name__ == "__main__":
    unittest.main()
