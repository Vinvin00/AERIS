"""Pretrained detector baseline tests (AERIS-006).

Most tests feed hand-built raw head outputs whose decoded boxes are derived by hand in the
comments, so they check preprocessing, decoding, NMS, class mapping and the inverse
letterbox without any model. A tiny generated ONNX graph exercises the ONNX Runtime
wrapper. Tests marked REAL need the downloaded model and sample image and skip, with the
reason, when those are absent. None of this measures detector quality.
"""
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np

from aeris.data.contracts import DetectionRecord, FrameRecord
from aeris.detection import baseline as b
from aeris.detection import export

ROOT = Path(__file__).resolve().parents[2]
CONFIG = b.load_config()
S = b.settings_from_config(CONFIG)
AIRPLANE, BIRD = 4, 14  # COCO contiguous indices (category ids 5 and 16)


def anchor(stride: int, x: int, y: int) -> int:
    """Row of grid cell (x, y) at one stride: levels 80x80, 40x40, 20x20 in order."""
    offset = {8: 0, 16: 80 * 80, 32: 80 * 80 + 40 * 40}[stride]
    return offset + y * (640 // stride) + x


def raw_with(*rows, dtype=np.float32):
    """Raw output with the given (row, dx, dy, lw, lh, obj, {class: prob}) entries."""
    raw = np.zeros((S.anchors, S.num_classes + 5), dtype=dtype)
    for row, dx, dy, lw, lh, obj, probs in rows:
        raw[row, :5] = (dx, dy, lw, lh, obj)
        for cls, prob in probs.items():
            raw[row, 5 + cls] = prob
    return raw


def frame(width, height, timestamp=None):
    reasons = {"camera_id": "test frame"}
    if timestamp is None:
        reasons["timestamp_s"] = "still image"
    return FrameRecord(1, "det-test", 0, timestamp, "unavailable" if timestamp is None else "provided",
                       width, height, "test.png", "0" * 64, None, "unlabeled", reasons)


class ConfigTests(unittest.TestCase):
    def test_settings(self):
        self.assertEqual((S.input_height, S.input_width, S.pad_value), (640, 640, 114))
        self.assertEqual(S.anchors, 8400)
        self.assertEqual(S.num_classes, 80)
        self.assertEqual((S.score_threshold, S.nms_iou_threshold), (0.01, 0.65))
        self.assertEqual(S.class_map, {AIRPLANE: 0})

    def test_class_mapping_version_matches_dataset_manifest(self):
        manifest = json.loads((ROOT / "configs/data/source_manifest.json").read_text())
        self.assertEqual(S.class_mapping_version, manifest["class_mapping"]["version"])
        self.assertEqual(manifest["class_mapping"]["classes"], {"0": "aircraft"})

    def test_recorded_terms_and_hashes(self):
        model = CONFIG["model"]
        self.assertEqual(model["code"]["license"], "Apache-2.0")
        self.assertRegex(model["code"]["commit"], "^[0-9a-f]{40}$")
        for artifact in (model["weights"], model["upstream_onnx_cross_check"]):
            self.assertRegex(artifact["sha256"], "^[0-9a-f]{64}$")
            self.assertGreater(artifact["size_bytes"], 0)
        self.assertFalse(model["weights"]["upstream_reported"]["measured_by_aeris"])

    def test_download_budget_is_bounded(self):
        budget = CONFIG["download_budget"]
        self.assertTrue(budget["declared_before_fetching"])
        for kind in ("download", "installed"):
            total = budget[f"total_{kind}_upper_bound_bytes"]
            self.assertEqual(total, sum(i[f"{kind}_bytes"] for i in budget["items"]))
            self.assertLessEqual(total, budget["owner_limit_bytes"])

    def test_tolerances_declared(self):
        tol = CONFIG["parity_tolerances"]
        self.assertTrue(tol["declared_before_measurement"])
        for name in ("raw_regression_abs", "raw_probability_abs", "final_box_coordinate_abs_px",
                     "final_score_abs", "boundary_abs"):
            self.assertGreater(tol[name], 0)
        self.assertEqual(export.tolerances_from_config(CONFIG)["boundary_abs"], 0.001)

    def test_aeris_export_recorded(self):
        record = CONFIG["export"]["aeris_export"]
        self.assertRegex(record["sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(record["path"], str(export.AERIS_ONNX.relative_to(ROOT)))
        self.assertEqual(CONFIG["export"]["opset"], 17)

    def test_unimplemented_settings_are_refused(self):
        edits = [("input", "channel_order", "RGB"), ("postprocess", "class_selection", "multi-label"),
                 ("postprocess", "max_detections_per_frame", 100), ("postprocess", "clip_to_frame", False)]
        for section, key, value in edits:
            changed = json.loads(json.dumps(CONFIG))
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(b.DetectorError):
                b.settings_from_config(changed)
        changed = json.loads(json.dumps(CONFIG))
        changed["class_mapping"]["map"] = {"80": 0}
        with self.assertRaises(b.DetectorError):
            b.settings_from_config(changed)
        with self.assertRaises(b.DetectorError):
            b.settings_from_config({"input": {}})


def upstream_preproc(img, input_size, swap=(2, 0, 1)):
    """Verbatim copy of yolox/data/data_augment.py::preproc at commit 6ddff482
    (Apache-2.0, Megvii), kept here as the reference for byte equality."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class PreprocessTests(unittest.TestCase):
    def test_matches_upstream_bytes(self):
        rng = np.random.default_rng(6)
        for height, width in ((720, 1280), (576, 768), (333, 1000), (1080, 607), (200, 300)):
            image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            tensor, letterbox = b.preprocess(image, S)
            reference, ratio = upstream_preproc(image, (640, 640))
            with self.subTest(size=(height, width)):
                self.assertEqual(tensor.shape, (1, 3, 640, 640))
                self.assertEqual(tensor.dtype, np.float32)
                np.testing.assert_array_equal(tensor[0], reference)
                self.assertEqual((letterbox.resized_width, letterbox.resized_height),
                                 (int(width * ratio), int(height * ratio)))

    def test_pad_value_and_placement(self):
        tensor, letterbox = b.preprocess(np.zeros((720, 1280, 3), np.uint8), S)
        self.assertEqual((letterbox.resized_width, letterbox.resized_height), (640, 360))
        self.assertTrue((tensor[0, :, :360] == 0).all())
        self.assertTrue((tensor[0, :, 360:] == 114).all())

    def test_channel_order_is_kept_bgr(self):
        image = np.zeros((64, 64, 3), np.uint8)
        image[..., 0] = 10  # blue in OpenCV's BGR order
        tensor, _ = b.preprocess(image, S)
        self.assertTrue((tensor[0, 0] == 10).all() and (tensor[0, 1:] == 0).all())

    def test_rejects_other_inputs(self):
        for image in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32),
                      np.zeros((10, 10, 4), np.uint8), np.zeros((0, 10, 3), np.uint8),
                      np.zeros((1, 5000, 3), np.uint8), [[0]]):
            with self.assertRaises(b.DetectorError):
                b.preprocess(image, S)


class InverseTests(unittest.TestCase):
    def test_opencv_resize_maps_pixel_edges_linearly(self):
        # Each column holds its own center in edge coordinates (x + 0.5). If resize maps
        # edges as x_resized = x * 640 / 1000, resized column i samples (i + 0.5) * 1000/640.
        image = np.tile(np.arange(1000, dtype=np.float32) + 0.5, (4, 1))
        resized = cv2.resize(image, (640, 4), interpolation=cv2.INTER_LINEAR)
        expected = (np.arange(640) + 0.5) * 1000 / 640
        np.testing.assert_allclose(resized[0, 1:-1], expected[1:-1], atol=1e-3)

    def test_full_content_maps_to_full_frame(self):
        # 1000x333: r = 0.64, resized 640 x floor(213.12) = 213.
        _, letterbox = b.preprocess(np.zeros((333, 1000, 3), np.uint8), S)
        self.assertEqual((letterbox.resized_width, letterbox.resized_height), (640, 213))
        np.testing.assert_allclose(letterbox.to_original([[0, 0, 640, 213]]), [[0, 0, 1000, 333]],
                                   rtol=0, atol=1e-9)
        # Upstream's divide-by-r gives 213 / 0.64 = 332.8125 for the bottom edge.
        self.assertAlmostEqual(213 / 0.64, 332.8125)

    def test_scale_by_two(self):
        letterbox = b.Letterbox(1280, 720, 640, 360)
        np.testing.assert_array_equal(letterbox.to_original([[48, 28, 64, 44]]), [[96, 56, 128, 88]])
        self.assertEqual(letterbox.to_original(np.zeros((0, 4))).shape, (0, 4))


class DecodeTests(unittest.TestCase):
    def test_anchor_order_and_formula(self):
        # Stride 16, cell (3, 2): row 6400 + 2*40 + 3 = 6483. dx 0.5, dy 0.25, log sizes 0:
        # cx = 3.5*16 = 56, cy = 2.25*16 = 36, w = h = 16.
        row = anchor(16, 3, 2)
        self.assertEqual(row, 6483)
        raw = raw_with((row, 0.5, 0.25, 0.0, 0.0, 0.0, {}))
        np.testing.assert_array_equal(b.decode(raw, S)[row], [56, 36, 16, 16])
        np.testing.assert_array_equal(b.decode(raw[None], S, np.array([row])), [[56, 36, 16, 16]])
        self.assertEqual(anchor(32, 19, 19), 8399)
        np.testing.assert_array_equal(b.decode(raw, S)[8399], [(19 + 0) * 32, 19 * 32, 32, 32])

    def test_log_size(self):
        raw = raw_with((0, 0.0, 0.0, math.log(3), math.log(0.5), 0.0, {}), dtype=np.float64)
        np.testing.assert_allclose(b.decode(raw, S)[0], [0, 0, 24, 4])

    def test_rejects_bad_outputs(self):
        for bad in (np.zeros((2, S.anchors, 85)), np.zeros((S.anchors, 84)), np.zeros((100, 85))):
            with self.assertRaises(b.DetectorError):
                b.decode(bad, S)
        nan = raw_with((5, float("nan"), 0, 0, 0, 0, {}))
        with self.assertRaises(b.DetectorError):
            b.decode(nan, S)
        huge = raw_with((5, 0, 0, 1000.0, 0, 1.0, {AIRPLANE: 1.0}), dtype=np.float64)
        with self.assertRaises(b.DetectorError):
            b.candidates(huge, S)


def reference_nms(boxes, scores, threshold):
    """Brute-force greedy NMS in Python floats."""
    def iou(p, q):
        inter = max(0.0, min(p[2], q[2]) - max(p[0], q[0])) * max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / union if union > 0 else 0.0
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) <= threshold for k in keep):
            keep.append(i)
    return keep


class NmsTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(6)
        for trial in range(40):
            n = rng.randint(0, 30)
            boxes, scores = [], []
            for _ in range(n):
                x, y = rng.uniform(0, 50), rng.uniform(0, 50)
                boxes.append([x, y, x + rng.uniform(1, 30), y + rng.uniform(1, 30)])
                scores.append(rng.choice([0.5, 0.25, rng.random()]))  # include ties
            kept = b.nms(np.array(boxes).reshape(-1, 4), np.array(scores), 0.65)
            self.assertEqual(kept.tolist(), reference_nms(boxes, scores, 0.65), trial)

    def test_equal_iou_is_not_suppressed(self):
        boxes = np.array([[0, 0, 2, 1], [1, 0, 3, 1]], float)  # IoU = 1/3
        self.assertEqual(b.nms(boxes, np.array([0.9, 0.8]), 1 / 3).tolist(), [0, 1])
        self.assertEqual(b.nms(boxes, np.array([0.9, 0.8]), 0.33).tolist(), [0])

    def test_ties_keep_lower_index(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
        self.assertEqual(b.nms(boxes, np.array([0.5, 0.5]), 0.65).tolist(), [0])

    def test_per_class(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10]], float)
        kept = b.batched_nms(boxes, np.array([0.6, 0.9, 0.7]), np.array([4, 14, 4]), 0.65)
        self.assertEqual(kept.tolist(), [1, 2])  # IoU(0, 2) = 90/110 > 0.65
        self.assertEqual(b.batched_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0), 0.65).size, 0)


class PostprocessTests(unittest.TestCase):
    LETTERBOX = b.Letterbox(1280, 720, 640, 360)  # r = 0.5

    def test_airplane_maps_to_aeris_class_in_original_pixels(self):
        # Stride 16 cell (3, 2), sizes 16: input box [48, 28, 64, 44] -> original x2.
        raw = raw_with((anchor(16, 3, 2), 0.5, 0.25, 0.0, 0.0, 0.9, {AIRPLANE: 0.8}))
        (det,) = b.postprocess(raw, self.LETTERBOX, S)
        self.assertEqual(det.bbox, (96.0, 56.0, 128.0, 88.0))
        self.assertEqual((det.class_id, det.source_class, det.anchor), (0, AIRPLANE, 6483))
        self.assertAlmostEqual(det.score, 0.72, places=6)

    def test_other_classes_are_dropped_after_argmax(self):
        row = anchor(16, 3, 2)
        bird_wins = raw_with((row, 0.5, 0.5, 0.0, 0.0, 1.0, {AIRPLANE: 0.5, BIRD: 0.6}))
        self.assertEqual(b.postprocess(bird_wins, self.LETTERBOX, S), [])
        found = b.candidates(bird_wins, S)
        self.assertEqual(found.source_classes.tolist(), [BIRD])
        plane_wins = raw_with((row, 0.5, 0.5, 0.0, 0.0, 1.0, {AIRPLANE: 0.6, BIRD: 0.5}))
        self.assertEqual([d.class_id for d in b.postprocess(plane_wins, self.LETTERBOX, S)], [0])

    def test_score_threshold_is_inclusive(self):
        at = raw_with((0, 0.5, 0.5, 0.0, 0.0, 1.0, {AIRPLANE: 0.01}), dtype=np.float64)
        below = raw_with((0, 0.5, 0.5, 0.0, 0.0, 1.0, {AIRPLANE: 0.00999}), dtype=np.float64)
        self.assertEqual(len(b.postprocess(at, self.LETTERBOX, S)), 1)
        self.assertEqual(b.postprocess(below, self.LETTERBOX, S), [])
        self.assertEqual(b.postprocess(raw_with(), self.LETTERBOX, S), [])

    def test_nms_within_airplane_class_only(self):
        # Two stride-8 airplanes one cell apart with 32 px boxes: IoU = 24/40 * 1 = 0.6 < 0.65;
        # with 64 px boxes IoU = 56/72 > 0.65, so only the higher score survives.
        a, c = anchor(8, 10, 10), anchor(8, 11, 10)
        near = raw_with((a, 0, 0, math.log(4), math.log(4), 1.0, {AIRPLANE: 0.9}),
                        (c, 0, 0, math.log(4), math.log(4), 1.0, {AIRPLANE: 0.8}), dtype=np.float64)
        apart = raw_with((a, 0, 0, math.log(8), math.log(8), 1.0, {AIRPLANE: 0.9}),
                         (c, 0, 0, math.log(8), math.log(8), 1.0, {AIRPLANE: 0.8}), dtype=np.float64)
        self.assertEqual(len(b.postprocess(near, self.LETTERBOX, S)), 2)
        self.assertEqual([d.anchor for d in b.postprocess(apart, self.LETTERBOX, S)], [a])
        bird_over = raw_with((a, 0, 0, math.log(8), math.log(8), 1.0, {BIRD: 0.95}),
                             (c, 0, 0, math.log(8), math.log(8), 1.0, {AIRPLANE: 0.8}), dtype=np.float64)
        self.assertEqual([d.anchor for d in b.postprocess(bird_over, self.LETTERBOX, S)], [c])

    def test_clip_and_drop_padding(self):
        # Stride 32 cell (19, 0), dx = dy = 0.5, w = 64, h = 32: input [592, 0, 656, 32],
        # original [1184, 0, 1312, 64] clipped to x_max 1280.
        edge = raw_with((anchor(32, 19, 0), 0.5, 0.5, math.log(2), 0.0, 1.0, {AIRPLANE: 1.0}),
                        dtype=np.float64)
        (det,) = b.postprocess(edge, self.LETTERBOX, S)
        np.testing.assert_allclose(det.bbox, (1184, 0, 1280, 64))
        # Cell (5, 15): input y [480, 512] lies in the pad (content ends at 360), so the
        # clipped original box [720, 720] is empty and dropped.
        pad = raw_with((anchor(32, 5, 15), 0.5, 0.5, 0.0, 0.0, 1.0, {AIRPLANE: 1.0}))
        self.assertEqual(b.postprocess(pad, self.LETTERBOX, S), [])
        self.assertEqual(len(b.candidates(pad, S).scores), 1)

    def test_order_by_score(self):
        raw = raw_with((anchor(8, 1, 1), 0.5, 0.5, 0, 0, 1.0, {AIRPLANE: 0.3}),
                       (anchor(8, 40, 20), 0.5, 0.5, 0, 0, 1.0, {AIRPLANE: 0.7}),
                       (anchor(8, 70, 30), 0.5, 0.5, 0, 0, 1.0, {AIRPLANE: 0.5}))
        scores = [d.score for d in b.postprocess(raw, self.LETTERBOX, S)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(scores), 3)


class RecordTests(unittest.TestCase):
    def test_contract_records(self):
        raw = raw_with((anchor(16, 3, 2), 0.5, 0.25, 0.0, 0.0, 0.9, {AIRPLANE: 0.8}))
        letterbox = b.Letterbox(1280, 720, 640, 360)
        still = frame(1280, 720)
        (record,) = b.to_records(b.postprocess(raw, letterbox, S), still, letterbox, S, "test model")
        self.assertIsInstance(record, DetectionRecord)
        record.validate_frame(still)
        self.assertEqual((record.bbox.x_min, record.bbox.y_max), (96.0, 88.0))
        self.assertEqual(record.class_mapping_version, "aeris-aircraft-v1")
        self.assertEqual(dict(record.availability_reason), {"timestamp_s": "still image"})
        self.assertEqual(DetectionRecord.from_json(record.to_json()), record)
        timed = frame(1280, 720, timestamp=0.5)
        (record,) = b.to_records(b.postprocess(raw, letterbox, S), timed, letterbox, S, "test model")
        self.assertEqual((record.timestamp_s, dict(record.availability_reason)), (0.5, {}))
        with self.assertRaises(b.DetectorError):
            b.to_records([], frame(1000, 720), letterbox, S, "test model")


class CompareTests(unittest.TestCase):
    """PyTorch/ONNX comparison rules on hand-built raw outputs (no model)."""

    LETTERBOX = b.Letterbox(1280, 720, 640, 360)
    TOL = export.tolerances_from_config(CONFIG)
    PLANE = (anchor(16, 3, 2), 0.5, 0.25, 0.0, 0.0, 0.9, {AIRPLANE: 0.8})

    def compare(self, ref, test):
        return export.compare(ref, test, self.LETTERBOX, S, self.TOL)

    def test_identical_outputs_pass(self):
        raw = raw_with(self.PLANE)
        result = self.compare(raw, raw.copy())
        self.assertTrue(result["passed"])
        self.assertEqual((result["matched"], result["unmatched"]), (1, []))
        self.assertEqual(result["final_box_max_abs_px"], 0.0)

    def test_raw_differences_beyond_tolerance_fail(self):
        raw = raw_with(self.PLANE)
        shifted = raw.copy()
        shifted[100, 0] += 0.002  # regression channel, tolerance 0.001
        result = self.compare(raw, shifted)
        self.assertFalse(result["checks"]["raw_regression_abs"])
        self.assertFalse(result["passed"])
        probs = raw.copy()
        probs[100, 40] += 0.0002  # probability channel, tolerance 0.0001
        self.assertFalse(self.compare(raw, probs)["checks"]["raw_probability_abs"])

    def test_box_difference_in_original_pixels(self):
        # dx 0.5 -> 0.5005 moves the input box by 0.008 px, the original box by 0.016 px.
        ref = raw_with(self.PLANE, dtype=np.float64)
        test = raw_with((anchor(16, 3, 2), 0.5005, 0.25, 0.0, 0.0, 0.9, {AIRPLANE: 0.8}),
                        dtype=np.float64)
        result = self.compare(ref, test)
        self.assertAlmostEqual(result["final_box_max_abs_px"], 0.016, places=9)
        self.assertTrue(result["passed"])

    def test_extra_detection_at_score_threshold_is_exempt(self):
        # The bird scores 0.00995 (dropped) in the reference and 0.01 (kept, since >= 0.01) in
        # the test: a raw difference of 5e-5, and the kept score is on the threshold.
        extra = (anchor(8, 60, 30), 0.5, 0.5, 0, 0, 1.0)
        ref = raw_with(self.PLANE, extra + ({BIRD: 0.00995},), dtype=np.float64)
        test = raw_with(self.PLANE, extra + ({BIRD: 0.01},), dtype=np.float64)
        result = self.compare(ref, test)
        self.assertEqual([u["side"] for u in result["unmatched"]], ["test_only"])
        self.assertTrue(result["unmatched"][0]["exempt"])
        self.assertTrue(result["passed"])

    def test_extra_confident_detection_fails(self):
        ref = raw_with(self.PLANE, dtype=np.float64)
        test = raw_with(self.PLANE, (anchor(8, 60, 30), 0.5, 0.5, 0, 0, 1.0, {BIRD: 0.5}),
                        dtype=np.float64)
        result = self.compare(ref, test)
        self.assertFalse(result["unmatched"][0]["exempt"])
        self.assertFalse(result["checks"]["unmatched_all_exempt"])
        self.assertFalse(result["passed"])

    def test_nms_boundary_is_exempt(self):
        # Two 32 px airplane boxes one stride-8 cell apart. With the second box's dx at 0.3483
        # the centres are 6.7864 px apart and IoU = 25.2136 / 38.7864 = 0.65006 > 0.65, so
        # NMS removes it; at 0.3487, IoU = 0.64992 and it survives. Both are within 0.001 of
        # the NMS threshold.
        def pair(dx):
            return raw_with((anchor(8, 10, 10), 0.5, 0.5, math.log(4), math.log(4), 1.0, {AIRPLANE: 0.9}),
                            (anchor(8, 11, 10), dx, 0.5, math.log(4), math.log(4), 1.0, {AIRPLANE: 0.8}),
                            dtype=np.float64)
        result = self.compare(pair(0.3483), pair(0.3487))
        self.assertEqual((result["detections_reference"], result["detections_test"]), (1, 2))
        self.assertEqual([u["side"] for u in result["unmatched"]], ["test_only"])
        self.assertTrue(result["unmatched"][0]["exempt"])
        self.assertTrue(result["passed"])
        far = self.compare(pair(0.3483), pair(0.9))  # IoU 0.61: clearly kept, not a boundary
        self.assertFalse(far["unmatched"][0]["exempt"])


def constant_onnx(path, raw, output_shape=None):
    """A graph whose output is `raw` plus 0 * sum(images), so it depends on its input."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    raw = raw.astype(np.float32).reshape(1, *raw.shape[-2:])
    graph = helper.make_graph(
        [helper.make_node("ReduceSum", ["images"], ["total"], keepdims=0),
         helper.make_node("Mul", ["total", "zero"], ["nothing"]),
         helper.make_node("Add", ["raw", "nothing"], ["output"])],
        "constant", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape or list(raw.shape))],
        initializer=[numpy_helper.from_array(raw, "raw"),
                     numpy_helper.from_array(np.array(0, np.float32), "zero")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


class OnnxRuntimeTests(unittest.TestCase):
    def test_end_to_end_through_onnxruntime(self):
        raw = raw_with((anchor(16, 3, 2), 0.5, 0.25, 0.0, 0.0, 0.9, {AIRPLANE: 0.8}),
                       (anchor(8, 5, 5), 0.5, 0.5, 0.0, 0.0, 0.9, {BIRD: 0.8}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constant.onnx"
            constant_onnx(path, raw)
            model = b.OnnxModel(path, S)
            self.assertEqual(model.providers, ["CPUExecutionProvider"])
            result = b.detect(model, np.zeros((720, 1280, 3), np.uint8), S)
        np.testing.assert_array_equal(result.raw[0], raw)
        self.assertEqual([d.bbox for d in result.detections], [(96.0, 56.0, 128.0, 88.0)])
        self.assertEqual(sorted(result.candidates.source_classes.tolist()), [AIRPLANE, BIRD])

    def test_rejects_wrong_model_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.onnx"
            constant_onnx(path, np.zeros((100, 85)))
            with self.assertRaises(b.DetectorError):
                b.OnnxModel(path, S)

    def test_artifact_hash_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"abc")
            digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            self.assertEqual(b.require_artifact(path, digest), path)
            for sha in ("0" * 64, None):
                with self.assertRaises(b.DetectorError):
                    b.require_artifact(path, sha)
            with self.assertRaises(b.DetectorError):
                b.require_artifact(Path(tmp) / "missing.bin", digest)


UPSTREAM_ONNX = ROOT / CONFIG["model"]["upstream_onnx_cross_check"]["path"]
DOG = ROOT / "data/raw/detector_samples/yolox-dog.jpg"
DOG_SHA256 = "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866"


@unittest.skipUnless(UPSTREAM_ONNX.is_file(), f"REAL: {UPSTREAM_ONNX} not downloaded")
class RealUpstreamOnnxTests(unittest.TestCase):
    """REAL pretrained weights (upstream ONNX export) through the AERIS pipeline."""

    def test_upstream_export_loads_under_the_declared_io(self):
        path = b.require_artifact(UPSTREAM_ONNX, CONFIG["model"]["upstream_onnx_cross_check"]["sha256"])
        model = b.OnnxModel(path, S)
        result = b.detect(model, np.full((720, 1280, 3), 114, np.uint8), S)
        self.assertEqual(result.raw.shape, (1, 8400, 85))
        self.assertTrue(((result.raw[0, :, 4:] >= 0) & (result.raw[0, :, 4:] <= 1)).all())

    @unittest.skipUnless(DOG.is_file(), f"REAL: {DOG} not fetched (python -m aeris.detection.samples fetch)")
    def test_real_photo_without_aircraft(self):
        # The YOLOX demo photo (768x576) shows a dog, a bicycle and a pickup truck. The
        # pretrained model must find them in original-frame pixels, and none maps to aircraft.
        model = b.OnnxModel(UPSTREAM_ONNX, S)
        image = cv2.imread(str(b.require_artifact(DOG, DOG_SHA256)))
        result = b.detect(model, image, S)
        self.assertEqual((result.letterbox.resized_width, result.letterbox.resized_height), (640, 480))
        self.assertEqual(result.detections, [])
        found = result.candidates
        confident = {b.COCO_CLASSES[c] for c, s in zip(found.source_classes, found.scores) if s > 0.5}
        self.assertTrue({"dog", "bicycle", "truck"} <= confident, confident)
        self.assertNotIn(AIRPLANE, found.source_classes[found.scores > 0.05].tolist())
        dog = found.boxes_input[(found.source_classes == 16).argmax()]
        x0, y0, x1, y1 = result.letterbox.to_original(dog[None])[0]
        # The dog sits in the lower left quarter of the photo.
        self.assertTrue(x0 < 200 and x1 < 384 and y0 > 150 and y1 > 480, (x0, y0, x1, y1))


WEIGHTS = ROOT / CONFIG["model"]["weights"]["path"]
TORCH_READY = all(importlib.util.find_spec(m) for m in ("torch", "torchvision", "yolox"))


@unittest.skipUnless(TORCH_READY and WEIGHTS.is_file() and DOG.is_file(),
                     "REAL: needs torch, torchvision and YOLOX at the pinned commit "
                     "(requirements/detector-torch-linux-x86_64-py311.txt), the weights and dog.jpg")
class RealTorchReferenceTests(unittest.TestCase):
    """REAL pretrained YOLOX-S in PyTorch: upstream agreement, export and ONNX parity."""

    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)
        cls.model = export.TorchModel(CONFIG, S)
        cls.tensor, cls.letterbox = b.preprocess(cv2.imread(str(DOG)), S)
        cls.raw = cls.model(cls.tensor)

    def test_reference_is_the_pinned_upstream(self):
        versions = self.model.versions
        self.assertEqual(versions["yolox"]["commit"], CONFIG["model"]["code"]["commit"])
        self.assertEqual(versions["yolox"]["changed_files"], [])
        self.assertEqual(self.raw.shape, (1, 8400, 85))

    def test_numpy_decode_matches_upstream_decode(self):
        upstream = self.model(self.tensor, decode=True)[0]
        np.testing.assert_allclose(b.decode(self.raw, S), upstream[:, :4], rtol=1e-6, atol=1e-3)
        np.testing.assert_array_equal(upstream[:, 4:], self.raw[0, :, 4:])

    def test_postprocess_matches_upstream_postprocess(self):
        # yolox.utils.postprocess (torchvision batched_nms) on the upstream-decoded output.
        import torch
        from yolox.utils import postprocess

        decoded = torch.from_numpy(self.model(self.tensor, decode=True))
        (upstream,) = postprocess(decoded, S.num_classes, S.score_threshold, S.nms_iou_threshold)
        upstream = upstream.numpy()
        found = b.candidates(self.raw, S)
        self.assertEqual(len(found.scores), len(upstream))
        np.testing.assert_array_equal(found.source_classes, upstream[:, 6].astype(int))
        np.testing.assert_allclose(found.scores, upstream[:, 4] * upstream[:, 5], rtol=1e-6)
        np.testing.assert_allclose(found.boxes_input, upstream[:, :4], atol=1e-3)

    def test_real_photo_through_pytorch(self):
        found = b.candidates(self.raw, S)
        confident = {b.COCO_CLASSES[c] for c, s in zip(found.source_classes, found.scores) if s > 0.5}
        self.assertTrue({"dog", "bicycle", "truck"} <= confident, confident)
        self.assertEqual(b.map_candidates(found, self.letterbox, S), [])

    def test_export_matches_reference_on_fixed_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export.export_onnx(self.model, Path(tmp) / "aeris.onnx", CONFIG["export"]["opset"])
            digest = b.sha256_file(path)
            report = export.run_parity(CONFIG, S, self.model, {"aeris_export": b.OnnxModel(path, S)},
                                       export.parity_images(DOG))
        self.assertTrue(report["passed"]["aeris_export"], json.dumps(report, default=str)[:3000])
        dog = report["frames"][0]["results"]["aeris_export"]
        self.assertGreater(dog["matched"], 10)
        self.assertEqual(dog["matched"], dog["detections_reference"])
        recorded = CONFIG["export"]["aeris_export"]
        if self.model.versions["torch"] == recorded["produced_with"]["torch"]:
            self.assertEqual(digest, recorded["sha256"])


if __name__ == "__main__":
    unittest.main()
