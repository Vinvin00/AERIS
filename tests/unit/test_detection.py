"""Baseline detector: synthetic pre/post-processing tests plus local real-model checks.

Real-model tests skip when the (Git-ignored) weights or FGVC sample are absent. They
verify behaviour on three bounded images only, not detection quality.
"""
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

from aeris.data.validate import CLASS_MAPPINGS
from aeris.detection import baseline as b

ROOT = Path(__file__).resolve().parents[2]
CFG = b.load_config()
IMAGES = ROOT / "data" / "raw" / "g1-bounded-sample" / "images"
MODEL = ROOT / CFG["model"]["path"]
HAVE_MODEL = MODEL.is_file()
HAVE_IMAGES = (IMAGES / "1143132.jpg").is_file()


def raw_with(cell, coco_index, cx, cy, w, h, score=0.9):
    """Raw head output with one confident box; inverse of decode_head for that cell."""
    grid, stride = b._grids(416)
    raw = np.zeros((1, len(grid), 85), dtype=np.float32)
    raw[0, cell, 0:2] = np.array([cx, cy]) / stride[cell] - grid[cell]
    raw[0, cell, 2:4] = np.log(np.array([w, h]) / stride[cell])
    raw[0, cell, 4] = 1.0
    raw[0, cell, 5 + coco_index] = score
    return raw


class SyntheticTests(unittest.TestCase):
    def test_config_is_consistent(self):
        assert len(CFG["model"]["sha256"]) == 64
        assert CFG["class_mapping_version"] in CLASS_MAPPINGS
        for entry in CFG["class_map"].values():
            assert entry["aeris_class_id"] in CLASS_MAPPINGS[CFG["class_mapping_version"]]
        assert b._grids(CFG["input"]["size"])[0].shape[0] == CFG["output"]["shape"][1]

    def test_letterbox_places_image_top_left_and_pads(self):
        image = np.full((100, 200, 3), 7, np.uint8)
        tensor, ratio = b.preprocess(image)
        assert tensor.shape == (1, 3, 416, 416) and tensor.dtype == np.float32
        assert ratio == 2.08
        assert float(tensor[0, 0, 0, 0]) == 7 and float(tensor[0, 0, 415, 415]) == 114
        assert float(tensor[0, 0, 207, 0]) == 7 and float(tensor[0, 0, 209, 0]) == 114

    def test_boxes_invert_letterbox_into_original_pixels(self):
        # 100x200 image -> ratio 2.08. Model-space box (208,104)-(416,208) is original (100,50)-(200,100).
        raw = raw_with(cell=100, coco_index=4, cx=312, cy=156, w=208, h=104)
        result = b.postprocess(raw, 2.08, 200, 100, CFG)
        assert len(result) == 1
        box, class_id, score, coco = result[0]
        assert coco == 4 and class_id == 0 and abs(score - 0.9) < 1e-6
        np.testing.assert_allclose(box, [100, 50, 200, 100], atol=1e-3)

    def test_wrong_class_dropped_and_clipped_to_frame(self):
        assert b.postprocess(raw_with(100, 0, 100, 100, 50, 50), 1.0, 416, 416, CFG) == []
        box = b.postprocess(raw_with(100, 4, 290, 290, 100, 100), 1.0, 300, 300, CFG)[0][0]
        np.testing.assert_allclose(box, [240, 240, 300, 300], atol=1e-3)

    def test_box_fully_outside_frame_is_dropped_not_degenerate(self):
        assert b.postprocess(raw_with(100, 4, 400, 400, 20, 20), 1.0, 300, 300, CFG) == []

    def test_low_scores_and_empty_output(self):
        low = raw_with(100, 4, 100, 100, 50, 50, score=CFG["postprocess"]["score_floor"] / 2)
        assert b.postprocess(low, 1.0, 416, 416, CFG) == []
        assert b.postprocess(np.zeros((1, 3549, 85), np.float32), 1.0, 416, 416, CFG) == []

    def test_nms_suppresses_overlap_and_keeps_distinct(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
        assert b.nms(boxes, np.array([0.6, 0.9, 0.5]), 0.45) == [1, 2]

    def test_bad_inputs_rejected(self):
        with self.assertRaises(b.DetectorError):
            b.preprocess(np.zeros((10, 10), np.uint8))
        with self.assertRaises(b.DetectorError):
            b.preprocess(np.zeros((10, 10, 3), np.float32))
        with self.assertRaises(b.DetectorError):
            b.decode_head(np.zeros((1, 10, 85), np.float32))

    def test_model_hash_mismatch_is_rejected(self):
        bad = json.loads(json.dumps(CFG))
        bad["model"]["sha256"] = "0" * 64
        if HAVE_MODEL:
            with self.assertRaises(b.DetectorError):
                b.verified_model_path(bad)
        with self.assertRaises(b.DetectorError):
            b.verified_model_path({"model": {"path": "models/onnx/missing.onnx", "sha256": "0" * 64}})

    def test_detect_returns_valid_contract_records(self):
        class Fake:
            name = "fake"
            def __call__(self, tensor):
                return raw_with(100, 4, 312, 156, 208, 104)
        records = b.detect(np.zeros((100, 200, 3), np.uint8), Fake(), CFG, sequence_id="s", frame_index=0)
        assert len(records) == 1 and records[0].class_mapping_version == "aeris-aircraft-v1"
        assert records[0].availability_reason == {"timestamp_s": "still image"}
        assert (records[0].bbox.x_min, records[0].bbox.x_max) == (100.0, 200.0)


@unittest.skipUnless(HAVE_MODEL and HAVE_IMAGES, "local weights/FGVC sample absent")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = b.verified_model_path(CFG)
        cls.runners = [b.OnnxRuntimeRunner(path), b.OpenCvDnnRunner(path)]

    @staticmethod
    def read(name):
        return cv2.imread(str(IMAGES / f"{name}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    @staticmethod
    def iou(a, c):
        iw = max(0, min(a[2], c[2]) - max(a[0], c[0]))
        ih = max(0, min(a[3], c[3]) - max(a[1], c[1]))
        return iw * ih / ((a[2]-a[0])*(a[3]-a[1]) + (c[2]-c[0])*(c[3]-c[1]) - iw*ih)

    def test_real_aircraft_maps_to_class_zero_and_matches_label_box(self):
        # Labels (x_min,y_min,x_max,y_max) from data/processed/g1-bounded-sample/annotations.jsonl
        truth = {"1143132": (3, 136, 991, 478)}
        for name, box in truth.items():
            for runner in self.runners:
                found = b.detect(self.read(name), runner, CFG, sequence_id=name, frame_index=0)
                assert found and found[0].class_id == 0
                got = found[0].bbox
                assert self.iou((got.x_min, got.y_min, got.x_max, got.y_max), box) > 0.85

    def test_cabin_interior_has_no_aircraft_detection(self):
        for runner in self.runners:
            assert b.detect(self.read("1885942"), runner, CFG, sequence_id="c", frame_index=0) == []

    def test_runtimes_agree_within_declared_tolerance(self):
        report = b.agreement_report([IMAGES / f"{n}.jpg" for n in ("1885942", "1143132", "1627560")],
                                    CFG, timing_runs=2)
        assert report["all_within_tolerance"], report["images"]

    def test_recorded_agreement_fixture_matches_pinned_model_and_tolerances(self):
        recorded = json.loads((ROOT / "tests/fixtures/detection/runtime_agreement.json").read_text())
        assert recorded["model_sha256"] == CFG["model"]["sha256"]
        numeric = {k: v for k, v in CFG["agreement"].items() if k != "note"}
        assert {k: recorded["tolerances"][k] for k in numeric} == numeric and recorded["all_within_tolerance"]


if __name__ == "__main__":
    sys.exit(unittest.main())
