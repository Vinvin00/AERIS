"""Analytic invariants are asserted independently of the generator equations."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from aeris.data.contracts import AnnotationRecord, FrameRecord, validate_sequence
from aeris.simulation.fixtures import generate


ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "simulation"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class FixtureTests(unittest.TestCase):
    def test_fixed_bundle_and_all_hashes(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        with TemporaryDirectory() as tmp:
            generated = Path(tmp)
            self.assertEqual(generate(generated), manifest["sha256"])
            self.assertEqual((generated / "manifest.json").read_bytes(),
                             (ROOT / "manifest.json").read_bytes())
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
                self.assertEqual((ROOT / name).read_bytes(), (generated / name).read_bytes())
        self.assertIn("synthetic", manifest["scope"])
        self.assertEqual(manifest["seed"], 17)

    def test_motion_crossing_gaps_and_empty_frame(self):
        records = [AnnotationRecord.from_dict(row) for row in
                   jsonl(ROOT / "truth" / "annotations.jsonl")]
        validate_sequence(records)
        self.assertEqual([record.frame.timestamp_s for record in records],
                         [0.0, 0.1, 0.25, 0.5, 0.9])
        self.assertEqual([len(record.objects) for record in records], [2, 2, 1, 2, 0])
        self.assertTrue(records[-1].is_negative)
        self.assertEqual([o.track_id for o in records[2].objects], ["B"])
        expected = [
            {"A": (10.0, 20.0), "B": (30.0, 28.0)},
            {"A": (12.0, 20.8), "B": (28.0, 27.2)},
            {"B": (25.0, 26.0)},
            {"A": (20.0, 24.0), "B": (20.0, 24.0)},
            {},
        ]
        for record, centers in zip(records, expected):
            actual = {o.track_id: ((o.bbox.x_min + o.bbox.x_max) / 2,
                                   (o.bbox.y_min + o.bbox.y_max) / 2)
                      for o in record.objects}
            self.assertEqual(actual, centers)
        self.assertAlmostEqual((12.0 - 10.0) / 0.1, 20.0)
        self.assertAlmostEqual((20.0 - 12.0) / (0.5 - 0.1), 20.0)
        self.assertAlmostEqual((20.0 - 28.0) / (0.5 - 0.1), -20.0)

    def test_projection_independent_truth_and_runtime_boundary(self):
        camera = json.loads((ROOT / "input" / "camera.json").read_text())
        self.assertEqual(camera["K"], [[100.0, 0.0, 32.0], [0.0, 100.0, 24.0],
                                       [0.0, 0.0, 1.0]])
        self.assertEqual(camera["distortion_model"], "none")
        rows = jsonl(ROOT / "truth" / "projection.jsonl")
        self.assertEqual([row["world_xyz_m"] for row in rows],
                         [[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [-1.0, 0.0, 10.0]])
        self.assertEqual([row["projected_center_xy_px"] for row in rows],
                         [[32.0, 24.0], [42.0, 24.0], [22.0, 24.0]])
        self.assertEqual([row["bbox_xyxy_px"] for row in rows],
                         [[30.0, 22.0, 34.0, 26.0], [40.0, 22.0, 44.0, 26.0],
                          [20.0, 22.0, 24.0, 26.0]])
        for name in ("frames.jsonl", "projection_frames.jsonl"):
            for row in jsonl(ROOT / "input" / name):
                frame = FrameRecord.from_dict(row)
                image = (ROOT / "input" / frame.image_path).read_bytes()
                self.assertEqual(hashlib.sha256(image).hexdigest(), frame.image_checksum)
                self.assertEqual(len(image), len(b"P6\n64 48\n255\n") + 64 * 48 * 3)
                self.assertNotIn("objects", row)
                self.assertNotIn("world_xyz_m", row)
        self.assertFalse((ROOT / "input" / "truth").exists())
        self.assertEqual(set(path.name for path in (ROOT / "input").iterdir()),
                         {"camera.json", "frames.jsonl", "projection_frames.jsonl", "images"})


if __name__ == "__main__":
    unittest.main()
