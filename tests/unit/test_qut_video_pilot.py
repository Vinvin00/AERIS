"""QUT pilot alignment and evaluator-only association guards, using no source media."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from qut_video_pilot import evaluate, interpolate_label, load_labels, unique_near  # noqa: E402


class QutVideoPilotTests(unittest.TestCase):
    def test_global_frame_mapping_requires_original_frame_bounds_and_last_frame(self):
        with TemporaryDirectory() as directory:
            csv = Path(directory) / "labels.csv"
            csv.write_text(",frame,u,v,range\n0,1002,12,15,120\n1,1003,13,16,119\n")
            self.assertEqual(
                load_labels(csv, frame_count=4, width=40, height=30,
                            first_global_frame=1000, last_global_frame=1003),
                {2: (12.0, 15.0), 3: (13.0, 16.0)})
            with self.assertRaisesRegex(ValueError, "frame count"):
                load_labels(csv, frame_count=5, width=40, height=30,
                            first_global_frame=1000, last_global_frame=1003)
            csv.write_text(",frame,u,v,range\n0,1002,41,15,120\n1,1003,13,16,119\n")
            with self.assertRaisesRegex(ValueError, "coordinates"):
                load_labels(csv, frame_count=4, width=40, height=30,
                            first_global_frame=1000, last_global_frame=1003)

    def test_unique_association_rejects_missing_and_ambiguous_candidates(self):
        self.assertEqual(unique_near((10, 10), [(11, 10), (100, 100)], radius_px=5), 0)
        self.assertIsNone(unique_near((10, 10), [(100, 100)], radius_px=5))
        self.assertIsNone(unique_near((10, 10), [(11, 10), (12, 10)], radius_px=5))

    def test_interpolation_refuses_missing_label_gaps_and_extrapolation(self):
        self.assertEqual(interpolate_label(0.02, [0.0, 0.04], [(10, 20), (12, 24)]),
                         (11.0, 22.0))
        self.assertIsNone(interpolate_label(0.02, [0.0, 0.08], [(10, 20), (12, 24)]))
        self.assertIsNone(interpolate_label(0.1, [0.0, 0.04], [(10, 20), (12, 24)]))

    def test_evaluation_only_scores_unique_observed_target_and_future_label(self):
        box = SimpleNamespace(x_min=9.0, x_max=11.0, y_min=19.0, y_max=21.0)
        prediction = SimpleNamespace(track_id="1", center=(12.0, 20.0),
                                     target_time_s=0.1, model_name="constant_velocity",
                                     horizon_s=0.1)
        runtime = {
            "frames": [{"frame_index": i, "timestamp_s": i * 0.04} for i in range(4)],
            "detections": defaultdict(list, {0: [SimpleNamespace(bbox=box)]}),
            "observed": defaultdict(list, {0: [("1", (10.0, 20.0))]}),
            "predictions": defaultdict(list, {0: [prediction]})}
        labels = {0: (10.0, 20.0), 1: (11.0, 20.0),
                  2: (12.0, 20.0), 3: (13.0, 20.0)}
        result = evaluate(runtime, labels)
        self.assertEqual(result["detector_centroid_error"]["support"], 1)
        self.assertEqual(result["forecast_centroid_error"]["constant_velocity"]["0.1"], {
            "support": 1, "mean_px": 0.5, "median_px": 0.5, "max_px": 0.5,
            "availability_reason": None})
        runtime["observed"][0].append(("2", (10.0, 21.0)))
        result = evaluate(runtime, labels)
        self.assertEqual(result["associated_forecast_issue_frames"], 0)
        self.assertEqual(result["forecast_centroid_error"]["constant_velocity"]["0.1"]
                         ["support"], 0)
