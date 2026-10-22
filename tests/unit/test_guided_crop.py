"""AERIS-034 tracker-guided crops: placement, mapping, filtering and the tracker's extend-only pass."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from aeris.data.contracts import BBox, DetectionRecord
from aeris.detection import baseline
from aeris.detection.guided import GuidedCrops, crop_detections, crop_windows
from aeris.tracking.tracker import KalmanTracker, TrackerError, load_config

ROOT = Path(__file__).resolve().parents[2]
W, H = 1280, 720


def state(track_id, lifecycle, center, velocity=(0.0, 0.0), t=1.0, last=1.0):
    return SimpleNamespace(track_id=track_id, lifecycle=lifecycle, center=center, velocity_px_s=velocity,
                           timestamp_s=t, last_measurement_time_s=last)


def record(box, t=1.04, index=1, score=0.9, seq="s", width=W, height=H):
    return DetectionRecord(1, seq, index, t, width, height, BBox(*box), 0, "aeris-aircraft-v1", score, "test", {})


def run_crops(windows, boxes_by_window, frame=(), image=None):
    image = np.zeros((H, W, 3), np.uint8) if image is None else image
    calls = iter(boxes_by_window)
    return crop_detections(image, windows, lambda crop: next(calls), list(frame), sequence_id="s", frame_index=1,
                           timestamp_s=1.04, provenance="p", class_mapping_version="aeris-aircraft-v1",
                           nms=baseline.nms, nms_iou=0.45)


class WindowTests(unittest.TestCase):
    def test_only_confirmed_and_lost_tracks_most_recent_first_ties_by_id_and_capped(self):
        states = [state("t9", "tentative", (100, 100)), state("t2", "lost", (300, 300), last=0.5),
                  state("t1", "confirmed", (500, 300), last=1.0), state("t3", "confirmed", (700, 300), last=1.0),
                  state("t4", "confirmed", (900, 300), last=0.9), state("t5", "lost", (1000, 300), last=0.2)]
        ids = [w[0] for w in crop_windows(states, 1.04, W, H, 416)]
        self.assertEqual(ids, ["t1", "t3", "t4", "t2"])
        self.assertEqual(len(crop_windows(states, 1.04, W, H, 416, max_crops=2)), 2)

    def test_centre_is_propagated_from_the_last_state_and_window_clipped_into_frame(self):
        (_, x0, y0, w, h), = crop_windows([state("t1", "confirmed", (600, 300), velocity=(100, -50), t=1.0)],
                                          1.5, W, H, 416)
        self.assertEqual((w, h), (416, 416))
        self.assertEqual((x0 + w / 2, y0 + h / 2), (650, 275))
        (_, x0, y0, w, h), = crop_windows([state("t1", "confirmed", (5, 710))], 1.0, W, H, 416)
        self.assertEqual((x0, y0), (0, H - 416))
        (_, x0, y0, w, h), = crop_windows([state("t1", "confirmed", (50, 50))], 1.0, 300, 200, 416)
        self.assertEqual((x0, y0, w, h), (0, 0, 300, 200))

    def test_a_state_from_the_future_is_refused(self):
        with self.assertRaises(ValueError):
            crop_windows([state("t1", "confirmed", (50, 50), t=2.0)], 1.0, W, H, 416)


class CropDetectionTests(unittest.TestCase):
    def test_boxes_map_to_original_frame_pixels_with_full_frame_size(self):
        out = run_crops([("t1", 400, 200, 416, 416)], [[([100, 150, 120, 160], 0, 0.6)]])
        self.assertEqual([(d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max) for d in out],
                         [(500, 350, 520, 360)])
        self.assertEqual((out[0].width, out[0].height, out[0].score, out[0].model_provenance), (W, H, 0.6, "p"))

    def test_interior_edge_fragments_dropped_but_frame_border_kept(self):
        inner = run_crops([("t1", 400, 200, 416, 416)], [[([0, 150, 30, 160], 0, 0.6)]])
        border = run_crops([("t1", 0, 200, 416, 416)], [[([0, 150, 30, 160], 0, 0.6)]])
        self.assertEqual((len(inner), len(border)), (0, 1))

    def test_overlapping_windows_are_merged_and_frame_pass_duplicates_dropped(self):
        windows = [("t1", 400, 200, 416, 416), ("t2", 420, 200, 416, 416)]
        same = [[([100, 150, 120, 160], 0, 0.6)], [([80, 150, 100, 160], 0, 0.5)]]   # one object, both windows
        self.assertEqual(len(run_crops(windows, same)), 1)
        dup = run_crops([("t1", 400, 200, 416, 416)], [[([100, 150, 120, 160], 0, 0.6)]],
                        frame=[record((501, 350, 521, 360))])
        near = run_crops([("t1", 400, 200, 416, 416)], [[([100, 150, 120, 160], 0, 0.6)]],
                         frame=[record((515, 350, 535, 360))])
        self.assertEqual((len(dup), len(near)), (0, 1))

    def test_guided_crops_run_the_detector_on_native_resolution_windows(self):
        seen = []

        class Runner:
            name = "stub"
            def __call__(self, tensor):
                seen.append(tensor.shape)
                return "raw"

        stub_b = SimpleNamespace(
            preprocess=lambda crop, size, pad: (crop, 1.0), nms=baseline.nms,
            postprocess=lambda raw, ratio, w, h, cfg: [([100, 150, 120, 160], 0, 0.6, 4)])
        stages = SimpleNamespace(_b=stub_b, runner=Runner(), config={
            "input": {"size": 416, "pad_value": 114}, "postprocess": {"nms_iou": 0.45},
            "model": {"name": "m", "sha256": "0" * 64}, "class_mapping_version": "aeris-aircraft-v1"})
        guided = GuidedCrops(stages)
        image = np.zeros((H, W, 3), np.uint8)
        out, n = guided(image, [state("t1", "confirmed", (600, 400))], [], sequence_id="s", frame_index=1,
                        timestamp_s=1.04)
        self.assertEqual((n, guided.crops_run, seen), (1, 1, [(416, 416, 3)]))   # crop not downscaled
        self.assertEqual(out[0].bbox.x_min, 392 + 100)
        self.assertTrue(out[0].model_provenance.endswith("+guided-crop"))


def det(t, index, cx, cy, w=20, h=20, score=0.9):
    return record((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), t=t, index=index, score=score)


class ExtendOnlyTests(unittest.TestCase):
    def confirmed(self):
        tracker = KalmanTracker("s")
        for i in range(3):
            tracker.step(i * 0.04, [det(i * 0.04, i, 100 + i, 100)])
        return tracker

    def test_extend_only_never_starts_or_confirms_a_track(self):
        tracker = KalmanTracker("s")
        states = tracker.step(0.0, [], extend_only=[det(0.0, 0, 100, 100)])
        self.assertEqual((states, tracker.counters["births"]), ([], 0))
        tracker.step(0.04, [det(0.04, 1, 300, 300)])                       # tentative track
        states = tracker.step(0.08, [], extend_only=[det(0.08, 2, 300, 300)])
        self.assertEqual([s.observed for s in states], [False])
        self.assertEqual(tracker.counters["extend_only_matches"], 0)

    def test_extend_only_measures_a_confirmed_or_lost_track_unmatched_by_frame_detections(self):
        tracker = self.confirmed()
        tracker.step(0.12, [])                                              # now lost
        states = tracker.step(0.16, [], extend_only=[det(0.16, 4, 105, 100)])
        self.assertEqual([(s.lifecycle, s.observed, s.measurement_center_px) for s in states],
                         [("confirmed", True, (105.0, 100.0))])
        self.assertEqual(tracker.counters["extend_only_matches"], 1)

    def test_frame_detections_win_and_extend_only_is_not_used_twice(self):
        tracker = self.confirmed()
        states = tracker.step(0.12, [det(0.12, 3, 103, 100)], extend_only=[det(0.12, 3, 104, 101, score=0.99)])
        self.assertEqual(states[0].measurement_center_px, (103.0, 100.0))
        self.assertEqual(tracker.counters["extend_only_matches"], 0)

    def test_none_and_empty_extend_only_leave_states_unchanged(self):
        scene = [(i * 0.04, [det(i * 0.04, i, 100 + 5 * i, 100)] if i % 3 else []) for i in range(15)]

        def run(extra):
            tracker = KalmanTracker("s")
            return [[(s.track_id, s.center, s.covariance, s.lifecycle) for s in tracker.step(t, d, 0.3, **extra)]
                    for t, d in scene]
        self.assertEqual(run({}), run({"extend_only": None}))
        self.assertEqual(run({}), run({"extend_only": []}))

    def test_extend_only_records_are_checked_like_detections(self):
        tracker = KalmanTracker("s")
        with self.assertRaises(TrackerError):
            tracker.step(0.0, [], extend_only=[det(0.04, 0, 100, 100)])


class ConfigValidationTests(unittest.TestCase):
    """AERIS-033 review findings 1-3."""

    def test_empty_or_non_object_configs_are_rejected_not_replaced_by_the_baseline(self):
        for bad in ({}, [], "x"):
            with self.assertRaises(TrackerError):
                KalmanTracker("s", bad)

    def test_wrong_value_types_raise_tracker_error(self):
        for key, value in (("gate_chi2", "13"), ("accel_sigma_px_s2", None), ("confirm_hits", True)):
            with self.assertRaises(TrackerError, msg=key):
                KalmanTracker("s", {**load_config(), key: value})

    def test_position_gate_boundary_is_the_two_dof_threshold(self):
        cfg = {**load_config(), "gate_mode": "position", "gate_chi2_position": 9.21}

        def matched_at(d2):
            tracker = KalmanTracker("s", cfg)
            for i in range(4):
                tracker.step(i * 0.04, [det(i * 0.04, i, 100, 100)])
            track = tracker._tracks[0]
            f, q = tracker._transition(0.04, (track.x[4], track.x[5]))
            p = f @ track.p @ f.T + q
            s_xx = p[0, 0] + tracker._measurement_noise(20, 20)[0, 0]
            dx = float(np.sqrt(d2 * s_xx))
            states = tracker.step(0.16, [det(0.16, 4, float((f @ track.x)[0]) + dx, float((f @ track.x)[1]))])
            return tracker.counters["births"] == 1 and states[0].observed
        self.assertTrue(matched_at(9.0))
        self.assertFalse(matched_at(9.4))
        self.assertFalse(matched_at(13.0))      # would pass the 4-dof gate_chi2 (13.277) by mistake


if __name__ == "__main__":
    unittest.main()
