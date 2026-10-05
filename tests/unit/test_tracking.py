"""Tracker tests on synthetic scenes. Oracle detections are derived from synthetic truth by
the tests themselves; nothing here measures real-world tracking performance."""
import json
import unittest
from pathlib import Path

import numpy as np

from aeris.data.contracts import (
    AnnotationRecord, BBox, DetectionRecord, FrameRecord, GroundTruthObject,
)
from aeris.evaluation.metrics import TrackObservation, tracking_metrics
from aeris.tracking.tracker import (
    CONFIRMED, LOST, TENTATIVE, KalmanTracker, TrackerError, load_config,
)

ROOT = Path(__file__).resolve().parents[2]
W, H = 400, 300


def box(cx, cy, w=20, h=20):
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def det(t, index, b, score=0.9, cls=0, seq="s"):
    return DetectionRecord(1, seq, index, t, W, H, b, cls, "aeris-aircraft-v1", score, "oracle-test", {})


def truth(t, index, tracks, seq="s"):
    frame = FrameRecord(1, seq, index, t, "provided", W, H, f"{index}.png", "0" * 64, None,
                        "complete", {"camera_id": "synthetic"})
    objects = tuple(GroundTruthObject(f"{k}-{index}", 0, b, k, False, False, False, {})
                    for k, b in tracks.items())
    return AnnotationRecord(1, frame, "aeris-aircraft-v1", objects)


def crossing(n=14, dt=0.1):
    """A moves right, B left; centers meet near frame 7 with a 8 px vertical offset."""
    out = []
    for i in range(n):
        t = round(i * dt, 6)
        out.append((t, {"A": box(50 + 200 * t, 100), "B": box(350 - 200 * t, 108)}))
    return out


def run(scene, drop=(), cfg=None):
    tracker = KalmanTracker("s", cfg)
    history = []
    for i, (t, objs) in enumerate(scene):
        detections = [det(t, i, b) for k, b in objs.items() if (i, k) not in drop]
        history.append(tracker.step(t, detections))
    return tracker, history


def ids_by_truth(history, scene, key):
    """Track ID nearest to a truth object in each frame where a confirmed track exists."""
    ids = []
    for states, (_, objs) in zip(history, scene):
        cx, cy = (objs[key].x_min + objs[key].x_max) / 2, (objs[key].y_min + objs[key].y_max) / 2
        near = [s for s in states if s.lifecycle != TENTATIVE
                and abs(s.center[0] - cx) < 12 and abs(s.center[1] - cy) < 12]
        ids.append(min(near, key=lambda s: abs(s.center[0] - cx) + abs(s.center[1] - cy)).track_id
                   if near else None)
    return ids


class ConfigTests(unittest.TestCase):
    def test_default_config_and_rejections(self):
        KalmanTracker("s")
        for key in ("gate_chi2", "max_coast_s", "tentative_timeout_s"):
            for bad in (0, -1, float("nan"), float("inf")):
                cfg = load_config()
                cfg[key] = bad
                with self.assertRaises(TrackerError):
                    KalmanTracker("s", cfg)
        cfg = load_config()
        del cfg["accel_sigma_px_s2"]
        with self.assertRaises(TrackerError):
            KalmanTracker("s", cfg)


class StabilityAndLifecycleTests(unittest.TestCase):
    def test_crossing_keeps_ids(self):
        scene = crossing()
        _, history = run(scene)
        a, b = ids_by_truth(history, scene, "A"), ids_by_truth(history, scene, "B")
        assert len(set(a[3:])) == 1 and len(set(b[3:])) == 1 and a[-1] != b[-1], (a, b)
        assert a[-1] is not None and b[-1] is not None

    def test_missing_detections_coast_then_recover_same_id(self):
        scene = crossing(n=16)
        drop = {(i, "A") for i in (6, 7, 8)}        # 0.3 s gap < max_coast_s, covers the crossing
        _, history = run(scene, drop)
        ids = ids_by_truth(history, scene, "A")
        assert len({i for i in ids if i}) == 1, ids
        lost = [s for s in history[7] if s.lifecycle == LOST]
        assert lost and lost[0].observed is False and abs(lost[0].time_since_seen_s - 0.2) < 1e-9
        assert history[-1][0].lifecycle == CONFIRMED or history[-1][1].lifecycle == CONFIRMED

    def test_timeout_is_in_seconds_not_frames(self):
        for dt, expect_alive_at_gap in ((0.1, True), (0.6, False)):
            tracker = KalmanTracker("s")
            t = 0.0
            for i in range(4):                       # confirm
                tracker.step(t, [det(t, i, box(100 + 20 * t, 100))])
                t = round(t + dt, 6)
            # two empty frames: 2*dt seconds without a measurement
            tracker.step(t, [])
            states = tracker.step(round(t + dt, 6), [])
            gap = states[0].time_since_seen_s if states else None
            assert bool(states) == expect_alive_at_gap, (dt, gap)
        # Large single gap kills a confirmed track regardless of frame count.
        tracker = KalmanTracker("s")
        for i in range(4):
            tracker.step(i * 0.1, [det(i * 0.1, i, box(100, 100))])
        assert tracker.step(5.0, []) == [] and tracker.counters["deleted"] == 1

    def test_tentative_requires_hits_and_times_out(self):
        tracker = KalmanTracker("s")
        s0 = tracker.step(0.0, [det(0.0, 0, box(100, 100))])
        assert s0[0].lifecycle == TENTATIVE
        s1 = tracker.step(0.1, [det(0.1, 1, box(102, 100))])
        s2 = tracker.step(0.2, [det(0.2, 2, box(104, 100))])
        assert s1[0].lifecycle == TENTATIVE and s2[0].lifecycle == CONFIRMED
        spur = KalmanTracker("s")
        spur.step(0.0, [det(0.0, 0, box(100, 100))])
        assert spur.step(0.7, []) == []              # 0.7 s > tentative_timeout_s

    def test_empty_detections_from_start(self):
        tracker = KalmanTracker("s")
        assert tracker.step(0.0, []) == [] and tracker.step(0.1, []) == []

    def test_duplicate_detections_do_not_spawn_extra_tracks(self):
        tracker = KalmanTracker("s")
        for i in range(5):
            t = i * 0.1
            states = tracker.step(t, [det(t, i, box(100 + 20 * t, 100), 0.9),
                                      det(t, i, box(100 + 20 * t + 0.5, 100), 0.5)])
        assert len(states) == 1 and tracker.counters["duplicates_dropped"] == 5

    def test_detection_overlapping_existing_track_does_not_birth(self):
        tracker = KalmanTracker("s")
        for i in range(4):
            t = i * 0.1
            tracker.step(t, [det(t, i, box(100, 100))])
        # Large box around the same spot gates out of the Kalman match yet overlaps heavily.
        states = tracker.step(0.4, [det(0.4, 4, box(100, 100)), det(0.4, 4, box(100, 100, 30, 26))])
        assert len(states) == 1

    def test_implausible_jump_is_gated_not_associated(self):
        tracker = KalmanTracker("s")
        for i in range(4):
            t = i * 0.1
            states = tracker.step(t, [det(t, i, box(100, 100))])
        first = states[0].track_id
        states = tracker.step(0.4, [det(0.4, 4, box(300, 250))])    # 200+ px in 0.1 s
        by_id = {s.track_id: s for s in states}
        assert by_id[first].lifecycle == LOST and len(by_id) == 2
        assert tracker.counters["gated_pairs"] == 1

    def test_classes_never_cross_associate(self):
        tracker = KalmanTracker("s")
        for i in range(4):
            t = i * 0.1
            tracker.step(t, [det(t, i, box(100, 100), cls=0)])
        states = tracker.step(0.4, [det(0.4, 4, box(100, 100), cls=1)])
        assert {s.class_id for s in states} == {0, 1} and len(states) == 2

    def test_ids_are_never_reused(self):
        tracker = KalmanTracker("s")
        seen = []
        for k in range(3):                           # birth, timeout, rebirth cycles
            base = k * 10.0
            seen += [s.track_id for s in tracker.step(base, [det(base, 0, box(100, 100))])]
        assert len(set(seen)) == 3


class NumericsAndCausalityTests(unittest.TestCase):
    def test_covariance_finite_psd_and_velocity_converges_with_variable_dt(self):
        # The default accel sigma (200 px/s^2) is deliberately loose; a constant-velocity
        # scene is tested with a matching tight motion model.
        tracker = KalmanTracker("s", {**load_config(), "accel_sigma_px_s2": 2.0})
        t, rng = 0.0, np.random.default_rng(3)
        last = None
        for i in range(60):
            dt = float(rng.choice([0.04, 0.1, 0.25, 0.5]))
            noise = [float(v) for v in rng.normal(0, 0.3, 2)]
            t = round(t + dt, 6) if i else 0.0
            states = tracker.step(t, [det(t, i, box(50 + 20 * t + noise[0], 100 - 6 * t + noise[1]))])
            p = np.array(states[0].covariance)
            assert np.all(np.isfinite(p)) and np.allclose(p, p.T)
            assert np.linalg.eigvalsh(p).min() > -1e-9
            last = states[0]
        assert abs(last.velocity_px_s[0] - 20) < 3 and abs(last.velocity_px_s[1] + 6) < 3

    def test_process_noise_scales_with_delta_time(self):
        tracker = KalmanTracker("s")
        a2 = tracker.cfg["accel_sigma_px_s2"] ** 2
        _, q1 = tracker._transition(0.1)
        _, q2 = tracker._transition(0.4)
        assert abs(q1[0, 0] - a2 * 0.1**4 / 4) < 1e-9 and abs(q2[2, 2] - a2 * 0.4**2) < 1e-6
        assert abs(q2[4, 4] / q1[4, 4] - 4) < 1e-9

    def test_longer_prediction_gap_grows_uncertainty(self):
        def trace_after(gap):
            tracker = KalmanTracker("s")
            for i in range(4):
                tracker.step(i * 0.1, [det(i * 0.1, i, box(100, 100))])
            return np.trace(np.array(tracker.step(0.3 + gap, [])[0].covariance))
        assert trace_after(0.1) < trace_after(0.4)

    def test_prefix_output_does_not_depend_on_future_frames(self):
        scene = crossing(n=14)
        _, full = run(scene)
        _, prefix = run(scene[:8])
        for a, b in zip(prefix, full[:8]):
            assert [(s.track_id, s.center, s.velocity_px_s, s.covariance) for s in a] == \
                   [(s.track_id, s.center, s.velocity_px_s, s.covariance) for s in b]

    def test_input_validation(self):
        tracker = KalmanTracker("s")
        tracker.step(0.0, [det(0.0, 0, box(100, 100))])
        for bad_t in (0.0, -1.0, float("nan"), True):
            with self.assertRaises(TrackerError):
                tracker.step(bad_t, [])
        with self.assertRaises(TrackerError):
            tracker.step(0.1, [det(0.2, 1, box(100, 100))])          # detection from another time
        with self.assertRaises(TrackerError):
            tracker.step(0.1, [det(0.1, 1, box(100, 100), seq="other")])
        with self.assertRaises(TrackerError):
            tracker.step(0.1, [DetectionRecord(1, "s", 1, 0.1, W + 1, H, box(100, 100), 0,
                                               "aeris-aircraft-v1", 0.9, "x", {})])

    def test_predicted_box_outside_frame_is_null_with_reason(self):
        tracker = KalmanTracker("s")
        for i in range(4):
            t = i * 0.1
            tracker.step(t, [det(t, i, box(300 + 100 * t, 100))])
        states = tracker.step(1.2, [])
        assert states[0].bbox is None and "bbox" in states[0].availability_reason


class TrackingMetricTests(unittest.TestCase):
    """AC3: evaluator runs on identity-labeled *synthetic* scenes; real claims stay blocked."""

    def observations(self, history):
        return [TrackObservation("s", i, s.track_id, s.bbox, s.class_id, s.observed)
                for i, states in enumerate(history) for s in states
                if s.lifecycle != TENTATIVE and s.bbox is not None]

    def test_synthetic_crossing_scores_through_trackeval(self):
        scene = crossing()
        _, history = run(scene)
        annotations = [truth(t, i, objs) for i, (t, objs) in enumerate(scene)]
        result = tracking_metrics(annotations, self.observations(history))
        assert result["available"] and result["backend"].startswith("trackeval")
        assert result["idf1"] > 0.8 and result["hota"] > 0.6, result
        _, gappy = run(scene, drop={(i, "A") for i in (6, 7, 8)})
        degraded = tracking_metrics(annotations, self.observations(gappy))
        assert degraded["predicted_only_positions_excluded"] > 0
        assert degraded["idf1"] <= result["idf1"]

    def test_unlabeled_identities_block_tracking_claims(self):
        scene = crossing(n=5)
        annotations = []
        for i, (t, objs) in enumerate(scene):
            record = truth(t, i, objs)
            objects = tuple(GroundTruthObject(o.annotation_id, 0, o.bbox, None, False, False, False,
                                              {"track_id": "no identity labels"}) for o in record.objects)
            annotations.append(AnnotationRecord(1, record.frame, "aeris-aircraft-v1", objects))
        _, history = run(scene)
        blocked = tracking_metrics(annotations, self.observations(history))
        assert blocked["available"] is False and blocked["hota"] is None and blocked["idf1"] is None

    def test_analytic_fixture_runs_through_tracker_and_evaluator(self):
        lines = (ROOT / "tests/fixtures/simulation/truth/annotations.jsonl").read_text().splitlines()
        annotations = [AnnotationRecord.from_dict(json.loads(line)) for line in lines]
        tracker = KalmanTracker("analytic-motion")
        history = []
        for a in annotations:        # oracle detections built from evaluator truth, for plumbing only
            dets = [DetectionRecord(1, a.frame.sequence_id, a.frame.frame_index, a.frame.timestamp_s,
                                    a.frame.width, a.frame.height, o.bbox, o.class_id,
                                    a.class_mapping_version, 0.9, "oracle-fixture", {})
                    for o in a.objects]
            history.append(tracker.step(a.frame.timestamp_s, dets))
        observations = [TrackObservation("analytic-motion", i, s.track_id, s.bbox, s.class_id, s.observed)
                        for i, states in enumerate(history) for s in states
                        if s.lifecycle != TENTATIVE and s.bbox is not None]
        result = tracking_metrics(annotations, observations)
        assert result["available"] and 0 <= result["hota"] <= 1 and 0 <= result["idf1"] <= 1


if __name__ == "__main__":
    unittest.main()
