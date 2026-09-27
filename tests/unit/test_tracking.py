"""Kalman tracking and gated Hungarian association tests (AERIS-009).

Unit tests use SYNTHETIC inputs: the AERIS-004 analytic image-space fixtures (crossing,
missed, duplicate and false-positive detections against known constant-velocity ground
truth) and hand-built detection streams for the Kalman math itself. Expected values for
the fixture-driven tests are derived from the AERIS-004 scenario design (see its own
comments), not by re-running the tracker's own helpers.

Tests marked REAL need 23 pinned frames of one real, identity-labelled AOT flight
(tests/fixtures/tracking/real_samples.json) and, for the end-to-end one, the AERIS-006
ONNX export; they skip with the reason when those are absent, and
AERIS_REQUIRE_REAL_TRACKING=1 turns the skips into failures. None of this measures
detector or tracker quality beyond what is stated.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest

import numpy as np

from aeris.data import aot
from aeris.data.contracts import BBox, DetectionRecord, FrameRecord
from aeris.evaluation.metrics import TrackerBox, evaluate_tracking
from aeris.tracking import tracker as tr

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "tests/fixtures/simulation/image_space"
REAL_PATH = ROOT / "tests/fixtures/tracking/real_samples.json"
STRICT = os.environ.get("AERIS_REQUIRE_REAL_TRACKING") == "1"


def real(ready: bool, reason: str):
    """Skip (or, under AERIS_REQUIRE_REAL_TRACKING=1, fail) when REAL inputs are missing."""
    if ready:
        return
    if STRICT:
        raise AssertionError(f"REAL evidence required but unavailable: {reason}")
    raise unittest.SkipTest(f"REAL: {reason}")


def config(**overrides) -> tr.TrackerConfig:
    base = tr.config_from_dict(tr.load_config())
    return base.__class__(**{**base.__dict__, **overrides})


def det(sequence_id, frame_index, timestamp_s, box, score=0.9, width=200, height=150):
    return DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height, box, 0,
                           "aeris-aircraft-v1", score, "test", {})


class ConfigTests(unittest.TestCase):
    def test_gate_chi2_matches_chi_square_quantile(self):
        from scipy.stats import chi2
        c = config()
        self.assertAlmostEqual(c.gate_chi2, chi2.ppf(0.99, 4))

    def test_config_loads_from_the_repository_file(self):
        loaded = config()
        self.assertEqual(loaded.min_hits_to_confirm, 2)
        self.assertGreater(loaded.lost_timeout_s, 0)


class KalmanMathTests(unittest.TestCase):
    """Deterministic (noiseless) checks of predict/update, independent of the fixtures."""

    def test_constant_velocity_prediction_is_exact_with_variable_dt(self):
        # A single track observed once, then only predicted (no further detections) over
        # two very different gaps: position must advance by v * dt exactly (no noise, no
        # detector score involved), demonstrating the model uses elapsed time, not frame
        # count, and never looks past the current call ("no future access").
        c = config(lost_timeout_s=100.0)
        t = tr.Tracker("seq", c)
        t.step(0.0, [det("seq", 0, 0.0, BBox(0, 0, 10, 10))])   # centre (5,5)
        s1 = t.step(1.0, [])[0]     # dt = 1 s, no detection: pure prediction
        s2 = t.step(11.0, [])[0]    # dt = 10 s: a much larger, still variable, gap
        # No velocity is known yet after only one detection (v=0 prior), so the exact
        # predicted centre after any predict-only step is unchanged from the last update.
        self.assertEqual(s1.center, (5.0, 5.0))
        self.assertEqual(s2.center, (5.0, 5.0))
        self.assertAlmostEqual(s1.age_s, 1.0)
        self.assertAlmostEqual(s2.age_s, 11.0)
        self.assertAlmostEqual(s1.time_since_seen_s, 1.0)
        self.assertAlmostEqual(s2.time_since_seen_s, 11.0)

    def test_process_noise_grows_with_elapsed_time(self):
        # Two tracks from the same first detection, predicted once over a short and a long
        # gap: the longer gap must inject strictly more velocity-variance process noise,
        # not merely stay positive -- a Q that stopped depending on dt (e.g. a constant)
        # would not show this growth.
        c = config(lost_timeout_s=100.0)
        short, long_ = tr.Tracker("a", c), tr.Tracker("b", c)
        short.step(0.0, [det("a", 0, 0.0, BBox(0, 0, 10, 10))])
        long_.step(0.0, [det("b", 0, 0.0, BBox(0, 0, 10, 10))])
        s_short = short.step(0.1, [])[0]
        s_long = long_.step(10.0, [])[0]
        self.assertGreater(np.array(s_long.covariance)[2, 2], np.array(s_short.covariance)[2, 2])
        self.assertGreater(np.array(s_long.covariance)[4, 4], np.array(s_short.covariance)[4, 4])

    def test_covariance_stays_finite_symmetric_and_positive_semidefinite(self):
        c = config(lost_timeout_s=100.0)
        t = tr.Tracker("seq", c)
        rng = np.random.default_rng(20260927)
        timestamp = 0.0
        for _ in range(30):
            timestamp += float(rng.uniform(0.01, 0.5))
            x0, x1 = sorted(float(v) for v in rng.uniform(0, 50, 2))
            y0, y1 = sorted(float(v) for v in rng.uniform(50, 100, 2))
            box = BBox(x0, y0, x1, y1)
            for state in t.step(timestamp, [det("seq", 0, timestamp, box)]):
                P = np.array(state.covariance)
                self.assertTrue(np.isfinite(P).all())
                self.assertTrue(np.allclose(P, P.T))
                self.assertGreaterEqual(np.linalg.eigvalsh(P).min(), -1e-9)

    def test_symmetrize_psd_clips_float_noise_but_rejects_a_real_negative_eigenvalue(self):
        noisy = np.array([[1.0, 1e-12], [-1e-12, 1.0]])  # asymmetric by float noise only
        fixed = tr._symmetrize_psd(noisy)
        self.assertTrue(np.allclose(fixed, np.eye(2)))
        bad = np.array([[1.0, 0.0], [0.0, -1.0]])  # a genuine negative eigenvalue
        with self.assertRaisesRegex(tr.TrackingError, "eigenvalue"):
            tr._symmetrize_psd(bad)


class EmptyInputTests(unittest.TestCase):
    def test_no_detections_and_no_tracks_is_a_no_op(self):
        t = tr.Tracker("seq", config())
        self.assertEqual(t.step(0.0, []), [])
        self.assertEqual(t.step(0.1, []), [])

    def test_first_detections_spawn_tentative_tracks_with_no_existing_tracks(self):
        t = tr.Tracker("seq", config())
        states = t.step(0.0, [det("seq", 0, 0.0, BBox(0, 0, 10, 10)),
                              det("seq", 0, 0.0, BBox(50, 50, 60, 60))])
        self.assertEqual({s.lifecycle for s in states}, {"tentative"})
        self.assertEqual(len({s.track_id for s in states}), 2)

    def test_existing_tracks_survive_an_empty_frame_within_the_timeout(self):
        c = config(lost_timeout_s=1.0)
        t = tr.Tracker("seq", c)
        t.step(0.0, [det("seq", 0, 0.0, BBox(0, 0, 10, 10))])
        t.step(0.1, [det("seq", 1, 0.1, BBox(0, 0, 10, 10))])
        states = t.step(0.2, [])
        self.assertEqual(len(states), 1)
        self.assertFalse(states[0].observed)


class LifecycleTests(unittest.TestCase):
    def test_min_hits_to_confirm_is_honoured_exactly(self):
        # A track must stay tentative through every hit below the configured threshold,
        # and become confirmed on exactly the hit that reaches it -- not one before.
        c = config(min_hits_to_confirm=3, lost_timeout_s=10.0)
        t = tr.Tracker("seq", c)
        s0 = t.step(0.0, [det("seq", 0, 0.0, BBox(0, 0, 10, 10))])[0]
        s1 = t.step(0.1, [det("seq", 1, 0.1, BBox(0, 0, 10, 10))])[0]
        s2 = t.step(0.2, [det("seq", 2, 0.2, BBox(0, 0, 10, 10))])[0]
        self.assertEqual((s0.lifecycle, s1.lifecycle, s2.lifecycle),
                         ("tentative", "tentative", "confirmed"))
        self.assertEqual({s0.track_id, s1.track_id, s2.track_id}, {s0.track_id})


class ValidationTests(unittest.TestCase):
    def test_timestamp_must_strictly_increase(self):
        t = tr.Tracker("seq", config())
        t.step(1.0, [])
        with self.assertRaisesRegex(tr.TrackingError, "must strictly increase"):
            t.step(1.0, [])
        with self.assertRaisesRegex(tr.TrackingError, "must strictly increase"):
            t.step(0.5, [])

    def test_detection_sequence_id_must_match_the_tracker(self):
        t = tr.Tracker("seq-a", config())
        with self.assertRaisesRegex(tr.TrackingError, "sequence_id"):
            t.step(0.0, [det("seq-b", 0, 0.0, BBox(0, 0, 10, 10))])

    def test_detection_timestamp_must_match_the_step(self):
        t = tr.Tracker("seq", config())
        with self.assertRaisesRegex(tr.TrackingError, "timestamp_s"):
            t.step(0.0, [det("seq", 0, 0.1, BBox(0, 0, 10, 10))])

    def test_nonfinite_timestamp_is_refused(self):
        t = tr.Tracker("seq", config())
        with self.assertRaisesRegex(tr.TrackingError, "finite"):
            t.step(float("nan"), [])

    def test_sequence_id_must_be_a_nonempty_string(self):
        with self.assertRaisesRegex(tr.TrackingError, "sequence_id"):
            tr.Tracker("", config())


class DuplicateSuppressionTests(unittest.TestCase):
    def test_the_lower_scoring_of_two_overlapping_boxes_is_dropped(self):
        high = det("seq", 0, 0.0, BBox(0, 0, 10, 10), score=0.9)
        low = det("seq", 0, 0.0, BBox(1, 1, 11, 11), score=0.5)  # IoU with `high` ~ 0.68
        kept = tr._suppress_duplicates([low, high], 0.5)
        self.assertEqual(kept, [high])

    def test_non_overlapping_boxes_are_both_kept(self):
        a = det("seq", 0, 0.0, BBox(0, 0, 10, 10), score=0.9)
        b = det("seq", 0, 0.0, BBox(50, 50, 60, 60), score=0.5)
        self.assertEqual(tr._suppress_duplicates([a, b], 0.5), [a, b])

    def test_below_threshold_overlap_keeps_both(self):
        a = det("seq", 0, 0.0, BBox(0, 0, 10, 10), score=0.9)
        b = det("seq", 0, 0.0, BBox(6, 0, 16, 10), score=0.5)  # IoU = 4/16 = 0.25
        self.assertEqual(tr._suppress_duplicates([a, b], 0.5), [a, b])

    def test_step_itself_applies_duplicate_suppression(self):
        # Not just the helper function in isolation: step() must call it, so two heavily
        # overlapping detections in one frame spawn exactly one track, not two.
        t = tr.Tracker("seq", config())
        high = det("seq", 0, 0.0, BBox(0, 0, 10, 10), score=0.9)
        low = det("seq", 0, 0.0, BBox(1, 1, 11, 11), score=0.5)  # IoU ~ 0.68 > 0.5
        states = t.step(0.0, [low, high])
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].center, (5.0, 5.0))  # spawned from the higher-scoring box


# ------------------------------------------------------------------ AERIS-004 fixtures (AC1)

def _load_sim():
    frames = [FrameRecord.from_json(line)
             for line in (SIM / "inputs/frames.jsonl").read_text().splitlines()]
    detections = [DetectionRecord.from_json(line)
                  for line in (SIM / "inputs/detections.jsonl").read_text().splitlines()]
    expected = json.loads((SIM / "truth/expected.json").read_text())
    return frames, detections, expected


FRAMES, DETECTIONS, EXPECTED = _load_sim()
CROSSING, EMPTY_BIRTH = "sim-cv-crossing", "sim-empty-birth"


def run_sequence(sequence_id: str, lost_timeout_s: float = 0.3):
    """Feed one AERIS-004 image-space scenario through a fresh Tracker frame by frame.

    Returns (per_frame: [(frame_index, [TrackState])], id_by_source: {source_track_id:
    tracker_track_id} for every ID that was ever confirmed, built from the first frame at
    which a confirmed state's detection-order position matches a uniquely-identifiable
    source track -- concretely, by nearest centre to the expected ground-truth state).
    """
    seq_frames = sorted((f for f in FRAMES if f.sequence_id == sequence_id), key=lambda f: f.frame_index)
    by_frame = {}
    for d in DETECTIONS:
        if d.sequence_id == sequence_id:
            by_frame.setdefault(d.frame_index, []).append(d)
    expected_seq = next(s for s in EXPECTED["sequences"] if s["sequence_id"] == sequence_id)
    truth_track_at = {}  # (frame_index) -> {track_id: (cx, cy)} for every visible true track
    for track in expected_seq["tracks"]:
        for row in track["states"]:
            if row["visible"]:
                truth_track_at.setdefault(row["frame_index"], {})[track["track_id"]] = tuple(row["center_px"])

    t = tr.Tracker(sequence_id, config(lost_timeout_s=lost_timeout_s))
    per_frame = []
    id_by_source: dict[str, str] = {}
    for f in seq_frames:
        states = t.step(f.timestamp_s, by_frame.get(f.frame_index, []))
        per_frame.append((f.frame_index, states))
        truths = truth_track_at.get(f.frame_index, {})
        for state in states:
            if state.lifecycle != "confirmed":
                continue
            # Map by nearest true centre (< 5 px, well inside jitter+one motion step): this
            # is bookkeeping for the test's own assertions, not part of the tracker.
            nearest = min(truths.items(), key=lambda kv: math.dist(kv[1], state.center), default=None)
            if nearest is not None and math.dist(nearest[1], state.center) < 5.0:
                id_by_source.setdefault(nearest[0], state.track_id)
    return per_frame, id_by_source


class ImageSpaceCrossingTests(unittest.TestCase):
    """sim-cv-crossing: t1 x t2 cross paths at t=0.5s; t1 misses frame 4; t3 (stationary)
    gets a duplicate detection at frame 9; a false positive appears at frame 2."""

    def test_crossing_tracks_keep_separate_stable_ids(self):
        per_frame, id_by_source = run_sequence(CROSSING)
        self.assertEqual(set(id_by_source), {"t1", "t2", "t3"})
        self.assertEqual(len(set(id_by_source.values())), 3)  # no two true tracks share an ID
        expected_seq = next(s for s in EXPECTED["sequences"] if s["sequence_id"] == CROSSING)
        truth_track_at = {}
        for track in expected_seq["tracks"]:
            for row in track["states"]:
                if row["visible"]:
                    truth_track_at.setdefault(row["frame_index"], {})[track["track_id"]] = tuple(row["center_px"])
        by_frame = dict(per_frame)
        # Just after the crossing (frame 8 at t=0.5s is the exact instant t1 and t2's true
        # centres coincide, so "closer to its own" is undefined there) every one of
        # t1/t2/t3's tracker IDs must sit closest to its OWN source's true centre, never
        # the other crossing track's: no identity swap happened.
        for frame_index in (9, 10, 13, 14):
            for source, track_id in id_by_source.items():
                state = next((s for s in by_frame[frame_index] if s.track_id == track_id), None)
                if state is None or state.lifecycle != "confirmed":
                    continue
                own_distance = math.dist(truth_track_at[frame_index][source], state.center)
                other_distances = [math.dist(centre, state.center) for other, centre in
                                   truth_track_at[frame_index].items() if other != source]
                self.assertTrue(all(own_distance < other for other in other_distances),
                                f"frame {frame_index}: {track_id} (source {source}) is closer to "
                                "another true track than its own")

    def test_a_single_missed_detection_keeps_the_same_id(self):
        # t1 is not detected at frame 4 (still confirmed elsewhere); it must go "lost", not
        # be deleted or replaced, and resume as "confirmed" under the same ID at frame 5.
        per_frame, id_by_source = run_sequence(CROSSING)
        by_frame = dict(per_frame)
        t1_id = id_by_source["t1"]
        frame4 = {s.track_id: s for s in by_frame[4]}
        frame5 = {s.track_id: s for s in by_frame[5]}
        self.assertIn(t1_id, frame4)
        self.assertEqual(frame4[t1_id].lifecycle, "lost")
        self.assertFalse(frame4[t1_id].observed)
        self.assertIn(t1_id, frame5)
        self.assertEqual(frame5[t1_id].lifecycle, "confirmed")
        self.assertTrue(frame5[t1_id].observed)

    def test_the_duplicate_and_the_false_positive_never_reach_confirmed(self):
        # Frame 2 has one false positive; frame 9 has one duplicate detection of t3. Each
        # can spawn a spurious tentative track, but neither may ever be promoted to
        # "confirmed": that would mean a duplicate or a false positive corrupted the
        # confirmed-track output the way t3's own continuous identity must not be.
        per_frame, id_by_source = run_sequence(CROSSING)
        confirmed_ever = {s.track_id for _, states in per_frame for s in states if s.lifecycle == "confirmed"}
        self.assertEqual(confirmed_ever, set(id_by_source.values()))

    def test_t3_stays_continuously_confirmed_through_the_duplicate_frame(self):
        # The duplicate at frame 9 must not cost t3 its own continuity: it stays matched
        # (never "lost") at every frame from the duplicate onward in this scenario.
        per_frame, id_by_source = run_sequence(CROSSING)
        t3_id = id_by_source["t3"]
        by_frame = dict(per_frame)
        for frame_index in (9, 10, 13, 14):
            state = next(s for s in by_frame[frame_index] if s.track_id == t3_id)
            self.assertEqual(state.lifecycle, "confirmed")
            self.assertTrue(state.observed)

    def test_final_positions_match_the_analytic_ground_truth(self):
        per_frame, id_by_source = run_sequence(CROSSING)
        last_frame_index, last_states = per_frame[-1]
        self.assertEqual(last_frame_index, 14)
        by_id = {s.track_id: s for s in last_states}
        # t1: (20,40) + (96,32)*1.0 = (116,40+32)=(116,72); t2: (116,72)+(-96,-32)*1.0=(20,40).
        self.assertAlmostEqual(by_id[id_by_source["t1"]].center[0], 116.0, delta=1.0)
        self.assertAlmostEqual(by_id[id_by_source["t1"]].center[1], 72.0, delta=1.0)
        self.assertAlmostEqual(by_id[id_by_source["t2"]].center[0], 20.0, delta=1.0)
        self.assertAlmostEqual(by_id[id_by_source["t2"]].center[1], 40.0, delta=1.0)
        self.assertAlmostEqual(by_id[id_by_source["t3"]].center[0], 136.0, delta=1.0)
        self.assertAlmostEqual(by_id[id_by_source["t3"]].center[1], 20.0, delta=1.0)


class ImageSpaceEmptyBirthTests(unittest.TestCase):
    """sim-empty-birth: b1 is visible only in frames 2-5; frames 0,1,6,7 have nothing."""

    def test_empty_frames_before_birth_produce_no_tracks(self):
        per_frame, _ = run_sequence(EMPTY_BIRTH)
        by_frame = dict(per_frame)
        self.assertEqual(by_frame[0], [])
        self.assertEqual(by_frame[1], [])

    def test_the_track_is_born_confirmed_then_goes_lost_without_being_deleted(self):
        per_frame, id_by_source = run_sequence(EMPTY_BIRTH)
        by_frame = dict(per_frame)
        self.assertEqual(by_frame[2][0].lifecycle, "tentative")
        self.assertEqual(by_frame[3][0].lifecycle, "confirmed")
        self.assertEqual(by_frame[5][0].lifecycle, "confirmed")
        # Frames 6 and 7 are empty again; within the 0.3 s timeout (frame 5 at t=0.625s to
        # frame 7 at t=0.875s is a 0.25 s gap) the track survives as "lost", not deleted.
        self.assertEqual(len(by_frame[6]), 1)
        self.assertEqual(by_frame[6][0].lifecycle, "lost")
        self.assertEqual(len(by_frame[7]), 1)
        self.assertEqual(by_frame[7][0].lifecycle, "lost")
        self.assertEqual({by_frame[2][0].track_id, by_frame[3][0].track_id, by_frame[5][0].track_id,
                          by_frame[6][0].track_id, by_frame[7][0].track_id}, {by_frame[2][0].track_id})

    def test_a_longer_gap_would_delete_it(self):
        # Same scenario, but with a timeout shorter than the 6->7 gap (0.125 s): the track
        # must be gone (not just "lost") by frame 7.
        per_frame, _ = run_sequence(EMPTY_BIRTH, lost_timeout_s=0.1)
        by_frame = dict(per_frame)
        self.assertEqual(by_frame[6], [])  # already deleted: frame5->6 gap is 0.125s > 0.1s
        self.assertEqual(by_frame[7], [])


# ------------------------------------------------------------------ real AOT sequence (AC3)

REAL = json.loads(REAL_PATH.read_text()) if REAL_PATH.is_file() else None
IMG_ROOT = (ROOT / "data/raw/aot/images" / REAL["part"] / "Images" / REAL["flight_id"]) if REAL else None
GT_CSV = (ROOT / "data/raw/aot" / REAL["part"] / "groundtruth.csv") if REAL else None


def _frame_ready(entry) -> bool:
    path = IMG_ROOT / entry["img_name"]
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


HAVE_REAL_FRAMES = bool(REAL) and GT_CSV.is_file() and all(_frame_ready(e) for e in REAL["frames"])
ONNX = ROOT / "models/onnx/yolox_s_aeris_opset17.onnx"


def _load_real_sequence():
    """The pinned real AOT flight's ground truth, in frame order, with the annotation
    records CONTRACTS.md requires (image path/checksum, provided timestamps)."""
    frame_indices = [e["frame"] for e in REAL["frames"]]
    frames = aot.read_frames(str(GT_CSV), REAL["part"], [REAL["flight_id"]])
    first_time_ns = min(frames[(REAL["flight_id"], i)].time_ns for i in frame_indices)
    records = []
    for entry in REAL["frames"]:
        af = frames[(REAL["flight_id"], entry["frame"])]
        path = IMG_ROOT / entry["img_name"]
        records.append(aot.annotation(af, first_time_ns=first_time_ns, image_path=str(path),
                                      image_sha256=entry["sha256"]))
    return sorted(records, key=lambda r: r.frame.frame_index)


class RealAotSequenceTests(unittest.TestCase):
    """AC3: tracking metrics run on identity-labelled real video (AOT), not a synthetic
    fixture. `sequence_id` real() checks make the two tests fail loudly, rather than
    silently pass, when the pinned frames or the ONNX export are unavailable."""

    def setUp(self):
        real(HAVE_REAL_FRAMES, f"pinned AOT frames for {REAL['flight_id'] if REAL else '?'} missing; "
                               "fetch them (see docs/tracking.md) before running strictly")

    def test_oracle_boxes_track_the_real_flight_with_no_identity_switches(self):
        # The tracker's own real-world performance, decoupled from detector recall: the
        # pinned flight's own scored ground-truth boxes are fed in directly as
        # "detections" (score 1.0, provenance says so explicitly). This is an ablation,
        # not a detector+tracker deployment claim -- see test_end_to_end_... below for that.
        truth = _load_real_sequence()
        sequence_id = truth[0].frame.sequence_id
        t = tr.Tracker(sequence_id, config())
        boxes = []
        for record in truth:
            oracle = [DetectionRecord(1, sequence_id, record.frame.frame_index, record.frame.timestamp_s,
                                      aot.WIDTH, aot.HEIGHT, obj.bbox, 0, "aeris-aircraft-v1", 1.0,
                                      "oracle ground truth (ablation, not a detector)", {})
                     for obj in record.objects if not obj.ignore]
            for state in t.step(record.frame.timestamp_s, oracle):
                if state.lifecycle in ("confirmed", "lost"):
                    boxes.append(TrackerBox(sequence_id, record.frame.frame_index, state.track_id,
                                            state.bbox, observed=state.observed))
        result = evaluate_tracking(truth, boxes)
        self.assertEqual(result["eligible_sequences"], [sequence_id])
        combined = result["combined"]
        self.assertIsNotNone(combined)
        self.assertEqual(combined["IDSW"], 0)  # the whole point of AC1's gating, on real data
        self.assertGreaterEqual(combined["MOTA"], 0.9)
        self.assertGreaterEqual(combined["IDF1"], 0.9)

    def test_end_to_end_real_detector_result_is_reported_honestly(self):
        # The real accepted ONNX detector, on this flight's genuinely small (16-45 px)
        # helicopter: AERIS-007 already measured near-zero recall below 32 px on AOT, so a
        # near-empty result here is the honest, expected outcome, not a bug. This proves
        # the real evaluation pipeline runs end to end; it is not a tracker quality claim.
        real(ONNX.is_file(), f"{ONNX} missing (AERIS-006 export)")
        from aeris.detection import baseline as det_module
        from aeris.video.pipeline import OnnxDetector
        truth = _load_real_sequence()
        sequence_id = truth[0].frame.sequence_id
        detector = OnnxDetector(threads=1)
        t = tr.Tracker(sequence_id, config())
        boxes = []
        for record in truth:
            result = det_module.detect(detector.model, _read_frame(record.frame.image_path), detector.settings)
            records = det_module.to_records(result.detections, record.frame, result.letterbox,
                                            detector.settings, detector.provenance)
            for state in t.step(record.frame.timestamp_s, records):
                if state.lifecycle in ("confirmed", "lost"):
                    boxes.append(TrackerBox(sequence_id, record.frame.frame_index, state.track_id,
                                            state.bbox, observed=state.observed))
        result = evaluate_tracking(truth, boxes)
        self.assertEqual(result["eligible_sequences"], [sequence_id])
        self.assertIsNotNone(result["combined"])  # the evaluator ran; the score itself is whatever it is


def _read_frame(path: str):
    import cv2
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise unittest.SkipTest(f"REAL: could not decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    unittest.main()
