"""Causal image-space forecast tests (AERIS-010).

Unit tests validate the Kalman math exactly (hand-derived transition/process-noise
closed forms, PSD safety, causality/warm-up) and cross-check both models against two
independently computed analytic trajectories: AERIS-004's own `sim-cv-crossing`
constant-velocity fixture (read-only reuse; not owned by this task) and this task's own
constant-acceleration fixture (`tests/fixtures/trajectory`).

Tests marked REAL need the 23 pinned frames of one real, identity-labelled AOT flight
already fetched for AERIS-009 (tests/fixtures/tracking/real_samples.json); they skip
with the reason when those are absent, and AERIS_REQUIRE_REAL_FORECAST=1 turns the
skips into failures. This measures the forecast models on real motion, not detector or
tracker quality.
"""
import hashlib
import json
import math
import os
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from aeris.data import aot
from aeris.data.contracts import AnnotationRecord, BBox, FrameRecord, GroundTruthObject, PredictionRecord
from aeris.estimation import trajectory as tj

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "tests/fixtures/simulation/image_space"
TRAJ = ROOT / "tests/fixtures/trajectory"
REAL_PATH = ROOT / "tests/fixtures/tracking/real_samples.json"
STRICT = os.environ.get("AERIS_REQUIRE_REAL_FORECAST") == "1"


def real(ready: bool, reason: str):
    """Skip (or, under AERIS_REQUIRE_REAL_FORECAST=1, fail) when REAL inputs are missing."""
    if ready:
        return
    if STRICT:
        raise AssertionError(f"REAL evidence required but unavailable: {reason}")
    raise unittest.SkipTest(f"REAL: {reason}")


def frame(sequence_id, index, timestamp_s, width=200, height=200, status="complete"):
    return FrameRecord(1, sequence_id, index, timestamp_s, "provided", width, height,
                       f"{sequence_id}/{index}.pgm",
                       hashlib.sha256(f"{sequence_id}-{index}".encode()).hexdigest(),
                       None, status, {"camera_id": "test"})


def obj(annotation_id, cx, cy, track_id="t1", ignore=False, w=10.0, h=10.0):
    box = BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    reasons = {} if track_id is not None else {"track_id": "test: untracked object"}
    return GroundTruthObject(annotation_id, 0, box, track_id, ignore, False, False, reasons)


def annotation(sequence_id, index, timestamp_s, objects, status="complete"):
    return AnnotationRecord(1, frame(sequence_id, index, timestamp_s, status=status),
                            "aeris-aircraft-v1", tuple(objects))


# ------------------------------------------------------------------ Kalman math (AC2)

class MathTests(unittest.TestCase):
    def test_cv_transition_matches_hand_derived_block_diagonal_form(self):
        f = tj._transition("constant_velocity", 0.4)
        expected = np.array([[1, 0.4, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.4], [0, 0, 0, 1]])
        np.testing.assert_array_equal(f, expected)

    def test_ca_transition_matches_hand_derived_block_diagonal_form(self):
        f = tj._transition("constant_acceleration", 0.5)
        block = np.array([[1, 0.5, 0.125], [0, 1, 0.5], [0, 0, 1]])
        expected = np.zeros((6, 6))
        expected[:3, :3] = block
        expected[3:, 3:] = block
        np.testing.assert_array_almost_equal(f, expected)

    def test_cv_process_noise_matches_discrete_white_noise_acceleration_closed_form(self):
        dt = 0.3
        q = tj._process_noise("constant_velocity", dt)
        block = tj.CV_ACCEL_STD_PX_S2 ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        np.testing.assert_array_almost_equal(q[:2, :2], block)
        np.testing.assert_array_almost_equal(q[2:, 2:], block)
        np.testing.assert_array_almost_equal(q[:2, 2:], np.zeros((2, 2)))  # axes independent

    def test_ca_process_noise_matches_independent_numeric_integration_of_the_transition(self):
        # Independently re-derive Q(dt) for one axis by numerically integrating
        # Phi(s) G G^T Phi(s)^T ds over [0, dt] for the continuous white-noise-jerk model
        # (Phi(s) = the 3x3 constant-acceleration transition, G = [0,0,1]) and compare to
        # the module's closed form -- not just re-checking the module against itself.
        dt = 0.65
        samples = np.linspace(0, dt, 20001)
        integrand = []
        for s in samples:
            phi = np.array([[1, s, s ** 2 / 2], [0, 1, s], [0, 0, 1]])
            v = phi @ np.array([0.0, 0.0, 1.0])
            integrand.append(np.outer(v, v))
        numeric = tj.CA_JERK_STD_PX_S3 ** 2 * np.trapezoid(np.array(integrand), samples, axis=0)
        q = tj._process_noise("constant_acceleration", dt)
        np.testing.assert_allclose(q[:3, :3], numeric, atol=1e-6)

    def test_process_noise_grows_with_elapsed_time(self):
        for model in tj.MODEL_NAMES:
            small = tj._process_noise(model, 0.1)
            large = tj._process_noise(model, 1.0)
            self.assertTrue((np.diag(large) > np.diag(small)).all())

    def test_symmetrize_psd_clips_float_noise_but_rejects_a_real_negative_eigenvalue(self):
        noisy = np.array([[1.0, 1e-13], [-1e-13, 1.0 - 1e-16]])
        result = tj._symmetrize_psd(noisy)
        np.testing.assert_array_almost_equal(result, np.eye(2))
        defective = np.array([[1.0, 0.0], [0.0, -0.5]])
        with self.assertRaises(tj.TrajectoryError):
            tj._symmetrize_psd(defective)

    def test_predict_and_update_actually_call_symmetrize_psd(self):
        # Each update() symmetrizes both the predicted and the posterior covariance (2
        # calls); forecast(), once the model has warmed up, symmetrizes the propagated
        # covariance once more (1 call).
        for model in tj.MODEL_NAMES:
            with unittest.mock.patch.object(tj, "_symmetrize_psd", wraps=tj._symmetrize_psd) as spy:
                filt = tj._CausalFilter(model, 0.0, (0.0, 0.0))
                self.assertEqual(spy.call_count, 0)
                updates_to_warm_up = tj.MIN_OBSERVATIONS[model] - 1
                for i in range(updates_to_warm_up):
                    filt.update(0.1 * (i + 1), (1.0 + i, 1.0 + i))
                self.assertEqual(spy.call_count, 2 * updates_to_warm_up)
                centre, _, reason = filt.forecast(0.1)
                self.assertIsNotNone(centre, reason)
                self.assertEqual(spy.call_count, 2 * updates_to_warm_up + 1)


# ------------------------------------------------------------------ causal filter behaviour (AC1)

class CausalFilterTests(unittest.TestCase):
    def test_constant_velocity_needs_two_observations_before_forecasting(self):
        filt = tj._CausalFilter("constant_velocity", 0.0, (0.0, 0.0))
        centre, covariance, reason = filt.forecast(0.5)
        self.assertIsNone(centre)
        self.assertIsNone(covariance)
        self.assertIn("need >= 2", reason)
        filt.update(0.1, (10.0, 0.0))
        centre, _, reason = filt.forecast(0.5)
        self.assertIsNotNone(centre)
        self.assertIsNone(reason)

    def test_constant_acceleration_needs_three_observations_before_forecasting(self):
        filt = tj._CausalFilter("constant_acceleration", 0.0, (0.0, 0.0))
        filt.update(0.1, (1.0, 0.0))
        centre, _, reason = filt.forecast(0.5)
        self.assertIsNone(centre, f"should still be null after {filt.n_observations} observations")
        self.assertIn("need >= 3", reason)
        filt.update(0.2, (3.0, 0.0))
        centre, _, reason = filt.forecast(0.5)
        self.assertIsNotNone(centre)
        self.assertIsNone(reason)

    def test_nonincreasing_timestamp_is_refused(self):
        filt = tj._CausalFilter("constant_velocity", 1.0, (0.0, 0.0))
        with self.assertRaises(tj.TrajectoryError):
            filt.update(1.0, (1.0, 1.0))
        with self.assertRaises(tj.TrajectoryError):
            filt.update(0.5, (1.0, 1.0))

    def test_exact_constant_velocity_recovery_converges_with_repeated_clean_observations(self):
        # x(t) = 10 + 40*t; feeding many exact points should make the filter's own
        # forecast converge tightly to the closed form, even though it never sees the
        # true velocity directly (it always assumes POSITION_STD_PX measurement noise).
        filt = tj._CausalFilter("constant_velocity", 0.0, (10.0, 0.0))
        for i in range(1, 30):
            t = i * 0.05
            filt.update(t, (10.0 + 40.0 * t, 0.0))
        centre, _, _ = filt.forecast(0.2)
        target_t = filt.t + 0.2
        self.assertAlmostEqual(centre[0], 10.0 + 40.0 * target_t, delta=0.05)

    def test_forecast_covariance_is_symmetric_and_grows_with_horizon(self):
        filt = tj._CausalFilter("constant_acceleration", 0.0, (0.0, 0.0))
        filt.update(0.1, (1.0, 1.0))
        filt.update(0.2, (3.0, 3.0))
        _, cov_short, _ = filt.forecast(0.1)
        _, cov_long, _ = filt.forecast(2.0)
        self.assertEqual(cov_short[0][1], cov_short[1][0])
        self.assertGreater(cov_long[0][0], cov_short[0][0])


# ------------------------------------------------------------------ causal_observations / forecast_truth

class CausalObservationsTests(unittest.TestCase):
    def test_ignored_and_untracked_objects_are_excluded(self):
        truth = [annotation("s1", 0, 0.0, [obj("a", 50, 50), obj("b", 5, 5, track_id="t-ignore", ignore=True),
                                           obj("c", 9, 9, track_id=None)])]
        tracks = tj.causal_observations(truth)
        self.assertEqual(set(tracks), {("s1", "t1")})

    def test_unlabeled_frames_contribute_no_observations(self):
        truth = [annotation("s1", 0, 0.0, [], status="unlabeled"),
                annotation("s1", 1, 0.1, [obj("a", 50, 50)])]
        tracks = tj.causal_observations(truth)
        self.assertEqual(tracks[("s1", "t1")], [(0.1, (50.0, 50.0))])

    def test_untimed_frames_are_excluded(self):
        f = FrameRecord(1, "s1", 0, None, "unavailable", 100, 100, "s1/0.pgm",
                        hashlib.sha256(b"x").hexdigest(), None, "complete",
                        {"camera_id": "test", "timestamp_s": "no time series"})
        record = AnnotationRecord(1, f, "aeris-aircraft-v1", (obj("a", 50, 50),))
        self.assertEqual(tj.causal_observations([record]), {})

    def test_forecast_truth_rejects_non_annotation_records(self):
        with self.assertRaises(tj.TrajectoryError):
            tj.forecast_truth([object()], "constant_velocity")

    def test_forecast_truth_rejects_unknown_model(self):
        with self.assertRaises(tj.TrajectoryError):
            tj.forecast_truth([], "constant_jerk")

    def test_every_prediction_is_a_valid_contract_record_with_consistent_availability(self):
        truth = [annotation("s1", i, i * 0.1, [obj("a", 50.0 + i * 10.0, 50.0)]) for i in range(4)]
        for model in tj.MODEL_NAMES:
            predictions = tj.forecast_truth(truth, model)
            self.assertTrue(predictions)
            for p in predictions:
                self.assertIsInstance(p, PredictionRecord)
                if p.center is None:
                    self.assertEqual(set(p.availability_reason), {"center", "covariance"})
                else:
                    self.assertEqual(p.availability_reason, {})
                    self.assertIsNotNone(p.covariance)


# ------------------------------------------------------------------ analytic fixtures (AC2)

def _load_sim_crossing():
    lines = (SIM / "truth/annotations.jsonl").read_text().splitlines()
    truth = [AnnotationRecord.from_json(line) for line in lines
            if json.loads(line)["frame"]["sequence_id"] == "sim-cv-crossing"]
    expected = json.loads((SIM / "truth/expected.json").read_text())
    seq = next(s for s in expected["sequences"] if s["sequence_id"] == "sim-cv-crossing")
    return truth, seq


class ConstantVelocityAnalyticFixtureTests(unittest.TestCase):
    """Cross-check against AERIS-004's own independently computed CV forecast targets
    (not this module's code) for the sim-cv-crossing fixture: three real constant-
    velocity ground-truth tracks, one stationary, with a genuine crossing."""

    def test_forecasts_match_the_independently_computed_targets_closely(self):
        truth, seq = _load_sim_crossing()
        targets = {(t["track_id"], t["issue_frame_index"], t["horizon_s"]): t
                  for t in seq["forecast_targets"]}
        time_by_frame = {row["frame_index"]: row["timestamp_s"] for row in seq["frame_times"]}
        predictions = tj.forecast_truth(truth, "constant_velocity")
        by_key = {(p.track_id, round(p.issue_time_s, 9), p.horizon_s): p for p in predictions}
        errors, missed_within_span = [], 0
        for (track_id, frame_index, horizon_s), target in targets.items():
            if not target["target_within_sequence"]:
                continue
            issue_time = time_by_frame[frame_index]
            prediction = by_key.get((track_id, round(issue_time, 9), horizon_s))
            self.assertIsNotNone(prediction, (track_id, frame_index, horizon_s))
            if prediction.center is None:
                missed_within_span += 1
                continue
            want = target["center_px"]
            errors.append(math.dist(prediction.center, want))
        self.assertGreater(len(errors), 50)  # the comparison actually exercised most targets
        self.assertLess(max(errors), 0.1)  # px; measured max ~0.027
        # Exactly the warm-up forecasts (each track's first observed frame) are missing,
        # not silently substituted -- AC2's "missed forecasts remain visible".
        self.assertGreater(missed_within_span, 0)


class ConstantAccelerationAnalyticFixtureTests(unittest.TestCase):
    """This task's own fixture: one track under exact constant acceleration
    (x0=50,y0=80, v0=(40,-20) px/s, a=(32,16) px/s^2, dyadic frame times), with forecast
    targets computed independently of aeris.estimation.trajectory (tests/fixtures/trajectory)."""

    def setUp(self):
        self.expected = json.loads((TRAJ / "expected.json").read_text())
        lines = (TRAJ / "constant_acceleration.jsonl").read_text().splitlines()
        self.assertEqual(hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
                         self.expected["sha256_annotations_jsonl"])
        self.truth = [AnnotationRecord.from_json(line) for line in lines]
        self.frame_interval = self.expected["frame_interval_s"]

    def test_fixture_timestamps_are_exact_in_double_precision_by_construction(self):
        form = self.expected["closed_form"]
        for record in self.truth:
            t = record.frame.timestamp_s
            box = record.objects[0].bbox
            cx, cy = (box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2
            want_x = form["x0"] + form["vx0"] * t + 0.5 * form["ax"] * t * t
            want_y = form["y0"] + form["vy0"] * t + 0.5 * form["ay"] * t * t
            self.assertEqual(cx, want_x)
            self.assertEqual(cy, want_y)

    def test_constant_acceleration_model_converges_to_the_closed_form(self):
        targets = {(t["issue_frame_index"], t["horizon_s"]): t["center_px"]
                  for t in self.expected["forecast_targets"] if t["within_fixture_span"]}
        predictions = tj.forecast_truth(self.truth, "constant_acceleration")
        by_key = {(round(p.issue_time_s / self.frame_interval), p.horizon_s): p for p in predictions}
        late_errors, early_present = [], False
        for (frame_index, horizon_s), want in targets.items():
            prediction = by_key[(frame_index, horizon_s)]
            if prediction.center is None:
                continue
            early_present = True
            error = math.dist(prediction.center, want)
            if frame_index >= 10:
                late_errors.append(error)
        self.assertTrue(early_present)
        self.assertGreater(len(late_errors), 20)
        self.assertLess(max(late_errors), 0.05)  # px; measured well under this after warm-up

    def test_constant_velocity_model_shows_larger_error_on_a_genuinely_accelerating_target(self):
        # The CV model assumes zero acceleration, so on a target that truly accelerates
        # it must drift away from the closed form as the horizon grows -- demonstrating
        # the two models are not interchangeable, not merely differently coded.
        targets = {(t["issue_frame_index"], t["horizon_s"]): t["center_px"]
                  for t in self.expected["forecast_targets"] if t["within_fixture_span"]}
        cv_predictions = {(round(p.issue_time_s / self.frame_interval), p.horizon_s): p
                          for p in tj.forecast_truth(self.truth, "constant_velocity")}
        short = cv_predictions[(5, 0.1)]
        long = cv_predictions[(5, 2.0)]
        self.assertIsNotNone(short.center)
        self.assertIsNotNone(long.center)
        short_error = math.dist(short.center, targets[(5, 0.1)])
        long_error = math.dist(long.center, targets[(5, 2.0)])
        self.assertGreater(long_error, short_error)
        self.assertGreater(long_error, 1.0)  # px; a CV model cannot track this accelerating target


# ------------------------------------------------------------------ comparable dual-model evaluation (AC3)

class ComparableEvaluationTests(unittest.TestCase):
    def setUp(self):
        lines = (TRAJ / "constant_acceleration.jsonl").read_text().splitlines()
        self.records = [AnnotationRecord.from_json(line) for line in lines]

    def test_both_models_share_an_identical_eligible_set_from_the_same_truth(self):
        results = tj.evaluate_models(self.records)
        cv, ca = results["constant_velocity"], results["constant_acceleration"]
        self.assertEqual(cv["eligible_total"], ca["eligible_total"])
        self.assertEqual(cv["eligible_set_digest"], ca["eligible_set_digest"])
        self.assertGreater(cv["eligible_total"], 0)

    def test_missed_forecasts_from_insufficient_warmup_are_visible_not_dropped(self):
        results = tj.evaluate_models(self.records)
        cv_missed = sum(v["missed"] for v in results["constant_velocity"]["per_horizon"].values())
        ca_missed = sum(v["missed"] for v in results["constant_acceleration"]["per_horizon"].values())
        # constant_acceleration needs one more causal observation to warm up than
        # constant_velocity does, so it must report strictly more missed forecasts on
        # the very same eligible set -- not the same number, and never fewer.
        self.assertGreater(ca_missed, cv_missed)

    def test_per_horizon_rmse_and_coverage_are_reported_for_both_models(self):
        lines = (SIM / "truth/annotations.jsonl").read_text().splitlines()
        truth = [AnnotationRecord.from_json(line) for line in lines
                if json.loads(line)["frame"]["sequence_id"] == "sim-cv-crossing"]
        results = tj.evaluate_models(truth)
        for model_name, result in results.items():
            for horizon_s, entry in result["per_horizon"].items():
                self.assertIn("rmse_px", entry)
                self.assertIn("uncertainty", entry)
                if entry["scored"] > 0:
                    self.assertIsNotNone(entry["rmse_px"])


# ------------------------------------------------------------------ real AOT sequence (AC3)

REAL = json.loads(REAL_PATH.read_text()) if REAL_PATH.is_file() else None
IMG_ROOT = (ROOT / "data/raw/aot/images" / REAL["part"] / "Images" / REAL["flight_id"]) if REAL else None
GT_CSV = (ROOT / "data/raw/aot" / REAL["part"] / "groundtruth.csv") if REAL else None


def _frame_ready(entry) -> bool:
    path = IMG_ROOT / entry["img_name"]
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


HAVE_REAL_FRAMES = bool(REAL) and GT_CSV.is_file() and all(_frame_ready(e) for e in REAL["frames"])


def _load_real_sequence():
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
    """AC3 on real, identity-labelled motion (not a synthetic fixture): the same 23
    pinned AOT helicopter frames AERIS-009 already fetched and validated its own
    tracker against. This is forecasting evidence, not a tracking or detection claim --
    the oracle-identity protocol never involves a detector or tracker here."""

    def setUp(self):
        real(HAVE_REAL_FRAMES, f"pinned AOT frames for {REAL['flight_id'] if REAL else '?'} missing; "
                               "fetch them (see docs/tracking.md) before running strictly")

    def test_both_models_forecast_the_real_flight_with_identical_eligible_support(self):
        truth = _load_real_sequence()
        results = tj.evaluate_models(truth)
        cv, ca = results["constant_velocity"], results["constant_acceleration"]
        self.assertEqual(cv["eligible_total"], ca["eligible_total"])
        self.assertEqual(cv["eligible_set_digest"], ca["eligible_set_digest"])
        self.assertGreater(cv["eligible_total"], 0)
        for model_name, result in results.items():
            scored_any = any(v["scored"] > 0 for v in result["per_horizon"].values())
            self.assertTrue(scored_any, f"{model_name}: no horizon scored anything on real data")


if __name__ == "__main__":
    unittest.main()
