"""Metric adapter and evaluation protocol tests (AERIS-005) on analytic fixtures.

Expected values are derived by hand from the AERIS-004 fixture design (see comments),
not by re-running the adapters' own helpers. SYNTHETIC inputs: these check evaluator
plumbing and protocol, not model quality.
"""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from aeris.data.contracts import (
    AnnotationRecord, BBox, DetectionRecord, PredictionRecord,
)
from aeris.evaluation import _common
from aeris.evaluation.metrics import (
    PYCOCOTOOLS_VERSION, TRACKEVAL_COMMIT, EvaluationError, EvaluatorVersionError, TrackerBox,
    evaluate_detection, evaluate_forecasts, evaluate_tracking, evaluator_versions,
    require_pinned_evaluators, sequence_bootstrap,
)

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "tests/fixtures/simulation/image_space"
TRUTH = [AnnotationRecord.from_json(line) for line in (SIM / "truth/annotations.jsonl").read_text().splitlines()]
DETECTIONS = [DetectionRecord.from_json(line) for line in (SIM / "inputs/detections.jsonl").read_text().splitlines()]
EXPECTED = json.loads((SIM / "truth/expected.json").read_text())
CROSSING = "sim-cv-crossing"


def detection_like(record, box, score, class_id=0):
    frame = record.frame
    return DetectionRecord(1, frame.sequence_id, frame.frame_index, frame.timestamp_s, frame.width,
                           frame.height, box, class_id, record.class_mapping_version, score,
                           "test detections", {})


def with_status(records, status, sequence_id=None, frames=None):
    out = []
    for record in records:
        hit = ((sequence_id is None or record.frame.sequence_id == sequence_id)
               and (frames is None or record.frame.frame_index in frames))
        if hit:
            objects = () if status == "unlabeled" else record.objects
            record = replace(record, frame=replace(record.frame, annotation_status=status), objects=objects)
        out.append(record)
    return out


def tracker_boxes(records, rename=lambda sequence, frame, track: track, drop=()):
    return [TrackerBox(r.frame.sequence_id, r.frame.frame_index,
                       "trk-" + rename(r.frame.sequence_id, r.frame.frame_index, item.track_id), item.bbox)
            for r in records for item in r.objects
            if (r.frame.sequence_id, r.frame.frame_index, item.track_id) not in drop and not item.ignore]


def analytic_forecasts(offset=(0.0, 0.0), covariance=None, skip=lambda target: False):
    out = []
    for sequence in EXPECTED["sequences"]:
        for target in sequence["forecast_targets"]:
            if skip(target):
                continue
            centre = (target["center_px"][0] + offset[0], target["center_px"][1] + offset[1])
            out.append(PredictionRecord(1, sequence["sequence_id"], target["track_id"],
                                        target["issue_time_s"], target["horizon_s"],
                                        target["target_time_s"], centre, covariance, "analytic-cv",
                                        {} if covariance else {"covariance": "not estimated"}))
    return out


class PinnedEvaluatorTests(unittest.TestCase):
    def test_installed_evaluators_match_pins(self):
        versions = require_pinned_evaluators()
        self.assertEqual((versions["pycocotools"], versions["trackeval"]["commit"]),
                         (PYCOCOTOOLS_VERSION, TRACKEVAL_COMMIT))
        self.assertEqual(evaluate_detection(TRUTH, DETECTIONS)["evaluator"]["versions"], versions)

    def test_unpinned_evaluator_is_refused(self):
        with patch.object(_common.importlib.metadata, "version", return_value="2.0.10"):
            with self.assertRaisesRegex(EvaluatorVersionError, "pycocotools 2.0.10 != pinned 2.0.11"):
                evaluate_detection(TRUTH, DETECTIONS)
            self.assertEqual(evaluator_versions()["pycocotools"], "2.0.10")

    def test_numpy_aliases_are_scoped_to_trackeval(self):
        evaluate_tracking(TRUTH, tracker_boxes(TRUTH))
        self.assertFalse({"float", "int"} & set(np.__dict__))


class CocoAdapterTests(unittest.TestCase):
    def test_perfect_detections(self):
        perfect = [detection_like(r, item.bbox, 1.0) for r in TRUTH for item in r.objects]
        report = evaluate_detection(TRUTH, perfect, score_threshold=0.5)
        self.assertEqual((report["ap50"], report["ap_50_95"], report["precision"], report["recall"],
                          report["false_positives"], report["false_negatives"]), (1.0, 1.0, 1.0, 1.0, 0, 0))

    def test_missed_duplicate_and_false_positive_detections(self):
        # 41 labelled boxes; detections miss t1 at frame 4, duplicate t3 at frame 9 (score
        # 0.55, IoU 72/128 but the box is already taken) and add a false positive (0.3).
        report = evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.5)
        self.assertEqual((report["true_positives"], report["false_positives"], report["false_negatives"]),
                         (40, 1, 1))
        self.assertEqual((report["precision"], report["recall"]), (40 / 41, 40 / 41))
        # Every true positive outscores both errors: precision 1 up to recall 40/41, i.e.
        # 98 of COCO's 101 recall points (0.00..0.97), then 0.
        self.assertAlmostEqual(report["ap50"], 98 / 101, places=12)
        self.assertLess(report["ap_50_95"], report["ap50"])
        low = evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.25)
        self.assertEqual((low["false_positives"], low["precision"]), (2, 40 / 42))
        self.assertEqual(low["false_positives_per_complete_frame"], 2 / 21)
        # The threshold is inclusive: the duplicate scores exactly 0.55.
        self.assertEqual(evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.55)["false_positives"], 1)
        # A box matched only by a detection below the threshold is missed at that threshold.
        rows = [(s["sequence_id"], r) for s in EXPECTED["sequences"] for r in s["detections"]]
        by_frame = {}
        for d in DETECTIONS:
            by_frame.setdefault((d.sequence_id, d.frame_index), []).append(d)
        strong = sum(by_frame[(sid, r["frame_index"])][r["detection_index"]].score >= 0.9
                     for sid, r in rows if r["kind"] == "true_positive")
        high = evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.9)
        self.assertEqual((high["true_positives"], high["false_negatives"]), (strong, 41 - strong))
        self.assertLess(strong, 40)

    def test_no_detections_and_budget(self):
        report = evaluate_detection(TRUTH, [], score_threshold=0.5)
        self.assertEqual((report["ap50"], report["ap_50_95"], report["recall"]), (0.0, 0.0, 0.0))
        self.assertIsNone(report["precision"])
        self.assertIn("no detections", report["availability_reason"]["precision"])
        budget = evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.25, max_detections_per_frame=1)
        # Frames with 2+ detections keep only their highest score.
        per_frame = {}
        for d in DETECTIONS:
            per_frame[(d.sequence_id, d.frame_index)] = per_frame.get((d.sequence_id, d.frame_index), 0) + 1
        self.assertEqual(budget["support"]["detections_dropped_by_budget"],
                         sum(n - 1 for n in per_frame.values()))
        # The false positive (0.3) and duplicate (0.55) never outscore a true positive.
        self.assertEqual(budget["false_positives"], 0)
        with self.assertRaises(EvaluationError):
            evaluate_detection(TRUTH, DETECTIONS, max_detections_per_frame=101)

    def test_threshold_metrics_need_a_supplied_threshold(self):
        report = evaluate_detection(TRUTH, DETECTIONS)
        self.assertIsNotNone(report["ap50"])
        for name in ("precision", "recall", "f1", "false_positives"):
            self.assertIsNone(report[name])
            self.assertIn("score threshold", report["availability_reason"][name])

    def test_inconsistent_inputs_are_rejected(self):
        record = TRUTH[0]
        stray = replace(detection_like(record, BBox(0, 0, 5, 5), 0.9), frame_index=99)
        with self.assertRaisesRegex(EvaluationError, "no truth record"):
            evaluate_detection(TRUTH, [stray])
        wrong_size = replace(detection_like(record, BBox(0, 0, 5, 5), 0.9), width=200)
        with self.assertRaisesRegex(EvaluationError, "width"):
            evaluate_detection(TRUTH, [wrong_size])
        with self.assertRaisesRegex(EvaluationError, "duplicate frame"):
            evaluate_detection(TRUTH + TRUTH[:1], [])


class TrackEvalAdapterTests(unittest.TestCase):
    def test_perfect_tracks(self):
        report = evaluate_tracking(TRUTH, tracker_boxes(TRUTH))
        combined = report["combined"]
        self.assertEqual(report["eligible_sequences"], [CROSSING, "sim-empty-birth"])
        self.assertEqual((combined["HOTA"], combined["MOTA"], combined["IDF1"], combined["IDSW"],
                          combined["IDTP"], combined["MT"]), (1.0, 1.0, 1.0, 0, 41, 4))
        # TrackEval counts a fragment whenever a truth track restarts, including after the
        # object's own visibility gap: t2 is hidden in frames 11-12, so perfect tracking has Frag 1.
        self.assertEqual(combined["Frag"], 1)

    def test_identity_swap_at_the_crossing(self):
        # After the crossing (frames > 8) the tracker swaps t1 and t2. IDF1 matching keeps
        # t1<->trk-t1 (7 frames) and t2<->trk-t2 (7 frames); t3 13 and b1 4 stay correct:
        # IDTP 31 of 41, so IDF1 = 62/82. Two identity switches, no misses: MOTA = 1 - 2/41.
        swap = {"t1": "t2", "t2": "t1"}
        boxes = tracker_boxes(TRUTH, rename=lambda s, f, t: swap.get(t, t) if s == CROSSING and f > 8 else t)
        combined = evaluate_tracking(TRUTH, boxes)["combined"]
        self.assertEqual((combined["IDSW"], combined["IDTP"], combined["IDFN"], combined["IDFP"]),
                         (2, 31, 10, 10))
        self.assertEqual((combined["IDF1"], combined["MOTA"]), (62 / 82, 1 - 2 / 41))
        self.assertLess(combined["HOTA"], 1.0)

    def test_missing_tracker_boxes(self):
        boxes = tracker_boxes(TRUTH, drop={(CROSSING, 3, "t1"), (CROSSING, 4, "t1")})
        combined = evaluate_tracking(TRUTH, boxes)["combined"]
        # 39 of 41 matched: IDF1 = 78/80, MOTA = 1 - 2/41; t1 restarts once (Frag 1 + t2's gap).
        self.assertEqual((combined["CLR_FN"], combined["IDF1"], combined["MOTA"], combined["Frag"]),
                         (2, 78 / 80, 1 - 2 / 41, 2))

    def test_hota_averages_over_localization_thresholds(self):
        # Shifting every box right by w/4.4 gives IoU 3.4/5.4 = 0.63 with its truth box.
        # HOTA(alpha) is 1 for the 12 alphas 0.05..0.60 and 0 for 0.65..0.95: mean 12/19,
        # while CLEAR/Identity (IoU >= 0.5) still see perfect tracking.
        def shifted(box):
            d = (box.x_max - box.x_min) / 4.4
            return BBox(box.x_min + d, box.y_min, box.x_max + d, box.y_max)
        boxes = [replace(b, bbox=shifted(b.bbox)) for b in tracker_boxes(TRUTH)]
        combined = evaluate_tracking(TRUTH, boxes)["combined"]
        self.assertAlmostEqual(combined["HOTA"], 12 / 19, places=12)
        self.assertEqual((combined["MOTA"], combined["IDF1"]), (1.0, 1.0))
        # TrackEval reports LocA 1 at thresholds without matches.
        self.assertAlmostEqual(combined["LocA"], (12 * 3.4 / 5.4 + 7) / 19, places=9)

    def test_predicted_only_boxes_are_reported_separately(self):
        boxes = [replace(b, observed=b.frame_index != 5) for b in tracker_boxes(TRUTH)]
        predicted = sum(not b.observed for b in boxes)
        report = evaluate_tracking(TRUTH, boxes)
        self.assertEqual((report["counts"]["predicted_only_boxes"], report["counts"]["predicted_only_excluded"]),
                         (predicted, predicted))
        self.assertLess(report["combined"]["IDTP"], 41)
        included = evaluate_tracking(TRUTH, boxes, include_predicted=True)
        self.assertEqual((included["combined"]["IDTP"], included["include_predicted"]), (41, True))

    def test_duplicate_tracker_ids_in_a_frame_are_rejected(self):
        boxes = tracker_boxes(TRUTH, rename=lambda s, f, t: "same" if s == CROSSING and f == 0 else t)
        with self.assertRaisesRegex(EvaluationError, "duplicate track ID"):
            evaluate_tracking(TRUTH, boxes)


class ForecastProtocolTests(unittest.TestCase):
    def test_perfect_forecasts_and_hand_derived_support(self):
        report = evaluate_forecasts(analytic_forecasts(), TRUTH)
        last = {(s["sequence_id"], t["track_id"]): t["visible_spans"][-1]["end_s"]
                for s in EXPECTED["sequences"] for t in s["tracks"]}
        targets = [(s["sequence_id"], t) for s in EXPECTED["sequences"] for t in s["forecast_targets"]]
        eligible = sum(t["target_time_s"] <= last[(sid, t["track_id"])] for sid, t in targets)
        self.assertEqual((report["eligible_total"], eligible), (92, 92))
        self.assertEqual(report["unscored_predictions"], {"target_beyond_labelled_span": len(targets) - 92})
        for horizon, entry in report["per_horizon"].items():
            if entry["eligible"]:
                self.assertEqual((entry["scored"], entry["missed"], entry["coverage"]),
                                 (entry["eligible"], 0, 1.0))
                self.assertLess(entry["rmse_px"], 1e-9)
        # Clips last 1 s, so no 2 s target has truth: unavailable, never zero.
        two = report["per_horizon"]["2.0"]
        self.assertEqual((two["eligible"], two["coverage"], two["rmse_px"]), (0, None, None))
        self.assertIn("no eligible truth", two["availability_reason"]["coverage"])
        self.assertIsNone(report["trajectory"]["ade_px"])

    def test_eligible_set_is_model_independent(self):
        digests = {evaluate_forecasts(p, TRUTH)["eligible_set_digest"]
                   for p in ([], analytic_forecasts(), analytic_forecasts(offset=(3.0, 4.0)))}
        self.assertEqual(len(digests), 1)

    def test_missing_forecasts_are_visible_not_silently_better(self):
        full = evaluate_forecasts(analytic_forecasts(offset=(3.0, 4.0)), TRUTH)
        partial = evaluate_forecasts(analytic_forecasts(offset=(3.0, 4.0), skip=lambda t: t["track_id"] == "t2"), TRUTH)
        for horizon in ("0.1", "0.25"):
            self.assertEqual(full["per_horizon"][horizon]["rmse_px"], 5.0)  # |(3, 4)| everywhere
            self.assertGreater(partial["per_horizon"][horizon]["missed"], 0)
            self.assertLess(partial["per_horizon"][horizon]["coverage"], 1.0)
            self.assertEqual(partial["per_horizon"][horizon]["eligible"], full["per_horizon"][horizon]["eligible"])
        # Only t1 forecasts are off by (3, 4); 12 of the 37 eligible 0.1 s forecasts are t1's.
        mixed = [replace(p, center=(p.center[0] + 3, p.center[1] + 4)) if p.track_id == "t1" else p
                 for p in analytic_forecasts()]
        entry = evaluate_forecasts(mixed, TRUTH)["per_horizon"]["0.1"]
        self.assertAlmostEqual(entry["rmse_px"], (12 * 25 / 37) ** 0.5, places=12)
        self.assertAlmostEqual(entry["mean_error_px"], 12 * 5 / 37, places=12)
        nulls = [replace(p, center=None, covariance=None, availability_reason={
            "center": "lost", "covariance": "lost"}) if p.track_id == "t3" else p
            for p in analytic_forecasts()]
        entry = evaluate_forecasts(nulls, TRUTH)["per_horizon"]["0.1"]
        # t3 is visible in all 13 frames, but a 0.1 s target from its last frame (1.0 s) has
        # no truth, so 12 eligible t3 forecasts are missed.
        self.assertEqual((entry["missed_null_centre"], entry["missed"]), (12, 12))

    def test_interval_coverage_uses_chi_square_quantiles(self):
        # Residual (3, 4): squared Mahalanobis distance 25/4 = 6.25 with covariance 4 I
        # (outside the 95% ellipse, quantile 5.99) and 1.0 with 25 I (inside 50%, 1.386).
        tight = evaluate_forecasts(analytic_forecasts((3.0, 4.0), ((4.0, 0.0), (0.0, 4.0))), TRUTH)
        wide = evaluate_forecasts(analytic_forecasts((3.0, 4.0), ((25.0, 0.0), (0.0, 25.0))), TRUTH)
        for level in ("0.5", "0.9", "0.95"):
            self.assertEqual(tight["per_horizon"]["0.1"]["uncertainty"]["levels"][level]["empirical_coverage"], 0.0)
            self.assertEqual(wide["per_horizon"]["0.1"]["uncertainty"]["levels"][level]["empirical_coverage"], 1.0)
        levels = wide["per_horizon"]["0.1"]["uncertainty"]["levels"]
        self.assertAlmostEqual(levels["0.95"]["chi2_quantile"], 5.991464547107979, places=12)
        self.assertAlmostEqual(levels["0.5"]["mean_ellipse_area_px2"], np.pi * 2 * np.log(2) * 25, places=9)

    def test_bounded_gap_policy_and_no_extrapolation(self):
        # t2 is hidden between 0.625 s and 0.875 s (a 0.25 s gap). With max_gap_s 0.2, targets
        # strictly inside that gap lose truth: h 0.1 from 0.5625/0.625, h 0.25 from 0.5/0.5625,
        # h 0.5 from 0.15625/0.25/0.3125 -> 7 of the 92. The 0.1875 s frame drop stays eligible.
        report = evaluate_forecasts(analytic_forecasts(), TRUTH, max_gap_s=0.2)
        self.assertEqual(report["eligible_total"], 85)
        gap = sum(entry["ineligible"].get("target_in_gap_longer_than_max", 0)
                  for entry in report["per_horizon"].values())
        self.assertEqual((gap, report["unscored_predictions"]["target_in_gap_longer_than_max"]), (7, 7))

    def test_future_leakage_suspects_and_unscored_reasons(self):
        base = analytic_forecasts()[0]
        early = replace(base, sequence_id="sim-empty-birth", track_id="b1", issue_time_s=0.0,
                        target_time_s=base.horizon_s)
        off_frame = replace(base, issue_time_s=0.03, target_time_s=0.03 + base.horizon_s)
        unknown = replace(base, track_id="nobody")
        report = evaluate_forecasts([early, off_frame, unknown], TRUTH)
        self.assertEqual(report["future_leakage_suspects"], 1)  # b1 first appears at 0.25 s
        self.assertEqual(report["unscored_predictions"], {"issued_before_track_first_observed": 1,
                                                          "issue_time_not_an_observation": 1,
                                                          "unknown_or_excluded_track": 1})
        with self.assertRaisesRegex(EvaluationError, "duplicate forecast"):
            evaluate_forecasts([base, base], TRUTH)
        with self.assertRaisesRegex(EvaluationError, "one model"):
            evaluate_forecasts([base, replace(base, model_name="other", horizon_s=0.25,
                                              target_time_s=base.issue_time_s + 0.25)], TRUTH)

    def test_sequence_bootstrap_needs_enough_sequences(self):
        groups = {f"s{i}": [float(i), float(i + 1)] for i in range(12)}
        interval = sequence_bootstrap(groups, lambda v: sum(v) / len(v), seed=3)
        self.assertLessEqual(interval["low"], 6.5)
        self.assertGreaterEqual(interval["high"], 6.5)
        self.assertEqual(interval, sequence_bootstrap(groups, lambda v: sum(v) / len(v), seed=3))
        few = sequence_bootstrap(dict(list(groups.items())[:9]), lambda v: sum(v) / len(v))
        self.assertEqual((few["low"], few["high"]), (None, None))
        self.assertIn("9 sequences < minimum 10", few["availability_reason"]["low"])
        report = evaluate_forecasts(analytic_forecasts(), TRUTH)
        self.assertIsNone(report["per_horizon"]["0.1"]["rmse_px_interval"]["low"])


class AvailabilityTests(unittest.TestCase):
    def test_width_bins_report_support_and_zero_support_is_null(self):
        bins = evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.5)["width_bins"]
        # t1 is 16 px wide ([16,32), 13 boxes, missed once); t2, t3, b1 are < 16 px (28 boxes).
        self.assertEqual((bins["[16,32)"]["support"], bins["[16,32)"]["recall"]), (13, 12 / 13))
        self.assertEqual((bins["[0,16)"]["support"], bins["[0,16)"]["recall"]), (28, 1.0))
        for name in ("[32,64)", "[64,inf)"):
            self.assertEqual((bins[name]["support"], bins[name]["recall"]), (0, None))
            self.assertIn("zero support", bins[name]["availability_reason"]["recall"])

    def test_negative_frame_false_positives(self):
        clean = evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.5)["negative_frames"]
        self.assertEqual((clean["count"], clean["with_false_positives"],
                          clean["false_positives_per_negative_frame"]), (4, 0, 0.0))
        negative = next(r for r in TRUTH if r.is_negative)
        extra = detection_like(negative, BBox(10, 10, 20, 20), 0.9)
        report = evaluate_detection(TRUTH, DETECTIONS + [extra], score_threshold=0.5)["negative_frames"]
        self.assertEqual((report["with_false_positives"], report["false_positives_per_negative_frame"]), (1, 0.25))

    def test_partial_labels_leave_unmatched_detections_unscored(self):
        # Frame 2 carries the 0.3-score false positive. Marking it partial must unscore it,
        # not count it, while its labelled objects still count toward recall_including_partial.
        truth = with_status(TRUTH, "partial", CROSSING, {2})
        report = evaluate_detection(truth, DETECTIONS, score_threshold=0.25)
        self.assertEqual(report["false_positives"], 1)  # only the frame-9 duplicate remains
        self.assertEqual(report["support"]["detections_matched_to_ignored_regions"], 1)
        self.assertEqual(report["frames"]["partial"], 1)
        self.assertEqual(report["recall_including_partial"], 40 / 41)
        self.assertEqual(report["true_positives"], 37)  # complete frames only (3 objects moved to partial)
        everything_partial = evaluate_detection(with_status(TRUTH, "partial"), DETECTIONS, score_threshold=0.25)
        for name in ("ap50", "precision", "false_positives"):
            self.assertIsNone(everything_partial[name])
            self.assertIn("partial labels", everything_partial["availability_reason"][name])
        self.assertEqual(everything_partial["recall_including_partial"], 40 / 41)

    def test_ignore_objects_are_neither_true_nor_false_positives(self):
        record = next(r for r in TRUTH if r.frame.sequence_id == CROSSING and r.frame.frame_index == 0)
        ignored = replace(record, objects=tuple(replace(o, ignore=True) if o.track_id == "t3" else o
                                                for o in record.objects))
        truth = [ignored if r is record else r for r in TRUTH]
        report = evaluate_detection(truth, DETECTIONS, score_threshold=0.5)
        self.assertEqual((report["support"]["ground_truth"], report["support"]["ignored_ground_truth"]), (40, 1))
        self.assertEqual((report["true_positives"], report["false_positives"]), (39, 1))
        self.assertEqual(report["support"]["detections_matched_to_ignored_regions"], 1)
        tracking = evaluate_tracking(truth, tracker_boxes(TRUTH))
        self.assertEqual(tracking["counts"]["removed_as_distractor_matches"], 1)
        self.assertEqual(tracking["combined"]["CLR_FP"], 0)

    def test_unlabeled_frames_and_unavailable_truth(self):
        truth = with_status(TRUTH, "unlabeled", "sim-empty-birth", {3})
        report = evaluate_detection(truth, DETECTIONS, score_threshold=0.5)
        self.assertEqual((report["frames"]["unlabeled_excluded"], report["support"]["detections_on_unlabeled_frames"]),
                         (1, 1))
        self.assertEqual(report["support"]["ground_truth"], 40)
        tracking = evaluate_tracking(truth, tracker_boxes(truth))
        self.assertEqual(tracking["eligible_sequences"], [CROSSING])
        self.assertIn("unlabeled", tracking["ineligible_sequences"]["sim-empty-birth"])
        partial = evaluate_tracking(with_status(TRUTH, "partial", CROSSING, {5}), tracker_boxes(TRUTH))
        self.assertEqual(partial["eligible_sequences"], ["sim-empty-birth"])
        self.assertIn("partial", partial["ineligible_sequences"][CROSSING])
        self.assertEqual(partial["counts"]["unscored_on_ineligible_sequences"], 37)
        no_ids = [replace(r, objects=tuple(replace(o, track_id=None, availability_reason={"track_id": "detector-only"})
                                           for o in r.objects)) for r in TRUTH]
        unavailable = evaluate_tracking(no_ids, [])
        self.assertIsNone(unavailable["combined"])
        self.assertIn("detector-only", unavailable["availability_reason"]["combined"])
        untimed = [replace(r, frame=replace(r.frame, timestamp_s=None, timestamp_source="unavailable",
                                            availability_reason={"timestamp_s": "still images"}))
                   for r in TRUTH]
        forecast = evaluate_forecasts(analytic_forecasts(), untimed)
        self.assertIsNone(forecast["per_horizon"])
        self.assertIn("no timed identity-labelled truth", forecast["availability_reason"]["per_horizon"])
        self.assertEqual(set(forecast["sequences"]["excluded"].values()),
                         {"untimed frames: forecast horizons need timestamps"})
        # A class the truth never labels is scored as a false positive, not silently dropped.
        other = detection_like(TRUTH[0], BBox(40, 80, 60, 90), 0.9, class_id=1)
        self.assertEqual(evaluate_detection(TRUTH, DETECTIONS + [other], score_threshold=0.5)["false_positives"], 2)

    def test_reports_are_strict_json(self):
        reports = (evaluate_detection(TRUTH, DETECTIONS, score_threshold=0.5),
                   evaluate_tracking(TRUTH, tracker_boxes(TRUTH)),
                   evaluate_forecasts(analytic_forecasts(covariance=((4.0, 0.0), (0.0, 4.0))), TRUTH))
        for report in reports:
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
