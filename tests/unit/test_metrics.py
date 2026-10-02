"""Small synthetic evaluator cases; no real performance claims."""
import dataclasses
import math
import unittest

from aeris.data.contracts import (
    AnnotationRecord, BBox, DetectionRecord, FrameRecord, GroundTruthObject, PredictionRecord,
)
from aeris.evaluation.metrics import (
    EvaluationError, TrackObservation, detection_metrics, forecast_metrics, tracking_metrics,
)


def frame(i, t=None, status="complete"):
    return FrameRecord(1, "s", i, t, "provided" if t is not None else "unavailable",
                       100, 100, f"{i}.png", "0" * 64, None, status,
                       {"camera_id": "unknown", **({"timestamp_s": "still"} if t is None else {})})


def ann(i, t=None, status="complete", objects=True):
    obj = (GroundTruthObject(str(i), 0, BBox(10, 10, 30, 30), "A" if t is not None else None,
                             False, False, False,
                             {} if t is not None else {"track_id": "untracked"}),) if objects else ()
    return AnnotationRecord(1, frame(i, t, status), "map", obj)


def det(i, t=None, box=None):
    return DetectionRecord(1, "s", i, t, 100, 100, box or BBox(10, 10, 30, 30),
                           0, "map", 0.9, "test-model", {} if t is not None else {"timestamp_s": "still"})


def pred(issue, horizon, center, model="cv"):
    return PredictionRecord(1, "s", "A", issue, horizon, issue + horizon, center, None,
                            model, {"covariance": "not supplied", **({"center": "missed"} if center is None else {})})


class MetricTests(unittest.TestCase):
    def test_coco_perfect_missing_and_negative(self):
        truth = [ann(0), ann(1, objects=False)]
        perfect = detection_metrics(truth, [det(0)], score_threshold=0.5)
        assert math.isclose(perfect["ap50"], 1) and math.isclose(perfect["ap50_95"], 1)
        assert perfect["counts"] == {"tp": 1, "fp": 0, "fn": 0}
        assert perfect["width_bins"][1]["support"] == 1
        missing = detection_metrics(truth, [], score_threshold=0.5)
        assert missing["ap50"] == 0 and missing["counts"]["fn"] == 1
        negative = detection_metrics(truth, [det(0), det(1)], score_threshold=0.5)
        assert negative["negative_frame_errors"] == 1
        assert negative["false_positives_per_complete_frame"] == 0.5


    def test_partial_and_unavailable_do_not_score(self):
        partial = ann(0, status="partial")
        unavailable = detection_metrics([partial], [det(0)], score_threshold=0.5)
        assert unavailable["available"] is False
        assert unavailable["reason"] == "no_complete_annotations"
        mixed = detection_metrics([ann(0), ann(1, status="partial")], [det(0), det(1)],
                                  score_threshold=0.5)
        assert mixed["skipped"]["partial"] == 1
        assert mixed["counts"]["fp"] == 0


    def test_trackeval_perfect_missing_wrong_id(self):
        truth = [ann(0, 0), ann(1, 0.1)]
        good = [TrackObservation("s", i, "A", BBox(10, 10, 30, 30), 0) for i in range(2)]
        perfect = tracking_metrics(truth, good)
        assert perfect["hota"] == 1 and perfect["idf1"] == 1
        missing = tracking_metrics(truth, good[:1])
        assert missing["hota"] < 1 and missing["idf1"] < 1
        wrong = tracking_metrics(truth, [good[0], TrackObservation("s", 1, "B", good[1].bbox, 0)])
        assert wrong["hota"] < 1 and wrong["idf1"] < 1
        assert tracking_metrics([ann(0)], [])["available"] is False


    def test_forecast_support_missing_and_interpolation(self):
        truth = [ann(0, 0), ann(1, 0.1), ann(2, 0.25)]
        result = forecast_metrics(truth, [pred(0, 0.1, (20, 20)), pred(0.1, 0.1, None)],
                                  models=["cv", "ca"])
        cv = result["models"]["cv"]["0.1"]
        assert cv["support"] == 2 and cv["matched"] == 1 and cv["missed"] == 1
        assert cv["rmse_px"] == 0
        assert result["models"]["ca"]["0.1"]["matched"] == 0
        bounded = forecast_metrics(truth, [], models=["cv"], max_interpolation_gap_s=0.1)
        assert bounded["models"]["cv"]["0.1"]["support"] == 1

    def test_complete_negative_without_detections(self):
        result = detection_metrics([ann(0, objects=False)], [], score_threshold=0.5)
        assert result["available"] and result["negative_frame_count"] == 1
        assert result["ap50"] is None and result["negative_frame_errors"] == 0

    def test_large_timestamps_do_not_turn_issue_truth_into_future_truth(self):
        issue = 1_000_000_000.0
        truth = [ann(0, issue), ann(1, issue + 0.5)]
        forecast = pred(issue, 0.1, (20, 20))
        result = forecast_metrics(truth, [forecast], models=["cv"],
                                  horizons_s=(0.1,), max_interpolation_gap_s=0.1)
        at_horizon = result["models"]["cv"]["0.1"]
        assert result["available"] is False
        assert result["reason"] == "no_eligible_future_truth"
        assert at_horizon["support"] == 0
        assert at_horizon["matched"] == 0
        assert at_horizon["rmse_px"] is None
        assert result["forecasts_without_eligible_truth"]["cv"] == 1

    def test_large_timestamp_exact_target_remains_eligible(self):
        issue = 1_000_000_000.0
        truth = [ann(0, issue), ann(1, issue + 0.1)]
        result = forecast_metrics(truth, [pred(issue, 0.1, (20, 20))], models=["cv"],
                                  horizons_s=(0.1,), max_interpolation_gap_s=0.1)
        at_horizon = result["models"]["cv"]["0.1"]
        assert at_horizon["support"] == 1
        assert at_horizon["matched"] == 1
        assert at_horizon["rmse_px"] == 0

    def test_nonpositive_horizon_is_rejected(self):
        truth = [ann(0, 0.0), ann(1, 0.1)]
        for bad in ((0.0,), (-0.1,), (0.1, 0.0), ()):
            with self.assertRaises(EvaluationError):
                forecast_metrics(truth, [pred(0.0, 0.1, (20, 20))], models=["cv"], horizons_s=bad)

    def test_wrong_class_detection_is_not_a_match(self):
        wrong = dataclasses.replace(det(0), class_id=1)
        result = detection_metrics([ann(0)], [wrong], score_threshold=0.5)
        assert result["counts"] == {"tp": 0, "fp": 1, "fn": 1}
        assert result["width_bins"][1]["matched"] == 0

    def test_epoch_timestamp_rounding_keeps_last_exact_sample_eligible(self):
        # 1700000000.13 + 0.2 lands one float ulp (~2.4e-7 s) above the recorded
        # 1700000000.33, which is the track's last sample, so it cannot be interpolated.
        # Truth eligibility needs no predictions, so a non-contract horizon is fine here.
        issue, sample = 1700000000.13, 1700000000.33
        assert issue + 0.2 > sample
        result = forecast_metrics([ann(0, issue), ann(1, sample)], [],
                                  models=["cv"], horizons_s=(0.2,))
        assert result["eligible_keys"] == 1
