"""Forecast tests on analytic/synthetic scenes. Nothing here measures real-world forecast accuracy."""
import math
import unittest

import numpy as np

from aeris.data.contracts import (
    PREDICTION_HORIZONS_S, AnnotationRecord, BBox, ContractError, FrameRecord, GroundTruthObject,
    PredictionRecord,
)
from aeris.estimation import trajectory as tj
from aeris.estimation.trajectory import (
    MODELS, TrajectoryError, TrajectoryForecaster, evaluate_forecasts, eligible_truth, interval_coverage,
)
from aeris.evaluation.metrics import forecast_metrics

W = H = 4000
TIGHT = {"process_noise": {2: 1.0, 3: 0.5}}    # tight models for clean synthetic motion


def annotation(t, index, objects):
    frame = FrameRecord(1, "s", index, t, "provided", W, H, f"{index}.png", "0" * 64, None, "complete",
                        {"camera_id": "synthetic"})
    objs = tuple(GroundTruthObject(f"{k}{index}", 0, BBox(c[0] - 10, c[1] - 10, c[0] + 10, c[1] + 10),
                                   k, False, False, False, {}) for k, c in objects.items())
    return AnnotationRecord(1, frame, "aeris-aircraft-v1", objs)


def position(key, t):
    if key == "A":                       # constant velocity (60, -20) px/s
        return 200 + 60 * t, 300 - 20 * t
    return 200 + 30 * t + 20 * t * t, 900 + 10 * t     # B: constant acceleration (40, 0) px/s^2


def scene(keys, n=70, dt=0.1, skip=()):
    out = []
    for i in range(n):
        t = round(i * dt, 6)
        objects = {k: position(k, t) for k in keys if (i, k) not in skip}
        out.append(annotation(t, i, objects))
    return out


def center(obj):
    return (obj.bbox.x_min + obj.bbox.x_max) / 2, (obj.bbox.y_min + obj.bbox.y_max) / 2


def run_models(annotations, cfg=None, noise=1.0, seed=1, models=tuple(MODELS), issue_every=1, future_garbage_after=None):
    rng = np.random.default_rng(seed)
    forecasters = {m: TrajectoryForecaster("s", m, cfg) for m in models}
    predictions = []
    for index, ann in enumerate(annotations):
        t = ann.frame.timestamp_s
        for obj in ann.objects:
            cx, cy = center(obj)
            if future_garbage_after is not None and t > future_garbage_after:
                cx, cy = cx + 1e4, cy - 1e4
            z = (cx + rng.normal(0, noise), cy + rng.normal(0, noise))
            for f in forecasters.values():
                f.update(obj.track_id, t, z, max(noise, 1e-3))
        if index % issue_every == 0 and (future_garbage_after is None or t <= future_garbage_after):
            for f in forecasters.values():
                for obj in ann.objects:
                    predictions += f.forecast(obj.track_id, t)
    return predictions


class AnalyticTrajectoryTests(unittest.TestCase):
    def forecasts_for(self, model, key, n=30, cfg=TIGHT):
        f = TrajectoryForecaster("s", model, cfg)
        for i in range(n):
            t = round(i * 0.1, 6)
            f.update(key, t, position(key, t), 1e-3)
        issue = round((n - 1) * 0.1, 6)
        return issue, f.forecast(key, issue)

    def test_constant_velocity_forecast_matches_the_line_for_every_horizon(self):
        issue, records = self.forecasts_for("constant_velocity", "A")
        assert [r.horizon_s for r in records] == list(PREDICTION_HORIZONS_S)
        for r in records:
            np.testing.assert_allclose(r.center, position("A", issue + r.horizon_s), atol=1e-2)
            assert abs(r.target_time_s - (issue + r.horizon_s)) < 1e-9 and r.model_name == "constant_velocity"

    def test_acceleration_model_matches_a_parabola_and_beats_cv_there(self):
        issue, ca = self.forecasts_for("constant_acceleration", "B")
        _, cv = self.forecasts_for("constant_velocity", "B")
        for r in ca:
            np.testing.assert_allclose(r.center, position("B", issue + r.horizon_s), atol=0.5)
        assert np.linalg.norm(np.subtract(cv[-1].center, position("B", issue + 2.0))) > 20

    def test_forecast_after_a_short_coast_propagates_from_the_last_measurement(self):
        f = TrajectoryForecaster("s", "constant_velocity", TIGHT)
        for i in range(30):
            f.update("A", round(i * 0.1, 6), position("A", round(i * 0.1, 6)), 1e-3)
        last, issue = 2.9, 3.3                      # 0.4 s without a measurement, under max_staleness_s
        records = f.forecast("A", issue)
        for r in records:
            np.testing.assert_allclose(r.center, position("A", issue + r.horizon_s), atol=1e-2)
        fresh = f.forecast("A", last)
        assert records[0].covariance[0][0] > fresh[0].covariance[0][0]      # more uncertainty after coasting

    def test_axis_covariance_propagation_matches_closed_forms(self):
        cfg = {**tj.DEFAULTS, "process_noise": {2: 9.0, 3: 4.0}}
        axis = tj._Axis(2, 0.0, 0.0, 2.0, cfg)
        axis.x, axis.p = np.array([5.0, 3.0]), np.array([[4.0, 0.5], [0.5, 2.0]])
        h, q = 0.7, 9.0
        x, p = axis.propagate(h)
        assert abs(x[0] - (5.0 + 3.0 * h)) < 1e-12 and abs(x[1] - 3.0) < 1e-12
        assert abs(p[0, 0] - (4.0 + 2 * h * 0.5 + h * h * 2.0 + q * h**3 / 3)) < 1e-12
        assert abs(p[0, 1] - (0.5 + h * 2.0 + q * h**2 / 2)) < 1e-12
        assert abs(p[1, 1] - (2.0 + q * h)) < 1e-12

    def test_process_noise_matches_numerical_integration_of_the_continuous_model(self):
        for order, density in ((2, 9.0), (3, 4.0)):
            dt = 0.37
            grid = np.linspace(0, dt, 4001)
            g = np.zeros(order)
            g[-1] = 1.0
            terms = []
            for s in grid:
                v = tj.transition(order, dt - s) @ g
                terms.append(density * np.outer(v, v))
            expected = np.trapezoid(np.array(terms), grid, axis=0)
            np.testing.assert_allclose(tj.process_noise(order, dt, density), expected, rtol=1e-6, atol=1e-9)

    def test_uncertainty_grows_with_horizon_and_stays_psd_under_irregular_dt(self):
        rng = np.random.default_rng(2)
        for model in MODELS:
            f = TrajectoryForecaster("s", model)
            t = 0.0
            for _ in range(80):
                t = round(t + float(rng.choice([0.03, 0.1, 0.25, 0.4])), 6)
                f.update("T", t, (500 + 20 * t + rng.normal(0, 1), 700 + rng.normal(0, 1)), 1.0)
            records = f.forecast("T", t)
            areas = [r.covariance[0][0] * r.covariance[1][1] for r in records]
            assert all(a > 0 for a in areas) and areas == sorted(areas), (model, areas)
            for r in records:
                assert np.linalg.eigvalsh(np.array(r.covariance)).min() >= 0


class CausalityTests(unittest.TestCase):
    def test_measurements_after_the_issue_time_cannot_change_a_forecast(self):
        annotations = scene("AB", n=40)
        clean = run_models(annotations[:25])
        garbage_future = run_models(annotations, future_garbage_after=annotations[24].frame.timestamp_s)
        def key(p):
            return p.model_name, p.track_id, p.issue_time_s, p.horizon_s
        a = {key(p): (p.center, p.covariance) for p in clean}
        b = {key(p): (p.center, p.covariance) for p in garbage_future}
        assert a == b and len(a) > 100

    def test_api_refuses_future_use_and_out_of_order_updates(self):
        f = TrajectoryForecaster("s", "constant_velocity")
        f.update("T", 1.0, (10, 10))
        f.update("T", 1.5, (11, 10))
        with self.assertRaises(TrajectoryError):
            f.forecast("T", 1.2)                    # earlier than a measurement already received
        for t in (1.5, 1.0):
            with self.assertRaises(TrajectoryError):
                f.update("T", t, (12, 10))

    def test_input_and_config_validation(self):
        with self.assertRaises(TrajectoryError):
            TrajectoryForecaster("s", "jerky")
        for bad in ({"max_staleness_s": 0}, {"initial_velocity_sigma_px_s": float("nan")},
                    {"process_noise": {3: 1.0}}):
            with self.assertRaises(TrajectoryError):
                TrajectoryForecaster("s", "constant_velocity", bad)
        f = TrajectoryForecaster("s", "constant_velocity")
        for args in ((float("nan"), (1, 1)), (1.0, (float("inf"), 1)), (-1.0, (1, 1)), (True, (1, 1))):
            with self.assertRaises(TrajectoryError):
                f.update("T", *args)
        with self.assertRaises(TrajectoryError):
            f.update("T", 1.0, (1, 1), 0.0)
        with self.assertRaises(ContractError):
            f.update("T", 1.0, (1, 1))
            f.update("T", 1.1, (1, 1))
            f.forecast("T", 1.1, horizons=(0.3,))      # not a contract horizon


class MissedForecastTests(unittest.TestCase):
    def test_unavailable_forecasts_are_records_with_reasons(self):
        f = TrajectoryForecaster("s", "constant_acceleration")
        assert all(r.center is None and "no measurements" in r.availability_reason["center"]
                   for r in f.forecast("ghost", 0.0))
        f.update("T", 0.0, (1, 1))
        f.update("T", 0.1, (1, 1))
        short = f.forecast("T", 0.1)
        assert all(r.center is None and r.covariance is None and "insufficient history" in
                   r.availability_reason["center"] for r in short)
        f.update("T", 0.2, (1, 1))
        assert all(r.center is not None for r in f.forecast("T", 0.2))
        stale = f.forecast("T", 5.0)
        assert all(r.center is None and "exceeds" in r.availability_reason["center"] for r in stale)

    def test_misses_stay_visible_in_the_evaluator(self):
        annotations = scene("A", n=40)
        full = run_models(annotations, models=("constant_velocity",))
        no_a_forecasts = [p for p in full if p.issue_time_s > 1.0]       # early issues never forecast
        report = evaluate_forecasts(annotations, no_a_forecasts, models=["constant_velocity"])
        at = report["errors"]["models"]["constant_velocity"]["0.1"]
        assert at["support"] == 39 and at["missed"] > 0 and at["matched"] < at["support"]
        assert at["coverage"] == at["matched"] / at["support"]
        # coverage_all_eligible counts the misses as not covered, matched coverage ignores them
        level = report["interval_coverage"]["constant_velocity"]["0.1"]["levels"]["0.9"]
        assert level["coverage_all_eligible"] < level["coverage_matched"]


class EvaluationTests(unittest.TestCase):
    def test_eligible_truth_matches_the_aeris_005_evaluator_including_gaps(self):
        annotations = scene("AB", n=40, skip={(10, "A"), (11, "A"), (12, "A"), (20, "B")})
        predictions = run_models(annotations, cfg=TIGHT)
        report = forecast_metrics(annotations, predictions, models=list(MODELS))
        truth = eligible_truth(annotations)
        assert len(truth) == report["eligible_keys"]
        for h in PREDICTION_HORIZONS_S:
            assert sum(k[3] == h for k in truth) == report["models"]["constant_velocity"][str(h)]["support"]

    def test_both_models_share_support_and_report_errors_and_coverage_per_horizon(self):
        annotations = scene("AB", n=70)
        report = evaluate_forecasts(annotations, run_models(annotations, cfg=TIGHT), models=list(MODELS))
        for h in PREDICTION_HORIZONS_S:
            cv, ca = (report["errors"]["models"][m][str(h)] for m in MODELS)
            assert cv["support"] == ca["support"] > 0
            assert cv["rmse_px"] is not None and ca["rmse_px"] is not None
            for model in MODELS:
                cov = report["interval_coverage"][model][str(h)]
                assert cov["support"] == cv["support"] and set(cov["levels"]) == {"0.5", "0.9", "0.95"}
                for level in cov["levels"].values():
                    assert 0 <= level["coverage_matched"] <= 1 and level["mean_ellipse_area_px2"] > 0
        assert "no calibration is claimed" in report["note"]

    def test_model_comparison_directions_on_matched_and_mismatched_motion(self):
        def rmse(keys):
            a = scene(keys, n=70)
            r = evaluate_forecasts(a, run_models(a, cfg=TIGHT), models=list(MODELS))
            return {m: {h: r["errors"]["models"][m][str(h)]["rmse_px"] for h in PREDICTION_HORIZONS_S}
                    for m in MODELS}, r
        steady, _ = rmse("A")
        for h in PREDICTION_HORIZONS_S:
            assert steady["constant_velocity"][h] < steady["constant_acceleration"][h]
        accelerating, report = rmse("B")
        for h in (1.0, 2.0):
            assert accelerating["constant_acceleration"][h] < accelerating["constant_velocity"][h]
        # a model that cannot represent the motion is overconfident: its empirical coverage collapses
        mismatched = report["interval_coverage"]["constant_velocity"]["1.0"]["levels"]["0.9"]
        assert mismatched["coverage_matched"] < 0.2

    def test_interval_coverage_hand_computed(self):
        def prediction(track, err_x, issue=0.0):
            truth_xy = (100.0 + err_x, 100.0)
            return truth_xy, PredictionRecord(1, "s", track, issue, 0.1, issue + 0.1, (100.0, 100.0),
                                              ((4.0, 0.0), (0.0, 4.0)), "m", {})
        truth, preds = {}, []
        for track, err in (("t1", 1.0), ("t2", 3.0), ("t3", 5.0)):       # d2 = 0.25, 2.25, 6.25
            xy, p = prediction(track, err)
            truth[("s", track, 0.0, 0.1)] = xy
            preds.append(p)
        truth[("s", "t4", 0.0, 0.1)] = (0.0, 0.0)                         # eligible but never forecast
        out = interval_coverage(truth, preds, ["m"], horizons=(0.1,))["m"]["0.1"]
        assert out["support"] == 4 and out["scored"] == 3
        assert out["levels"]["0.5"]["covered"] == 1       # 1.386
        assert out["levels"]["0.9"]["covered"] == 2       # 4.605
        assert out["levels"]["0.95"]["covered"] == 2      # 5.991 (6.25 outside)
        assert out["levels"]["0.9"]["coverage_matched"] == 2 / 3 and out["levels"]["0.9"]["coverage_all_eligible"] == 0.5
        expected_area = tj.CHI2_2DOF[0.9] * math.pi * 4.0
        assert abs(out["levels"]["0.9"]["mean_ellipse_area_px2"] - expected_area) < 1e-9


if __name__ == "__main__":
    unittest.main()
