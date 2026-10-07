"""Causal image-space forecasts: constant velocity and a separately specified acceleration model.

Each model is its own per-axis Kalman filter fed the *same* timed measurements
(``update``); neither reads tracker state, detections after the issue time, or truth.
``forecast`` propagates the current state ``issue - last_update + horizon`` seconds and
returns contract ``PredictionRecord`` objects (original-frame pixels, px^2 covariance).
A forecast that cannot be made is still returned, with a null center and a reason, so it
stays visible as a miss. Coverage is empirical and conditional; filter covariance is not
assumed calibrated. Parameters are untuned engineering defaults.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from aeris.data.contracts import PREDICTION_HORIZONS_S, AnnotationRecord, PredictionRecord
from aeris.evaluation.metrics import forecast_metrics

MODELS = {"constant_velocity": 2, "constant_acceleration": 3}   # state size per axis
COVERAGE_LEVELS = (0.5, 0.9, 0.95)
CHI2_2DOF = {0.5: 1.3862943611198906, 0.9: 4.605170185988092, 0.95: 5.991464547107979}


class TrajectoryError(ValueError):
    """Inputs violate the causal-forecast contract."""


def transition(order: int, dt: float) -> np.ndarray:
    f = np.eye(order)
    f[0, 1] = dt
    if order == 3:
        f[0, 2], f[1, 2] = dt * dt / 2, dt
    return f


def process_noise(order: int, dt: float, q: float) -> np.ndarray:
    """Continuous white-noise acceleration (order 2, q in px^2/s^3) or jerk (order 3, q in px^2/s^5)
    integrated over ``dt``, so the noise is consistent for any (irregular) delta time."""
    if order == 2:
        return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    return q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                         [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                         [dt**3 / 6, dt**2 / 2, dt]])


class _Axis:
    """One image axis of one track under one motion model."""

    def __init__(self, order: int, z: float, t: float, sigma: float, cfg: dict):
        self.order, self.cfg = order, cfg
        self.x = np.zeros(order)
        self.x[0] = z
        self.p = np.diag([sigma**2, cfg["initial_velocity_sigma_px_s"] ** 2,
                          cfg["initial_acceleration_sigma_px_s2"] ** 2][:order])
        self.t = t
        self.n = 1

    def propagate(self, dt: float):
        f, q = transition(self.order, dt), process_noise(self.order, dt, self.cfg["process_noise"][self.order])
        p = f @ self.p @ f.T + q
        return f @ self.x, (p + p.T) / 2

    def updated(self, z: float, t: float, sigma: float):
        """Posterior ``(x, p)`` after a measurement; pure, so a failure leaves this axis untouched."""
        x, p = self.propagate(t - self.t)
        h = np.zeros((1, self.order))
        h[0, 0] = 1.0
        s = float((h @ p @ h.T)[0, 0]) + sigma**2
        gain = (p @ h.T) / s
        x = x + (gain * (z - float((h @ x)[0]))).ravel()
        ikh = np.eye(self.order) - gain @ h
        p = ikh @ p @ ikh.T + gain @ gain.T * sigma**2     # Joseph form
        p = (p + p.T) / 2
        if not np.all(np.isfinite(p)) or np.linalg.eigvalsh(p).min() < -1e-8 * max(1.0, p.max()):
            raise TrajectoryError("covariance lost positive semi-definiteness")
        return x, p

    def commit(self, x, p, t: float) -> None:
        self.x, self.p, self.t, self.n = x, p, t, self.n + 1


DEFAULTS = {
    "process_noise": {2: 100.0 ** 2, 3: 500.0 ** 2},   # accel density (px^2/s^3) / jerk density (px^2/s^5)
    "initial_velocity_sigma_px_s": 100.0,
    "initial_acceleration_sigma_px_s2": 500.0,
    "min_updates": {2: 2, 3: 3},
    "max_staleness_s": 1.0,
}


class TrajectoryForecaster:
    """Causal forecaster for one sequence and one motion model."""

    def __init__(self, sequence_id: str, model: str, config: dict | None = None):
        if model not in MODELS:
            raise TrajectoryError(f"model must be one of {sorted(MODELS)}")
        self.sequence_id, self.model, self.order = sequence_id, model, MODELS[model]
        self.cfg = {**DEFAULTS, **(config or {})}
        for key in ("initial_velocity_sigma_px_s", "initial_acceleration_sigma_px_s2", "max_staleness_s"):
            if not math.isfinite(self.cfg[key]) or self.cfg[key] <= 0:
                raise TrajectoryError(f"{key} must be positive and finite")
        for table in ("process_noise", "min_updates"):
            if self.order not in self.cfg[table] or self.cfg[table][self.order] <= 0:
                raise TrajectoryError(f"{table} missing or nonpositive for order {self.order}")
        self._axes: dict[str, tuple[_Axis, _Axis]] = {}

    def update(self, track_id: str, timestamp_s: float, center: tuple[float, float],
               sigma_px: float = 1.0) -> None:
        """Add one measured center. Timestamps per track must strictly increase."""
        for value in (timestamp_s, *center, sigma_px):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TrajectoryError("timestamp, center and sigma must be finite numbers")
        if timestamp_s < 0 or sigma_px <= 0:
            raise TrajectoryError("timestamp must be nonnegative and sigma positive")
        axes = self._axes.get(track_id)
        if axes is None:
            self._axes[track_id] = tuple(_Axis(self.order, c, timestamp_s, sigma_px, self.cfg) for c in center)
            return
        if timestamp_s <= axes[0].t:
            raise TrajectoryError(f"track {track_id}: timestamp {timestamp_s} does not exceed {axes[0].t}")
        # Compute both axes first and commit together: a failure must not leave a half-updated track.
        posteriors = [axis.updated(c, timestamp_s, sigma_px) for axis, c in zip(axes, center)]
        for axis, (x, p) in zip(axes, posteriors):
            axis.commit(x, p, timestamp_s)

    def forecast(self, track_id: str, issue_time_s: float,
                 horizons: tuple[float, ...] = PREDICTION_HORIZONS_S) -> list[PredictionRecord]:
        """One record per horizon from measurements at or before ``issue_time_s``."""
        axes = self._axes.get(track_id)
        if axes is not None and issue_time_s < axes[0].t:
            raise TrajectoryError("issue time precedes a measurement already received (would use the future)")
        records = []
        for horizon in horizons:
            reason = None
            if axes is None:
                reason = "no measurements for this track"
            elif axes[0].n < self.cfg["min_updates"][self.order]:
                reason = f"insufficient history ({axes[0].n} of {self.cfg['min_updates'][self.order]} updates)"
            elif issue_time_s - axes[0].t > self.cfg["max_staleness_s"]:
                reason = f"last measurement {issue_time_s - axes[0].t:.3g} s old exceeds {self.cfg['max_staleness_s']} s"
            if reason is not None:
                records.append(self._record(track_id, issue_time_s, horizon, None, None,
                                            {"center": reason, "covariance": reason}))
                continue
            lead = issue_time_s - axes[0].t + horizon
            (xs, ps), (ys, py) = (a.propagate(lead) for a in axes)
            records.append(self._record(track_id, issue_time_s, horizon, (float(xs[0]), float(ys[0])),
                                        ((float(ps[0, 0]), 0.0), (0.0, float(py[0, 0]))), {}))
        return records

    def _record(self, track_id, issue, horizon, center, covariance, reason):
        return PredictionRecord(1, self.sequence_id, track_id, issue, horizon, issue + horizon,
                                center, covariance, self.model, reason)


def eligible_truth(annotations: list[AnnotationRecord], horizons=PREDICTION_HORIZONS_S,
                   max_interpolation_gap_s: float = 0.5) -> dict:
    """Evaluator-only future truth, keyed ``(sequence, track, issue, horizon)``.

    Mirrors the AERIS-005 rule (complete timed frames, observed non-ignored identity, exact
    match within float spacing or interpolation across a bounded gap, never extrapolation);
    tests cross-check the key set against ``forecast_metrics``.
    """
    by_track, issues = defaultdict(list), []
    for ann in annotations:
        if ann.frame.annotation_status != "complete" or ann.frame.timestamp_s is None:
            continue
        for obj in ann.objects:
            if obj.ignore or obj.track_id is None:
                continue
            key = (ann.frame.sequence_id, obj.track_id)
            center = ((obj.bbox.x_min + obj.bbox.x_max) / 2, (obj.bbox.y_min + obj.bbox.y_max) / 2)
            by_track[key].append((ann.frame.timestamp_s, center))
            issues.append((key, ann.frame.timestamp_s))
    for values in by_track.values():
        values.sort()
    truth = {}
    for key, issue in issues:
        values = by_track[key]
        for horizon in horizons:
            target = issue + horizon
            exact = next((c for t, c in values if abs(t - target) <= max(1e-9, 4 * math.ulp(target))), None)
            if exact is not None:
                truth[(key[0], key[1], issue, horizon)] = exact
                continue
            for (t0, c0), (t1, c1) in zip(values, values[1:]):
                if t0 < target < t1 and t1 - t0 <= max_interpolation_gap_s:
                    w = (target - t0) / (t1 - t0)
                    truth[(key[0], key[1], issue, horizon)] = (c0[0] + w * (c1[0] - c0[0]),
                                                               c0[1] + w * (c1[1] - c0[1]))
                    break
    return truth


def interval_coverage(truth: dict, predictions: list[PredictionRecord], models: list[str],
                      horizons=PREDICTION_HORIZONS_S, levels=COVERAGE_LEVELS) -> dict:
    """Empirical coverage of the 2-D Gaussian prediction ellipse at each level, per model/horizon.

    ``coverage_matched`` divides by forecasts that could be scored; ``coverage_all_eligible``
    divides by every eligible key, so misses count as not covered. Null when unsupported.
    """
    supplied = {(p.model_name, (p.sequence_id, p.track_id, p.issue_time_s, p.horizon_s)): p for p in predictions}
    result = {}
    for model in models:
        per_horizon = {}
        for horizon in horizons:
            keys = [k for k in truth if k[3] == horizon]
            scored = []
            for key in keys:
                p = supplied.get((model, key))
                if p is None or p.center is None or p.covariance is None:
                    continue
                err = np.array(truth[key]) - np.array(p.center)
                cov = np.array(p.covariance)
                if np.linalg.det(cov) <= 0:
                    continue
                scored.append((float(err @ np.linalg.solve(cov, err)), float(math.pi * math.sqrt(np.linalg.det(cov)))))
            out = {"support": len(keys), "scored": len(scored), "levels": {}}
            for level in levels:
                inside = sum(d2 <= CHI2_2DOF[level] for d2, _ in scored)
                out["levels"][str(level)] = {
                    "covered": inside,
                    "coverage_matched": inside / len(scored) if scored else None,
                    "coverage_all_eligible": inside / len(keys) if keys else None,
                    "mean_ellipse_area_px2": (CHI2_2DOF[level] * float(np.mean([a for _, a in scored]))
                                              if scored else None)}
            per_horizon[str(horizon)] = out
        result[model] = per_horizon
    return result


def evaluate_forecasts(annotations: list[AnnotationRecord], predictions: list[PredictionRecord], *,
                       models: list[str], horizons=PREDICTION_HORIZONS_S,
                       max_interpolation_gap_s: float = 0.5) -> dict:
    """Per-horizon errors (AERIS-005) plus interval coverage on the same eligible keys for all models."""
    errors = forecast_metrics(annotations, predictions, models=models, horizons_s=tuple(horizons),
                              max_interpolation_gap_s=max_interpolation_gap_s)
    truth = eligible_truth(annotations, tuple(horizons), max_interpolation_gap_s)
    if len(truth) != errors["eligible_keys"]:
        raise TrajectoryError("eligible truth disagrees with the AERIS-005 evaluator")
    return {"errors": errors,
            "interval_coverage": interval_coverage(truth, predictions, models, tuple(horizons)),
            "note": "Synthetic or caller-supplied data only. Coverage is empirical and conditional; "
                    "no calibration is claimed."}
