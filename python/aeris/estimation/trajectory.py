"""Causal image-space forecasts: constant-velocity and constant-acceleration models
producing contract-v1 `PredictionRecord`s from oracle-identity truth observations
(docs/trajectory_estimation.md; ADR 010).

CONTRACTS.md places the Predictor after the Tracker (`TrackStates -> Predictor`), but
`aeris.evaluation.forecast.evaluate_forecasts` (AERIS-005) already scores forecasts
against an oracle-identity protocol keyed by ground-truth track ID, independent of any
tracker or detector, precisely so forecasting accuracy can be measured apart from
association quality. This module targets exactly that protocol: it consumes truth
`AnnotationRecord`s directly (never future ones), builds each track's own causal
observation stream and forecasts from it. End-to-end scoring through tracker output is
a separate, later integration (the evaluator's own module docstring says so).

Both models are proper causal Kalman filters over a track's own position history --
never a two-frame velocity/acceleration difference (CONTRACTS.md explicitly forbids
that for the acceleration model), each with its own state and process-noise model:

- **Constant velocity** (`constant_velocity`): state `[cx, vx, cy, vy]`, discrete
  white-noise-acceleration process noise (a piecewise-constant unmodelled acceleration
  held over each step) -- the same physical assumption as `aeris.tracking.tracker`'s
  own centre model, independently implemented here since this module owns no
  dependency on tracking internals.
- **Constant acceleration** (`constant_acceleration`): state
  `[cx, vx, ax, cy, vy, ay]`, continuous white-noise-jerk process noise integrated
  exactly through its own transition matrix -- a different, independently derived
  noise model, per CONTRACTS.md's requirement that acceleration be "a separate model
  with defined state/noise assumptions."

Both need a minimum number of causal observations before their state is well
determined (velocity needs 2 positions; acceleration needs 3, since one more
observation is needed to see curvature at all). Before that a forecast's centre and
covariance are null with an explicit reason, never a numeric guess from too little
history -- this is what lets `evaluate_forecasts` count them, correctly, as missed
rather than silently scoring an unfounded early guess.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from aeris.data.contracts import PREDICTION_HORIZONS_S, AnnotationRecord, PredictionRecord

MODEL_NAMES = ("constant_velocity", "constant_acceleration")
MIN_OBSERVATIONS = {"constant_velocity": 2, "constant_acceleration": 3}
STATE_DIMS = {"constant_velocity": 4, "constant_acceleration": 6}
# Index of [centre, velocity, (acceleration)] within each axis' block; the full state
# interleaves axis blocks: constant_velocity = [cx, vx, cy, vy],
# constant_acceleration = [cx, vx, ax, cy, vy, ay].
_AXIS_DIM = {"constant_velocity": 2, "constant_acceleration": 3}

# Engineering defaults, not fitted or validated against real data (same caveat as
# aeris.tracking.tracker's fixed R -- CONTRACTS.md requires a validated detector-score
# mapping for measurement noise, and none exists).
POSITION_STD_PX = 1.0
INITIAL_VELOCITY_STD_PX_S = 1.0e3
INITIAL_ACCEL_STD_PX_S2 = 1.0e3
CV_ACCEL_STD_PX_S2 = 40.0
CA_JERK_STD_PX_S3 = 200.0
PSD_EPS = -1e-9


class TrajectoryError(ValueError):
    """A trajectory forecast was requested with invalid or non-causal inputs."""


def _symmetrize_psd(matrix: np.ndarray) -> np.ndarray:
    """Force exact symmetry and clip float-noise-scale negative eigenvalues to 0.

    Raises if an eigenvalue is negative beyond PSD_EPS: that is a genuine defect in the
    transition/process-noise math, not floating-point rounding, and must not be
    silently swallowed (CONTRACTS.md: "validate PSD/finite covariance").
    """
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    if values.min() < PSD_EPS:
        raise TrajectoryError(f"covariance: eigenvalue {values.min()!r} is negative beyond float noise")
    clipped = np.clip(values, 0.0, None)
    return vectors @ np.diag(clipped) @ vectors.T


def _cv_axis_block(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """[centre, velocity] transition and discrete white-noise-acceleration process noise."""
    f = np.array([[1.0, dt], [0.0, 1.0]])
    q = CV_ACCEL_STD_PX_S2 ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    return f, q


def _ca_axis_block(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """[centre, velocity, acceleration] transition and continuous white-noise-jerk
    process noise, integrated exactly through the transition matrix (verified against
    numeric quadrature in tests/unit/test_trajectory.py)."""
    f = np.array([[1.0, dt, dt ** 2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    q = CA_JERK_STD_PX_S3 ** 2 * np.array([
        [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
        [dt ** 3 / 6, dt ** 2 / 2, dt],
    ])
    return f, q


def _axis_block(model_name: str, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "constant_velocity":
        return _cv_axis_block(dt)
    if model_name == "constant_acceleration":
        return _ca_axis_block(dt)
    raise TrajectoryError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")


def _transition(model_name: str, dt: float) -> np.ndarray:
    f_axis, _ = _axis_block(model_name, dt)
    dim = _AXIS_DIM[model_name]
    full = np.zeros((2 * dim, 2 * dim))
    full[:dim, :dim] = f_axis
    full[dim:, dim:] = f_axis
    return full


def _process_noise(model_name: str, dt: float) -> np.ndarray:
    _, q_axis = _axis_block(model_name, dt)
    dim = _AXIS_DIM[model_name]
    full = np.zeros((2 * dim, 2 * dim))
    full[:dim, :dim] = q_axis
    full[dim:, dim:] = q_axis
    return full


def _measurement_matrix(model_name: str) -> np.ndarray:
    """Rows pick the centre component out of each axis' [centre, velocity, ...] block."""
    dim = _AXIS_DIM[model_name]
    h = np.zeros((2, 2 * dim))
    h[0, 0] = 1.0
    h[1, dim] = 1.0
    return h


def _measurement_noise() -> np.ndarray:
    return np.diag([POSITION_STD_PX ** 2, POSITION_STD_PX ** 2])


def _initial_covariance(model_name: str) -> np.ndarray:
    dim = _AXIS_DIM[model_name]
    if dim == 2:
        per_axis = [POSITION_STD_PX ** 2, INITIAL_VELOCITY_STD_PX_S ** 2]
    else:
        per_axis = [POSITION_STD_PX ** 2, INITIAL_VELOCITY_STD_PX_S ** 2, INITIAL_ACCEL_STD_PX_S2 ** 2]
    return np.diag(per_axis + per_axis)


class _CausalFilter:
    """One track's own causal Kalman filter for one model, fed one observation at a
    time in chronological order; `forecast()` never looks past the last update()."""

    def __init__(self, model_name: str, timestamp_s: float, centre: tuple[float, float]):
        if model_name not in MODEL_NAMES:
            raise TrajectoryError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")
        self.model_name = model_name
        dim = _AXIS_DIM[model_name]
        x = np.zeros(2 * dim)
        x[0], x[dim] = centre
        self.x = x
        self.P = _initial_covariance(model_name)
        self.t = timestamp_s
        self.n_observations = 1

    def update(self, timestamp_s: float, centre: tuple[float, float]) -> None:
        dt = timestamp_s - self.t
        if not dt > 0:
            raise TrajectoryError(
                f"observation timestamps must strictly increase: {timestamp_s!r} after {self.t!r}")
        f = _transition(self.model_name, dt)
        q = _process_noise(self.model_name, dt)
        x_pred = f @ self.x
        p_pred = _symmetrize_psd(f @ self.P @ f.T + q)
        h = _measurement_matrix(self.model_name)
        r = _measurement_noise()
        innovation = np.array(centre) - h @ x_pred
        s = h @ p_pred @ h.T + r
        gain = p_pred @ h.T @ np.linalg.inv(s)
        x_new = x_pred + gain @ innovation
        joseph = np.eye(len(x_pred)) - gain @ h
        p_new = _symmetrize_psd(joseph @ p_pred @ joseph.T + gain @ r @ gain.T)
        self.x, self.P, self.t = x_new, p_new, timestamp_s
        self.n_observations += 1

    def forecast(self, horizon_s: float) -> tuple[tuple[float, float] | None,
                                                   tuple[tuple[float, float], tuple[float, float]] | None,
                                                   str | None]:
        """(centre, covariance, reason): reason is set, and the rest None, exactly when
        there is not yet enough causal history to determine the model's state."""
        minimum = MIN_OBSERVATIONS[self.model_name]
        if self.n_observations < minimum:
            reason = (f"insufficient causal history for {self.model_name}: "
                      f"{self.n_observations} observation(s), need >= {minimum}")
            return None, None, reason
        f = _transition(self.model_name, horizon_s)
        q = _process_noise(self.model_name, horizon_s)
        x_f = f @ self.x
        p_f = _symmetrize_psd(f @ self.P @ f.T + q)
        h = _measurement_matrix(self.model_name)
        centre = h @ x_f
        cov = h @ p_f @ h.T
        return ((float(centre[0]), float(centre[1])),
                ((float(cov[0, 0]), float(cov[0, 1])), (float(cov[1, 0]), float(cov[1, 1]))),
                None)


def causal_observations(truth: Iterable[AnnotationRecord]) -> dict[tuple[str, str], list[tuple[float, tuple[float, float]]]]:
    """(sequence_id, track_id) -> chronological [(timestamp_s, centre_px)].

    Only labelled, non-ignored objects on timed, non-`unlabeled` frames supply
    observations -- the same rule `aeris.evaluation.forecast` uses to build its own
    truth, so this module observes exactly what that evaluator later scores against.
    """
    by_sequence: dict[str, list[AnnotationRecord]] = {}
    for record in truth:
        if not isinstance(record, AnnotationRecord):
            raise TrajectoryError("truth: expected AnnotationRecord entries")
        by_sequence.setdefault(record.frame.sequence_id, []).append(record)
    tracks: dict[tuple[str, str], list[tuple[float, tuple[float, float]]]] = {}
    for sequence_id, records in by_sequence.items():
        for record in sorted(records, key=lambda r: r.frame.frame_index):
            if record.frame.annotation_status == "unlabeled" or record.frame.timestamp_s is None:
                continue
            for item in record.objects:
                if item.ignore or item.track_id is None:
                    continue
                box = item.bbox
                centre = ((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)
                tracks.setdefault((sequence_id, item.track_id), []).append((record.frame.timestamp_s, centre))
    return tracks


def forecast_truth(truth: Iterable[AnnotationRecord], model_name: str, *,
                   horizons=PREDICTION_HORIZONS_S) -> list[PredictionRecord]:
    """Every causal forecast one model issues for every observed track in `truth`.

    At each of a track's own observed timestamps (in order), the model is updated with
    that observation first, then issues one forecast per horizon from the resulting
    filtered state -- past/current observations only (AC1), never a future one.
    """
    if model_name not in MODEL_NAMES:
        raise TrajectoryError(f"model_name must be one of {MODEL_NAMES}, got {model_name!r}")
    horizons = tuple(horizons)
    predictions = []
    for (sequence_id, track_id), observations in causal_observations(truth).items():
        filt: _CausalFilter | None = None
        for timestamp_s, centre in observations:
            if filt is None:
                filt = _CausalFilter(model_name, timestamp_s, centre)
            else:
                filt.update(timestamp_s, centre)
            for horizon_s in horizons:
                centre_f, covariance, reason = filt.forecast(horizon_s)
                availability = {} if centre_f is not None else {"center": reason, "covariance": reason}
                predictions.append(PredictionRecord(
                    schema_version=1, sequence_id=sequence_id, track_id=track_id,
                    issue_time_s=timestamp_s, horizon_s=horizon_s, target_time_s=timestamp_s + horizon_s,
                    center=centre_f, covariance=covariance, model_name=model_name,
                    availability_reason=availability))
    return predictions


def evaluate_models(truth: Iterable[AnnotationRecord], *, horizons=PREDICTION_HORIZONS_S,
                    **kwargs) -> dict[str, dict]:
    """Run every model on the same truth and score each with
    `aeris.evaluation.forecast.evaluate_forecasts` (AERIS-005, not modified here).

    Eligibility comes from truth alone, so the eligible set is identical for both
    models by construction (AC3); each model's own `scored`/`missed` split can still
    differ, honestly, since the acceleration model needs one more causal observation
    to warm up than the velocity model does.
    """
    truth = list(truth)
    from aeris.evaluation.forecast import evaluate_forecasts
    results = {}
    for model_name in MODEL_NAMES:
        predictions = forecast_truth(truth, model_name, horizons=horizons)
        results[model_name] = evaluate_forecasts(predictions, truth, horizons=horizons, **kwargs)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast every track in an AnnotationRecord JSON Lines file with both "
                    "trajectory models and report their evaluate_forecasts results.")
    parser.add_argument("annotations", type=Path, help="AnnotationRecord JSON Lines (contract v1)")
    parser.add_argument("--out", type=Path, help="directory to write predictions.jsonl and "
                                                  "evaluation.json into (not in Git)")
    args = parser.parse_args(argv)
    truth = [AnnotationRecord.from_json(line) for line in args.annotations.read_text().splitlines() if line]
    if not truth:
        print("annotations: no records found", file=sys.stderr)
        return 2
    evaluation = evaluate_models(truth)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "predictions.jsonl", "w") as handle:
            for model_name in MODEL_NAMES:
                for prediction in forecast_truth(truth, model_name):
                    handle.write(prediction.to_json() + "\n")
        (args.out / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True))
    print(json.dumps({model_name: {"eligible_total": result["eligible_total"]}
                      for model_name, result in evaluation.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
