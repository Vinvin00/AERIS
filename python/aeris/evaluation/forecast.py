"""Causal image-space forecast evaluation with explicit support, misses and availability.

Protocol (docs/evaluation.md):
- Oracle-identity protocol: forecasts are keyed by ground-truth track ID, issue time and
  horizon. That scores forecasting apart from association; end-to-end scoring through
  tracker matches is outside this adapter.
- Track observations come from labelled objects on ``complete`` and ``partial`` frames
  (a labelled object is known even when others are not). ``ignore`` objects and
  ``unlabeled`` frames never supply truth.
- The eligible set comes from truth alone, so it is the same for every model. It holds
  each (track, issue time, horizon) where the track is annotated at the issue time and
  truth at issue time + horizon can be interpolated linearly between two annotated
  observations at most ``max_gap_s`` apart. There is no extrapolation beyond the last
  observation. Interpolated truth is evaluator-only.
- A missing forecast (absent, or null centre) counts as missed. Errors are reported only
  next to coverage and the eligible-set digest, so dropping hard cases cannot silently
  improve a score. Forecasts outside the eligible set are counted by reason and never
  scored. One issued before the track's first annotated observation is a future-leakage
  suspect.
- Interval coverage uses the chi-square(2) quantile -2 ln(1 - p). Sequence-bootstrap RMSE
  intervals need at least MIN_SEQUENCES_FOR_INTERVALS sequences.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math

from aeris.data.contracts import PREDICTION_HORIZONS_S, AnnotationRecord, PredictionRecord
from aeris.evaluation._common import (
    MIN_SEQUENCES_FOR_INTERVALS, EvaluationError, ratio, sequence_bootstrap,
)

DEFAULT_MAX_GAP_S = 0.5
COVERAGE_LEVELS = (0.5, 0.9, 0.95)


def _tracks(truth) -> tuple[dict, dict]:
    """(sequence, track) -> [(time, centre)] from annotated frames; excluded sequences."""
    sequences = {}
    for record in truth:
        if not isinstance(record, AnnotationRecord):
            raise EvaluationError("truth: expected AnnotationRecord entries")
        sequences.setdefault(record.frame.sequence_id, []).append(record)
    tracks, excluded = {}, {}
    for sequence_id, records in sorted(sequences.items()):
        records.sort(key=lambda r: r.frame.frame_index)
        if any(r.frame.timestamp_s is None for r in records):
            excluded[sequence_id] = "untimed frames: forecast horizons need timestamps"
            continue
        found = False
        for record in records:
            if record.frame.annotation_status == "unlabeled":
                continue
            for item in record.objects:
                if item.ignore or item.track_id is None:
                    continue
                box = item.bbox
                tracks.setdefault((sequence_id, item.track_id), []).append(
                    (record.frame.timestamp_s, ((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)))
                found = True
        if not found:
            excluded[sequence_id] = "no identity-labelled objects"
    return tracks, excluded


def _target(observations, target_time, max_gap_s, tolerance):
    """Interpolated truth centre at target_time, or (None, reason)."""
    times = [t for t, _ in observations]
    index = bisect_left(times, target_time - tolerance)
    if index < len(times) and abs(times[index] - target_time) <= tolerance:
        return observations[index][1], None
    if index == len(times):
        return None, "target_beyond_labelled_span"
    (t0, (x0, y0)), (t1, (x1, y1)) = observations[index - 1], observations[index]
    if t1 - t0 > max_gap_s + tolerance:
        return None, "target_in_gap_longer_than_max"
    weight = (target_time - t0) / (t1 - t0)
    return (x0 + (x1 - x0) * weight, y0 + (y1 - y0) * weight), None


def _rmse(errors: list[float]) -> float:
    return math.sqrt(math.fsum(e * e for e in errors) / len(errors))


def evaluate_forecasts(predictions, truth, *, horizons=PREDICTION_HORIZONS_S,
                       max_gap_s: float = DEFAULT_MAX_GAP_S, coverage_levels=COVERAGE_LEVELS,
                       min_sequences_for_intervals: int = MIN_SEQUENCES_FOR_INTERVALS,
                       bootstrap_resamples: int = 1000, seed: int = 0,
                       time_tolerance_s: float = 1e-9) -> dict:
    """Per-horizon support, coverage, misses, errors and interval coverage for one model."""
    horizons = tuple(horizons)
    if not horizons or any(h not in PREDICTION_HORIZONS_S for h in horizons):
        raise EvaluationError(f"horizons must be drawn from {PREDICTION_HORIZONS_S}")
    if not max_gap_s > 0 or not all(0 < level < 1 for level in coverage_levels):
        raise EvaluationError("max_gap_s must be positive and coverage levels in (0, 1)")
    tracks, excluded = _tracks(truth)
    eligible, ineligible = {}, Counter()
    for (sequence_id, track_id), observations in tracks.items():
        for index, (issue_time, _) in enumerate(observations):
            for horizon in horizons:
                centre, reason = _target(observations, issue_time + horizon, max_gap_s, time_tolerance_s)
                if centre is None:
                    ineligible[(horizon, reason)] += 1
                else:
                    eligible[(sequence_id, track_id, index, horizon)] = {
                        "issue_time_s": issue_time, "target": centre}
    models = {p.model_name for p in predictions if isinstance(p, PredictionRecord)}
    if len(models) > 1:
        raise EvaluationError(f"one model per evaluation; got {sorted(models)}")
    scored, null_centre, unscored, suspects, seen = {}, set(), Counter(), 0, set()
    for prediction in predictions:
        if not isinstance(prediction, PredictionRecord):
            raise EvaluationError("predictions: expected PredictionRecord entries")
        raw_key = (prediction.sequence_id, prediction.track_id, prediction.issue_time_s,
                   prediction.horizon_s)
        if raw_key in seen:
            raise EvaluationError(f"duplicate forecast for {raw_key}")
        seen.add(raw_key)
        observations = tracks.get((prediction.sequence_id, prediction.track_id))
        if observations is None:
            unscored["unknown_or_excluded_track"] += 1
            continue
        times = [t for t, _ in observations]
        index = bisect_left(times, prediction.issue_time_s - time_tolerance_s)
        if index == len(times) or abs(times[index] - prediction.issue_time_s) > time_tolerance_s:
            if prediction.issue_time_s < times[0] - time_tolerance_s:
                unscored["issued_before_track_first_observed"] += 1
                suspects += 1
            else:
                unscored["issue_time_not_an_observation"] += 1
            continue
        if prediction.horizon_s not in horizons:
            unscored["horizon_not_evaluated"] += 1
            continue
        key = (prediction.sequence_id, prediction.track_id, index, prediction.horizon_s)
        if key in scored or key in null_centre:
            raise EvaluationError(f"duplicate forecast for {key}")
        if key not in eligible:
            _, reason = _target(observations, times[index] + prediction.horizon_s, max_gap_s,
                                time_tolerance_s)
            unscored[reason] += 1
            continue
        if prediction.center is None:
            null_centre.add(key)
            continue
        target = eligible[key]["target"]
        residual = (prediction.center[0] - target[0], prediction.center[1] - target[1])
        scored[key] = {"error": math.hypot(*residual), "residual": residual,
                       "covariance": prediction.covariance}
    per_horizon = {}
    for horizon in horizons:
        keys = sorted(k for k in eligible if k[3] == horizon)
        errors = [scored[k]["error"] for k in keys if k in scored]
        reasons: dict[str, str] = {}
        entry = {"eligible": len(keys), "scored": len(errors),
                 "missed": len(keys) - len(errors),
                 "missed_null_centre": sum(k in null_centre for k in keys),
                 "ineligible": {reason: n for (h, reason), n in sorted(ineligible.items()) if h == horizon}}
        entry["coverage"] = ratio(len(errors), len(keys), "coverage", reasons,
                                  "no eligible truth at this horizon")
        no_scores = "no scored forecasts at this horizon"
        entry["rmse_px"] = _rmse(errors) if errors else None
        entry["mean_error_px"] = math.fsum(errors) / len(errors) if errors else None
        if not errors:
            reasons.update(rmse_px=no_scores, mean_error_px=no_scores)
        groups = {}
        for k in keys:
            if k in scored:
                groups.setdefault(k[0], []).append(scored[k]["error"])
        entry["rmse_px_interval"] = sequence_bootstrap(
            groups, _rmse, resamples=bootstrap_resamples, seed=seed,
            min_sequences=min_sequences_for_intervals)
        entry["uncertainty"] = _coverage([scored[k] for k in keys if k in scored], coverage_levels)
        entry["availability_reason"] = reasons
        per_horizon[str(horizon)] = entry
    trajectory_errors, trajectory_eligible = [], 0
    for (sequence_id, track_id, index) in sorted({k[:3] for k in eligible}):
        keys = [(sequence_id, track_id, index, h) for h in horizons]
        if not all(k in eligible for k in keys):
            continue
        trajectory_eligible += 1
        if all(k in scored for k in keys):
            trajectory_errors.append([scored[k]["error"] for k in keys])
    trajectory_reasons = {}
    trajectory = {"eligible": trajectory_eligible, "support": len(trajectory_errors),
                  "missed": trajectory_eligible - len(trajectory_errors),
                  "coverage": ratio(len(trajectory_errors), trajectory_eligible, "coverage",
                                    trajectory_reasons,
                                    "no issue point with every evaluated horizon eligible"),
                  "ade_px": math.fsum(math.fsum(e) / len(e) for e in trajectory_errors)
                  / len(trajectory_errors) if trajectory_errors else None,
                  "fde_px": math.fsum(e[-1] for e in trajectory_errors) / len(trajectory_errors)
                  if trajectory_errors else None,
                  "definition": "eligible: issue points whose every evaluated horizon has truth; "
                                "ADE/FDE average over those with every horizon scored (support); "
                                "FDE uses the longest horizon",
                  "availability_reason": trajectory_reasons}
    if not trajectory_errors:
        trajectory_reasons.update(ade_px="no issue point with every horizon scored",
                                  fde_px="no issue point with every horizon scored")
    digest_keys = sorted([k[0], k[1], repr(v["issue_time_s"]), k[3]] for k, v in eligible.items())
    reasons = {}
    if not tracks:
        reasons["per_horizon"] = "no timed identity-labelled truth: forecast metrics are unavailable"
    if not models:
        reasons["model_name"] = "no predictions were supplied"
    return {
        "protocol": "oracle ground-truth identity; truth interpolation evaluator-only, no extrapolation",
        "model_name": models.pop() if models else None,
        "horizons": list(horizons), "max_gap_s": max_gap_s,
        "sequences": {"evaluated": sorted({k[0] for k in tracks}), "excluded": excluded},
        "eligible_total": len(eligible),
        "eligible_set_digest": hashlib.sha256(json.dumps(digest_keys).encode()).hexdigest(),
        "per_horizon": per_horizon if tracks else None,
        "trajectory": trajectory,
        "unscored_predictions": dict(sorted(unscored.items())),
        "future_leakage_suspects": suspects,
        "availability_reason": reasons,
    }


def _coverage(items: list[dict], levels) -> dict:
    """Empirical coverage of Gaussian ellipses from forecast covariances at stated levels."""
    usable, degenerate = [], 0
    for item in items:
        covariance = item["covariance"]
        if covariance is None:
            continue
        (a, b), (_, d) = covariance
        determinant = a * d - b * b
        if determinant <= 1e-12 * max(a * d, 1.0):
            degenerate += 1
            continue
        x, y = item["residual"]
        usable.append(((d * x * x - 2 * b * x * y + a * y * y) / determinant, determinant))
    result = {"with_covariance": len(usable) + degenerate, "degenerate_covariance": degenerate,
              "levels": {}}
    for level in levels:
        quantile = -2 * math.log(1 - level)
        reasons = {}
        result["levels"][str(level)] = {
            "chi2_quantile": quantile,
            "empirical_coverage": ratio(sum(m <= quantile for m, _ in usable), len(usable),
                                        "empirical_coverage", reasons,
                                        "no forecasts with a usable covariance"),
            "mean_ellipse_area_px2": ratio(math.fsum(math.pi * quantile * math.sqrt(det) for _, det in usable),
                                           len(usable), "mean_ellipse_area_px2", reasons,
                                           "no forecasts with a usable covariance"),
            "availability_reason": reasons}
    return result
