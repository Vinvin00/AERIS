# AERIS-025 — Forecasts scored against later detections

Status: implementation for independent review. Date: 2026-10-14.

Forecast settings (measurement noise, process noise, display gating) could not be compared
because no labeled real video exists. `forecast_vs_detections` in
`python/aeris/estimation/trajectory.py` scores each causal forecast against the same
track's later tracker-assigned detection centers and is written to `metrics.json` on every
CLI run, labels or not.

Choices:
- It lives in the trajectory module, not `evaluation/metrics.py`, because that evaluator is
  hash-pinned by `benchmarks/baseline_manifest.json` and this task must not move a gate.
- Only forecasts issued at an exact measurement are scored; that measurement is the origin
  for direction error. Targets may be interpolated across gaps up to 0.5 s and are never
  extrapolated. Direction is scored only when the measured displacement is at least 5 px
  (untuned), because heading is undefined for a near-stationary target.
- Median and 90th-percentile direction error are reported because single reversed
  forecasts would dominate a mean.

Limits: not ground truth. Detector error, identity switches and track loss leak in, and
scoring is survivor-biased toward tracks that stayed measured. Image-space only, so a
panning camera changes what "direction" means. One smoke run on
`tangier-landing-6s.mp4` (1 track, 23 scored 1 s forecasts) produced numbers; that is a
plumbing check, not a result.

Tests: hand-computed cases in `tests/unit/test_trajectory.py::ForecastVsDetectionTests`
and a constant-velocity synthetic clip in `tests/integration/test_pipeline.py`.
