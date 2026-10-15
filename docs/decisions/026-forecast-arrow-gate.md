# AERIS-026 — Draw the forecast arrow only when heading is resolved

Status: implementation for independent review. Date: 2026-10-15.

The 1 s constant-velocity forecast arrow was drawn for every confirmed track as soon as the
forecaster had two updates. On new tracks, and when a panning camera holds the aircraft
nearly still in the image, the velocity estimate is mostly noise, so the arrow pointed in
arbitrary directions, sometimes backwards.

`TrajectoryForecaster.velocity` now returns the velocity and its per-axis standard deviation
propagated to the issue time. `cli.heading_resolved` draws the arrow only when
`speed >= ARROW_MIN_SPEED_SIGMAS * max(sigma_vx, sigma_vy)`. The same gate is used by the batch
overlay and the live demo. Forecast records are unchanged; only drawing is gated.
`metrics.json` adds `forecast_vs_detections_displayed` (the AERIS-025 score over drawn
arrows only) and `arrow_min_speed_sigmas`.

Choosing the threshold, on `tangier-landing-6s.mp4` with constant velocity at 1 s, scored
against later detections (AERIS-025; not ground truth):

| k | arrows shown | direction scored | median error | 90th percentile |
|---|---|---|---|---|
| 0 (no gate) | 36 | 23 | 8.4° | 157.4° |
| 1.0 | 18 | 13 | 3.5° | 7.8° |
| 1.5 | 11 | 11 | 3.5° | 5.7° |
| 2.0 | 8 | 8 | 3.0° | 4.2° |
| 3.0 | 2 | 2 | 4.6° | 5.5° |

k = 1.0 removes the reversed arrows and keeps half of them; larger values mostly hide more.
**This choice comes from one clip with one track, and it is evaluated on that same clip.**
It is not held-out evidence and does not support an accuracy claim. Re-select k on a
development set once more footage exists. The steady-state velocity standard deviation
here was about 31 px/s and is set by the untuned process noise, so changing the process
noise also changes what a given k means.

Tests: `tests/unit/test_trajectory.py::VelocityTests` and
`tests/integration/test_pipeline.py::...test_forecast_arrow_needs_a_resolved_heading`.
The latter fails with the gate disabled (k = 0).
