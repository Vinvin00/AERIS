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
`metrics.json` adds `forecast_vs_detections_displayed` and `arrow_min_speed_sigmas`. The
displayed score covers every constant-velocity horizon at each (track, issue time) whose 1 s
arrow was drawn, not only the 1 s forecast itself.

Choosing the threshold: `aeris tangier-landing-6s.mp4 --no-render` with CLI defaults
(`configs/detector/baseline.json`, YOLOX-Tiny full frame, `--min-score 0.3`), clip sha256
`0a7a60f5…a22e3`, code at this task's after-state; 1 s constant-velocity arrows scored against
later detections (AERIS-025; not ground truth). "Drawn" counts arrows the overlay draws (every
frame, coasting included); "issued at a measurement" is the subset the score can use; direction
is scored only where the target moved at least 5 px.

| k | arrows drawn | issued at a measurement | direction scored | median | 90th pct | max |
|---|---|---|---|---|---|---|
| 0 (no gate) | 55 | 36 | 23 | 8.4° | 157.4° | 175.4° |
| 1.0 | 26 | 18 | 13 | 3.5° | 7.8° | 175.4° |
| 1.5 | 15 | 11 | 11 | 3.5° | 5.7° | 8.4° |
| 2.0 | 10 | 8 | 8 | 3.0° | 4.2° | 5.7° |
| 3.0 | 2 | 2 | 2 | 4.6° | 5.5° | 5.7° |

Without the gate four scored arrows point backwards (> 90° off: t = 1.4, 1.5, 3.2, 4.6 s).
k = 1.0 removes three of them; one (t = 3.2 s, 175.4°) is still drawn, and 13 of its 26 drawn
arrows cannot be direction-scored. k = 1.5 removes all four but draws 15 arrows instead of 26.
k = 1.0 was kept for coverage, knowing it does not remove every reversal. (Corrected after
formal review, 2026-10-15: the earlier text said k = 1.0 "removes the reversed arrows" and
labelled the issued-at-measurement count as arrows shown.)
**This choice comes from one clip with one track, and it is evaluated on that same clip.**
It is not held-out evidence and does not support an accuracy claim. Re-select k on a
development set once more footage exists. The steady-state velocity standard deviation
here was about 31 px/s and is set by the untuned process noise, so changing the process
noise also changes what a given k means.

Tests: `tests/unit/test_trajectory.py::VelocityTests` (velocity and its standard deviation
against an independent hand-written filter, irregular dt and coasting),
`tests/integration/test_pipeline.py::...test_forecast_arrow_needs_a_resolved_heading` (batch;
fails with the gate disabled, k = 0) and `...test_live_loop_gates_the_arrow_like_the_batch_overlay`
(live loop: a stationary target draws no arrow, a moving one does; fails with the live gate removed).
