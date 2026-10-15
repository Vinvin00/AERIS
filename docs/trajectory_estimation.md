# Causal image-space forecasts (AERIS-010)

`python/aeris/estimation/trajectory.py`. Two forecast models, a coverage evaluator and
per-horizon errors on shared eligible truth. **No real trajectories exist, so no real
forecast accuracy or interval calibration is claimed.** Everything below is analytic or
synthetic.

## Models

Each model is its own pair of per-axis Kalman filters (x and y independent; cross-axis
covariance is ignored, so predicted covariance is diagonal) fed the same timed measured
centers through `update(track_id, t, center, sigma_px)`. Neither reads tracker state,
later detections or truth.

| Model | Per-axis state | Process noise (continuous, integrated over the actual dt) |
|---|---|---|
| `constant_velocity` | position, velocity | white-noise acceleration, density `q` px^2/s^3 |
| `constant_acceleration` | position, velocity, acceleration | white-noise jerk, density `q` px^2/s^5 |

The acceleration model is a separately specified filter with its own state and noise, not a
difference of two noisy velocity estimates. Updates use the Joseph form with a PSD check.
Initial velocity and acceleration uncertainty are explicit parameters. Defaults
(`DEFAULTS`) are loose, **untuned** engineering values.

`forecast(track_id, issue_time_s)` returns one contract `PredictionRecord` per horizon
0.1, 0.25, 0.5, 1 and 2 s (a different horizon fails contract validation): center and 2x2
covariance in original-frame pixels, `target = issue + horizon`. The state is propagated by
`issue - last_measurement + horizon` seconds, so a short coast widens the covariance.

## Causality and misses (AC1, AC2)

`update` is atomic: both axes are computed first and committed together, so a failure (for example a PSD violation on extreme input) leaves the track unchanged. It rejects non-increasing timestamps; `forecast` raises if the issue time precedes a
measurement already received; a test shows that corrupting every measurement after the
issue time changes no earlier forecast. A forecast that cannot be made (unknown track, fewer
than 2 / 3 updates, last measurement older than 1 s) is **still returned**, with a null
center/covariance and a reason, and a track with no records at all is a miss too:
`evaluate_forecasts` counts them as `missed` (never silently improving the error) and
`coverage_all_eligible` treats them as not covered.

## Analytic validation (AC2)

Tests compare against closed forms and independent numerics: a noiseless line and a
parabola are forecast exactly at all five horizons; the position-velocity covariance
propagation equals the hand-derived formulas; both process-noise matrices equal a numerical
integration of the continuous model; covariance stays PSD and grows with horizon under random
delta times. (An early draft used the discrete rather than continuous noise form for the
velocity model; the numerical-integration test caught the inconsistency.)

## Evaluation (AC3)

`evaluate_forecasts` runs the AERIS-005 `forecast_metrics` (support, matched, missed, RMSE,
mean error per horizon on identical eligible keys for every model) and adds
`interval_coverage`: the fraction of scored forecasts whose truth lies inside the
2-D Gaussian ellipse at 50/90/95 % (chi-square 2 dof), the mean ellipse area, and the
coverage over all eligible keys. The module re-derives eligible truth with the AERIS-005 rule
(`eligible_truth`) and raises if its key count disagrees with the evaluator; a test also checks
per-horizon support. ADE/FDE are not reported: forecasts are independent per-horizon points.

`tests/fixtures/trajectory/synthetic_forecast_summary.json` records a synthetic run
(constant-velocity A, constant-acceleration B, 1 px noise, 70 frames at 10 fps). Findings
that the tests pin:

- With tight models matched to the motion, constant velocity beats acceleration on steady
  motion at every horizon (1 s RMSE 3.0 vs 26.0 px) and acceleration beats constant velocity on
  accelerating motion at 1 and 2 s (26.8 vs 57.0 px at 1 s).
- A model that cannot represent the motion is overconfident: constant velocity on B has
  0 % empirical 90 % coverage at 1 s.
- **The acceleration model is not generally better.** With the loose default noise it chases
  measurement noise and loses to constant velocity even on accelerating motion (1 s RMSE 68.1
  vs 27.6 px), and with tight noise its RMSE on steady motion is dominated by the first few
  updates, while its large early uncertainty keeps coverage high. Choose noise on a
  development split before comparing candidates.

## Limits

No real data; no calibrated uncertainty (empirical coverage on synthetic scenes only);
axis-independent filters; no maneuver detection, no camera-motion compensation; defaults and
the 1 s staleness limit are untuned; max interpolation gap and eligibility come from
AERIS-005. Tested on macOS arm64, Python 3.14.

## Forecasts scored against later detections (AERIS-025)

`forecast_vs_detections` scores each forecast against the same track's later assigned
detection centers, so settings can be compared on unlabeled footage. It is written to
`metrics.json` on every CLI run. Per model and horizon it reports position error
(median, RMSE) and direction error: the angle between the forecast displacement and the
measured displacement from the issue-time measurement, scored only when the target moved
at least `min_motion_px` (5 px, untuned). Origins are exact measurements; targets may be
interpolated across gaps up to 0.5 s, never extrapolated. **It is not ground truth:**
detector error and identity switches leak in, and only forecasts whose track survived to
the target time are scored. It does not replace labeled evaluation.

## Camera motion (AERIS-028)

With `--camera-motion estimate` (and always in the web live view), `CameraMotion`
(`python/aeris/estimation/camera_motion.py`) fits a RANSAC similarity to Lucas-Kanade tracks of
background corners between consecutive processed frames, with detection boxes masked out, and
chains them into a transform from each frame to the first processed frame. A separate
constant-velocity forecaster runs on measurements in that reference frame and drives the arrow
and heading gate; the arrow is mapped back onto the current frame. `predictions.jsonl` and the
label-based evaluators are unchanged (original-frame pixels). `metrics.json` adds
`forecast_vs_detections_scene` and `camera_motion` counts. A step with too few background
corners is counted as failed and treated as no motion, which equals the uncompensated case.

