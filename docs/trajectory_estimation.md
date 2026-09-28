# Causal image-space forecasts (AERIS-010)

Status: attempt 1, awaiting independent review.

`aeris.estimation.trajectory` turns a track's own causal position history into
image-space `PredictionRecord`s at the contract horizons (0.1/0.25/0.5/1/2 s), with two
independently specified models: constant velocity and constant acceleration. Decisions
and alternatives are in ADR 010. This module never looks at a future observation: at
each of a track's own observed timestamps, in order, it updates its state with that
observation first, then forecasts from the resulting filtered state.

```
python3 -m aeris.estimation.trajectory ANNOTATIONS.jsonl --out DIR
```

reads a plain `AnnotationRecord` JSON Lines file (contract v1), runs both models, and
writes `predictions.jsonl` (every issued `PredictionRecord`, both models) and
`evaluation.json` (`evaluate_models`'s comparable per-horizon result for each model).

## Protocol and scope

`aeris.evaluation.forecast.evaluate_forecasts` (AERIS-005) already scores forecasts
against an **oracle-identity protocol**: ground-truth track ID, issue time and horizon,
independent of any detector or tracker, precisely so forecasting accuracy can be judged
apart from association quality. This module targets exactly that protocol and consumes
`AnnotationRecord` truth directly. End-to-end scoring through a tracker's own output
(CONTRACTS.md's `TrackStates -> Predictor` boundary) is a separate, later integration;
the evaluator's own module docstring says the same for tracking through detection.

## Two independently specified models (AC1)

Both models are proper causal Kalman filters over a track's own position history --
**never a two-frame velocity or acceleration difference**, which CONTRACTS.md
explicitly rules out for the acceleration model:

- **`constant_velocity`**: state `[cx, vx, cy, vy]`. Process noise is the discrete
  white-noise-acceleration model (a piecewise-constant unmodelled acceleration held
  over each step): `Q(dt) = q_v^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]` per axis. This
  is the same physical assumption as `aeris.tracking.tracker`'s own centre model,
  independently implemented here since this module owns no dependency on tracking
  internals.
- **`constant_acceleration`**: state `[cx, vx, ax, cy, vy, ay]`. Process noise is the
  continuous white-noise-jerk model, integrated exactly through its own transition
  matrix: `Q(dt) = q_j^2 * [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2],
  [dt^3/6, dt^2/2, dt]]` per axis (`tests/unit/test_trajectory.py` cross-checks this
  closed form against independent numeric quadrature of the same transition, not just
  against the module's own code). This is a genuinely different noise assumption from
  the velocity model's, per CONTRACTS.md's requirement that acceleration be "a separate
  model with defined state/noise assumptions."

Both take a fixed measurement noise `R = diag(POSITION_STD_PX^2, POSITION_STD_PX^2)` --
an engineering default, not fitted or validated against real data, the same caveat
`aeris.tracking.tracker`'s own fixed `R` carries (CONTRACTS.md requires a validated
detector-score mapping, and none exists).

**Causality and warm-up.** A model's state is not well determined from too few
observations: velocity needs 2 positions, acceleration needs 3 (one more, to see
curvature at all). Below that minimum, `forecast()` returns a null centre and
covariance with an explicit `availability_reason` -- never a numeric guess from
insufficient history. This is deliberate: it is what lets `evaluate_forecasts` count
these as **missed**, visibly, rather than silently scoring an unfounded early forecast
(AC2). `tests/unit/test_trajectory.py::ComparableEvaluationTests.
test_missed_forecasts_from_insufficient_warmup_are_visible_not_dropped` shows the
acceleration model reports strictly more missed forecasts than the velocity model on
the identical eligible set, for exactly this reason.

## Covariance propagation and PSD safety (AC2)

Every predicted and forecast covariance is symmetrized and its eigenvalues clipped at a
float-noise tolerance (`_symmetrize_psd`), raising `TrajectoryError` if an eigenvalue is
negative beyond that tolerance -- the same numerical-stability pattern as
`aeris.tracking.tracker`'s own `_symmetrize_psd`, independently implemented.
`tests/unit/test_trajectory.py::MathTests` validates the transition and process-noise
matrices exactly against hand-derived closed forms (not derived from the module's own
code) for both models, confirms `update()`/`forecast()` actually call the PSD-safety
helper (not just the helper in isolation), and confirms `_symmetrize_psd` clips
float-scale noise but still raises on a genuine negative eigenvalue.

## Analytic fixture evidence (AC2)

Two independently computed trajectories cross-check both models, neither owned by this
task's own code:

- **`sim-cv-crossing`** (AERIS-004, read-only reuse): three real constant-velocity
  ground-truth tracks (one stationary, two crossing), with forecast targets computed
  independently by AERIS-004's own generator. The `constant_velocity` model's forecasts
  match those targets to within **0.027 px** across 75 compared (track, issue frame,
  horizon) points, and the handful of missing forecasts are exactly each track's own
  warm-up frame -- visible as `missed`, not silently absent.
- **`tests/fixtures/trajectory/constant_acceleration.jsonl`** (this task's own fixture):
  one track under exact constant acceleration (`x0=50, y0=80`, `v0=(40,-20) px/s`,
  `a=(32,16) px/s^2`, 40 frames at dyadic 1/16 s intervals so every position is exact in
  IEEE-754 double precision by construction -- `expected.json`'s `forecast_targets` are
  computed independently of `aeris.estimation.trajectory`). The `constant_acceleration`
  model converges to within **0.05 px** of the closed form after warm-up (frame index
  >= 10); the `constant_velocity` model, which cannot represent acceleration at all,
  measurably diverges as the horizon grows on this same genuinely accelerating target
  (its 2.0 s-horizon error is more than 80x its 0.1 s-horizon error at the same issue
  point) -- direct evidence the two models are not interchangeable, not merely
  differently coded.

## Comparable dual-model evaluation (AC3)

`evaluate_models(truth)` runs both models on the same truth and scores each with
`evaluate_forecasts` unmodified. Eligibility depends on truth alone, so both models'
`eligible_total` and `eligible_set_digest` are identical by construction; each model's
own `scored`/`missed` split can still differ, honestly, reflecting its own warm-up cost
and its own forecast accuracy -- never the truth or the eligible set itself.

**Real evidence**: the same 23 pinned, contiguous real AOT flight frames AERIS-009
already fetched and validated its own tracker against
(`tests/fixtures/tracking/real_samples.json`, one scored Helicopter). Both models give
`eligible_total = 71` with an identical eligible-set digest. Measured RMSE (px) and
coverage (fraction scored):

| Horizon (s) | CV eligible/scored/missed | CV RMSE px | CA eligible/scored/missed | CA RMSE px |
|---|---|---|---|---|
| 0.1 | 21 / 20 / 1 | 8.4 | 21 / 19 / 2 | 8.8 |
| 0.25 | 19 / 18 / 1 | 14.4 | 19 / 17 / 2 | 18.8 |
| 0.5 | 17 / 16 / 1 | 33.8 | 17 / 15 / 2 | 43.8 |
| 1.0 | 12 / 11 / 1 | 72.7 | 12 / 10 / 2 | 149.2 |
| 2.0 | 2 / 1 / 1 | 314.6 | 2 / 0 / 2 | unavailable (0 scored) |

This is real, non-synthetic identity-labelled motion, not a detector or tracker
deployment claim -- the oracle-identity protocol never involves either here. The
acceleration model scores no better than the velocity model on this flight (worse, in
fact, at every horizon): a single real helicopter track over 2.2 s is not enough to
either confirm or rule out an acceleration model's benefit, and its extra warm-up
observation costs it real coverage on this short a sequence. This is reported plainly,
not smoothed over -- there is no claim here that acceleration modelling helps on real
data; only that both models run correctly, honestly, on identical real evidence.

## Tests

`tests/unit/test_trajectory.py`: math (transition/process-noise closed forms including
one cross-check against independent numeric quadrature, PSD safety, growth with
horizon), causal filter behaviour (warm-up minimums, timestamp monotonicity,
convergence, covariance growth), `causal_observations` filtering (ignored objects, null
track ID, unlabeled frames, untimed frames), contract validity of every issued
`PredictionRecord`, the two analytic fixture cross-checks above, comparable dual-model
evaluation (identical eligible set, visible missed forecasts, per-horizon RMSE/coverage
reporting), and the real AOT sequence. REAL tests need the pinned AOT frames fetched for
AERIS-009 (`python -m aeris.tracking.real_sample fetch`); `AERIS_REQUIRE_REAL_FORECAST=1`
turns their skip into a failure when those frames are missing.

## Outputs

`PredictionRecord`s: `sequence_id`, `track_id`, `issue_time_s`, `horizon_s`,
`target_time_s`, `center` (original-frame px, or null), `covariance` (2x2 px², or null),
`model_name`, `availability_reason`. `evaluate_models` returns
`{"constant_velocity": ..., "constant_acceleration": ...}`, each `evaluate_forecasts`'s
own report (per-horizon RMSE, mean error, coverage, empirical interval coverage,
trajectory ADE/FDE, and the eligible-set digest).

## Limitations and follow-ups

- **Measurement noise is a fixed engineering default** (`POSITION_STD_PX`), not
  validated against real detector or annotation error, matching `aeris.tracking.tracker`'s
  own disclosed limitation.
- **The process-noise standard deviations** (`CV_ACCEL_STD_PX_S2`, `CA_JERK_STD_PX_S3`)
  are engineering defaults, not fitted to any dataset; a future task could validate them
  against real trajectory residuals.
- **No end-to-end path through a tracker's own `TrackState` output exists yet** --
  by design, matching `evaluate_forecasts`'s own oracle-identity scope; a real
  detector+tracker+forecaster pipeline is a later integration (AERIS-012).
- **The real evidence is one short (2.2 s), one-track flight.** It shows both models run
  correctly and honestly on real motion; it is not enough data to conclude whether
  acceleration modelling helps or hurts on real aircraft video in general.
- **Governance**: `orchestration/gates.json`'s `trajectory` check does not set
  `AERIS_REQUIRE_REAL_FORECAST=1` (the same gap already flagged for 006/007/008/009);
  `verify` was run with it set explicitly. Separately, this task's own REAL test adds
  roughly 12 s to the shared unit suite (reusing AERIS-009's own real-CSV-parsing cost),
  pushing the full suite to measured 131 s against the `unit` check's 120 s timeout --
  the "008/009 unit timeout" governance follow-up is no longer an occasional near-miss
  on this build; it now fails more often than not.
