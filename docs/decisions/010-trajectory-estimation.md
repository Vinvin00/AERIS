# ADR 010 — Constant-velocity and constant-acceleration causal forecasts

Status: proposed for AERIS-010 attempt 1 review. Date: 2026-09-27.

Context: AERIS-010 turns a track's own causal position history into image-space
forecasts at the contract horizons, with two models CONTRACTS.md requires be
independently specified ("Acceleration is a separate model with defined state/noise
assumptions, not a noisy two-frame difference"), scored comparably on identical
eligible truth. AERIS-005's `evaluate_forecasts` already implements the scoring
protocol (oracle-identity, truth-only eligibility, visible misses); this task supplies
the two forecasters that feed it.

Decision:

- **Protocol: oracle-identity, matching AERIS-005's evaluator exactly.**
  `aeris.evaluation.forecast.evaluate_forecasts` scores forecasts keyed by ground-truth
  track ID, issue time and horizon, independent of any detector or tracker, "so that
  forecasting accuracy can be judged apart from association quality" (its own module
  docstring). This module consumes `AnnotationRecord` truth directly and builds each
  track's own causal observation stream from it, rather than from a tracker's
  `TrackState` output. An end-to-end path through a tracker is a distinct, later
  integration; scoring it here would blur forecasting error together with tracker
  association error, exactly what the oracle-identity protocol exists to avoid.
- **Both models are proper causal Kalman filters over a track's own position
  history**, never a two-frame velocity or acceleration difference:
  - **`constant_velocity`**: state `[cx, vx, cy, vy]`, discrete white-noise-acceleration
    process noise (`q_v^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]` per axis) -- the same
    physical assumption as `aeris.tracking.tracker`'s own centre model. It is
    independently implemented here rather than imported, since this task owns no
    dependency on tracking internals and the model is only four states (no size).
  - **`constant_acceleration`**: state `[cx, vx, ax, cy, vy, ay]`, continuous
    white-noise-jerk process noise integrated exactly through its own 3x3 transition
    matrix (`q_j^2 * [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2],
    [dt^3/6, dt^2/2, dt]]` per axis), independently re-derived and cross-checked in
    tests against numeric quadrature of the same transition -- a genuinely different
    noise model from the velocity model's piecewise-constant-noise form, not merely one
    more state appended with the same assumption.
  - Both use a fixed measurement noise `R`, an engineering default (CONTRACTS.md
    requires a validated detector-score-to-`R` mapping; none exists, matching
    `aeris.tracking.tracker`'s own disclosed limitation).
- **Warm-up minimums, not naive early forecasts.** A model's state is not observable
  from too few positions: velocity needs 2, acceleration needs 3 (curvature needs a
  third point to show at all). Below the minimum, `forecast()` returns a null centre
  and covariance with an explicit reason. This directly implements CONTRACTS.md's ban
  on "a noisy two-frame difference": rather than compute a shaky two-point acceleration
  estimate and report it as if it were meaningful, the model reports nothing, visibly,
  until it has enough independent information to be one.
- **Numerical stability**: covariance propagation is symmetrized and eigenvalue-clipped
  at a float-noise tolerance every predict/update/forecast step (`_symmetrize_psd`),
  raising on a genuine negative eigenvalue -- the same pattern as
  `aeris.tracking.tracker`'s own safety check, independently implemented.
- **Real evidence (AC3), same real flight AERIS-009 already validated its tracker
  against** (23 pinned contiguous AOT frames, one scored Helicopter). Both models give
  an identical eligible set (`eligible_total=71`, matching digest) by construction; per
  the measured table in `docs/trajectory_estimation.md`, the acceleration model does not
  score better than the velocity model on this one short (2.2 s), one-track flight --
  its extra warm-up observation costs it visible coverage, and its RMSE is higher at
  every horizon. This is reported as the honest, plain result: one short real sequence
  cannot establish whether acceleration modelling helps on real aircraft video, and no
  claim to that effect is made here.
- **Analytic fixtures, two independent trajectories.** AERIS-004's own `sim-cv-crossing`
  (read-only reuse, not modified) cross-checks the velocity model against forecast
  targets that module computed independently: max error 0.027 px across 75 compared
  points. This task's own `tests/fixtures/trajectory/constant_acceleration.jsonl` (one
  track, exact constant acceleration, dyadic frame times chosen so every position is
  exact in IEEE-754 double precision by construction) cross-checks the acceleration
  model similarly: converges to within 0.05 px after warm-up, while the velocity model
  measurably diverges on the same genuinely accelerating target as the horizon grows --
  direct evidence the two models behave differently, not just that they are coded
  differently.

Alternatives rejected:

- **Deriving acceleration from a two-frame velocity difference.** CONTRACTS.md
  explicitly forbids exactly this ("not a noisy two-frame difference"): differencing
  two already-noisy velocity estimates amplifies noise quadratically and gives no
  principled uncertainty estimate. A proper filter with its own state and process noise
  gives both a point estimate and a covariance with a defined provenance.
  - **Reusing `aeris.tracking.tracker`'s `_KalmanTrack` internals directly** for the
    velocity model. Rejected: that class carries a `[cx,cy,vx,vy,w,h]` state and a
    tracker-specific association lifecycle this module has no use for, and AERIS-010's
    write scope owns no dependency on tracking internals; a small, independently
    specified four-state filter is simpler to reason about and to validate on its own.
  - **Scoring forecasts through the tracker's own `TrackState` output** instead of
    oracle-identity truth. Rejected for this task: AERIS-005's evaluator already exists
    specifically to separate forecasting accuracy from association quality; building an
    end-to-end path now would conflate the two and duplicate scope that belongs to a
    later Python-integration task (AERIS-012).
  - **A single combined state covering both models** (e.g., always carrying
    acceleration and simply not using it for the velocity forecast). Rejected: it would
    let the "velocity" model's covariance silently absorb acceleration uncertainty it
    is not supposed to represent, undermining the comparability AC3 requires between two
    genuinely different, separately specified models.
  - **Naive per-horizon forecasts as soon as any observation exists** (reporting a
    forecast from a single point, e.g. assuming zero velocity). Rejected: a single
    position carries no velocity information at all; reporting a number there would be
    exactly the fabricated-confidence failure AGENTS.md and CONTRACTS.md both warn
    against ("Unknown measurements stay null with a reason").

Consequences:

- The process-noise standard deviations (`CV_ACCEL_STD_PX_S2`, `CA_JERK_STD_PX_S3`) and
  the fixed measurement noise (`POSITION_STD_PX`) are engineering defaults, not fitted
  or validated against real trajectory residuals; a future task could validate them
  against a larger real dataset once one exists.
- No end-to-end forecast-through-tracker path exists yet; AERIS-012 (integrated Python
  MVP) is the natural place to wire `aeris.tracking.tracker`'s `TrackState` output into
  a forecaster, at which point a *new*, separate evaluation protocol decision is needed
  (this module's oracle-identity scoring does not by itself validate that path).
- `orchestration/gates.json`'s `trajectory` check does not yet set
  `AERIS_REQUIRE_REAL_FORECAST=1` (the same governance gap already flagged for
  006/007/008/009); `verify` was run with it set explicitly.
- This task's own REAL test reuses AERIS-009's real-CSV-parsing cost (~12 s) and pushes
  the shared unit suite to a measured 131 s against the `unit` check's 120 s timeout on
  this build -- escalating the "008/009 unit timeout" governance follow-up from an
  occasional near-miss to a consistent overage. Not fixed here: `orchestration/gates.json`
  and `tools/orchestrate.py` are outside this task's write scope, and AGENTS.md forbids
  changing gates to make one's own task pass.
