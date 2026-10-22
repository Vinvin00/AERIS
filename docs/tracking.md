# Kalman tracker (AERIS-009)

`python/aeris/tracking/tracker.py` with parameters in `configs/tracker/baseline.json`.
**No real tracking performance exists.** No identity-labeled real video has been
acquired, so every number here comes from synthetic scenes with oracle detections.

## Model

State `[cx, cy, vx, vy, w, h]` (px, px/s), original-frame pixels. One `KalmanTracker`
per sequence; `step(timestamp_s, detections)` is called once per frame in strictly
increasing time and sees only the current and past detections (there is no API that
accepts future data; a test shows prefix outputs equal the full-run outputs).

- **Prediction:** `F` uses the real delta time. `Q` is discrete white-noise
  acceleration per axis (`sigma_a^2 * [[dt^4/4, dt^3/2],[dt^3/2, dt^2]]`) and a random walk
  for width/height (`sigma_s^2 * dt`). Tests check the dt scaling and that a longer
  prediction gap grows covariance.
- **Update:** Joseph-form covariance update on `[cx, cy, w, h]`. `R` is diagonal and
  comes from box size (a fraction of w/h with a 1 px floor). Detector confidence is
  not used because no validated confidence-to-R method exists.
- **Numerical guard:** after every predict/update the covariance must be finite, symmetric
  and positive semi-definite or `TrackerError` is raised (never silently repaired).
- **Association:** per class (classes never cross), squared Mahalanobis distance of
  `[cx, cy, w, h]` innovations, gate at chi-square 0.99 with 4 dof (13.277), optimal
  assignment with SciPy `linear_sum_assignment`. Gated pairs cost "infinity".
- **Duplicates:** same-class detections in one frame with IoU >= 0.95 are merged (lower
  score dropped). Unmatched detections overlapping an existing track's predicted box at
  IoU >= 0.5 do not start a track. Limitation: two truly distinct objects with
  near-identical boxes are indistinguishable and merge.
- **Lifecycle (seconds, not frames):** `tentative` -> `confirmed` after 3 hits; an
  unmatched tentative track is deleted after 0.5 s without a measurement; an unmatched
  confirmed track becomes `lost` (coasting, `observed = false`) and is deleted after
  1.0 s without a measurement; a matched `lost` track returns to `confirmed`. The
  deletion test runs before association, so a long gap cannot revive a track. Track IDs
  are local to the sequence and never reused.
- **Output:** `TrackState` has the contract fields (sequence, ID, timestamp, bbox,
  center, velocity, covariance, lifecycle, `age_s`, `time_since_seen_s`, last
  measurement time, observed/predicted flag). The bbox is clipped to the frame; a
  predicted box entirely outside it is `null` with an `availability_reason`.
  `measurement_center_px` is the original-frame center of the exact deduplicated
  detection assigned to an observed track at this step. It is null while coasting;
  the filtered `center` remains available but is not a new detector measurement.

## Defaults are not tuned

`accel_sigma_px_s2 = 200` is loose on purpose (it follows a maneuvering target but gives
noisy velocity on a steady one); the convergence test uses a tight 2 px/s^2 model.
All values are engineering defaults to be chosen on a development set and recorded
before comparing candidates.

## Manoeuvre options (AERIS-033)

Two optional keys, absent from `baseline.json`, whose defaults reproduce the model above exactly:
`size_sigma_frac_sqrt_s` adds size noise proportional to the box (`q_size = (size_sigma_px_sqrt_s² +
(frac·size)²)·dt`), and `gate_mode: "position"` gates association on the centre only with
`gate_chi2_position` (the update still uses the box). Eleven candidate configs
`configs/tracker/manoeuvre-*.json` were compared on the QUT development split
([ADR 033](decisions/033-manoeuvre-robust-tracker.md), [QUT evaluation](qut_eval.md)). The
pre-registered rule selected `manoeuvre-a3000-s0-full` (accel σ 3000 px/s², otherwise baseline).
It passed on a six-sequence holdout but **fails the rule on the full seven-sequence holdout**
(off-target and impurity guards, mostly in S2; [ADR 034](decisions/034-tracker-guided-crop.md)),
so it stays opt-in: selectable with `--tracker-config` in the CLI and the web server, with
`baseline.json` the default. The size-aware candidates kept an approaching aircraft on fewer
tracks (8–9 fragments against 24–26) but failed the fixed-50 px guards in the last second of the
clips; see ADR 033.

The tracker also accepts `extend_only` detections (AERIS-034 guided crops): a third association
pass offers them only to confirmed or lost tracks left unmatched, so they never start or confirm a
track.

## Evidence and AC3

`tests/unit/test_tracking.py` (20 tests): crossing ID stability, a 0.3 s detection
gap with recovery under the same ID, second-based timeouts (0.1 s vs 0.6 s frame
spacing and one 5 s gap), duplicate/empty/class/gating/birth-suppression cases, PSD
covariance under random variable dt, velocity convergence, causality, input
validation. `tests/fixtures/tracking/synthetic_crossing_metrics.json` records TrackEval
HOTA/IDF1 on a synthetic crossing (clean: HOTA 0.857, IDF1 0.923; with three
missing detections: HOTA 0.758, IDF1 0.857). The clean score is below 1 because the
first two frames of each track are tentative and not reported, which is counted as
missed identity. The AERIS-004 analytic fixture also runs through tracker and
evaluator.

**AC3 and real video.** The original AC3 ("tracking metrics run on identity-labeled real
video") could not be met: no such video exists (footage source/rights and the annotation
budget are open owner decisions). By owner decision (`docs/decisions/PLAN-001-aeris-009-
ac3-split.md`) AC3 now covers only what exists: the detections -> tracker ->
`tracking_metrics` path on synthetic identity-labeled scenes and the AERIS-004 fixture, and
a test that unlabeled identities make `tracking_metrics` return `available: false` with null
HOTA/IDF1, which keeps real-performance claims blocked. The real-video half is the deferred
task AERIS-021. **No real tracking performance is claimed anywhere in this repository.**

## Limits

No acceleration or turn model, no camera-motion compensation, no appearance features,
no re-identification after deletion, no detector-confidence use, and no real-video
tuning. A tentative track that misses for more than 0.5 s is lost, which hurts very
low-recall targets. Tested on macOS arm64, Python 3.14 only.
