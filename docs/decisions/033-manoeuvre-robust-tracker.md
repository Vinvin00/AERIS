# ADR 033 — Manoeuvre-robust tracker motion model

Date: 2026-10-22. Task: AERIS-033. Plan: PLAN-011 r3 (Addendum A).

## Motivation (baseline evidence only)

On the AERIS-035 development baseline (fast profile, `baseline.json`) the target track broke
29 times. A diagnosis of that baseline output only, with no candidate run, found 23 breaks where
an existing confirmed or lost track did not take the target detection. In 21 of them the
**box-size** part of the 4-D gate failed while the position part passed: an approaching aircraft's
box grows 30–50 % per frame (for example 200 → 290 px wide), and the size random walk
(`size_sigma_px_sqrt_s` 4 px/√s) cannot follow. With the synthetic pan, 14 of 34 such breaks
failed on position. The F-35 demo clip (unlabelled, hand-held pan) shows position jumps of a
median 24 px per processed frame against a ~15 px steady-state gate.

## Model change (configuration only)

Two optional tracker keys; absent, they reproduce AERIS-009 exactly:

- `size_sigma_frac_sqrt_s` (default 0): size process noise
  `q_size = (size_sigma_px_sqrt_s² + (frac · size)²) · dt`, so noise scales with the box.
- `gate_mode` (default `full`): `position` gates association on (cx, cy) only with
  `gate_chi2_position` (chi-square 0.99, 2 dof = 9.21); the Kalman update still uses all four
  measurements.

`accel_sigma_px_s2` already exists. `configs/tracker/baseline.json` is unchanged.

## Pre-registered candidates (written before the first development run)

Grid: accel σ ∈ {200, 1000, 3000} px/s² × size fraction ∈ {0, 1.0}/√s × gate ∈ {full, position},
minus the baseline point: 11 candidates, `configs/tracker/manoeuvre-a{accel}-s{frac}-{gate}.json`,
listed in `configs/qut/candidates.json` (SHA-256 `122d50516b5a4388a6184c0a8c24c4aa01669e1f32c0030b60953f84ab678b6b` when the development runs
started). Size fraction 1.0 is the round value above the ~0.75/√s that the diagnosis implies for a
30 % per-frame growth at 25 fps. Accel 3000 was seen on the F-35 scratch replay before PLAN-011
(disclosed there). Listed order for tie-breaks: as in the candidate file.

Selection: PLAN-011 rule against the fast-profile `baseline` on the pooled development split;
synthetic-pan and fast-zoom results are secondary evidence and do not select.

## Development results (fast profile, pooled)

**development**

| Candidate | Track recall | Fragments | Off-target | Impurity | Swaps | Non-target | CV 1 s median px | PLAN-011 rule |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.1154 | 26 | 15 | 10 | 0 | 1 | 25.3 | — |
| `manoeuvre-a200-s0-position` | 0.1238 | 14 | 27 | 27 | 0 | 0 | 26.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a200-s1-full` | 0.1239 | 13 | 24 | 24 | 0 | 0 | 26.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a200-s1-position` | 0.1233 | 13 | 22 | 22 | 0 | 0 | 26.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a1000-s0-full` | 0.1183 | 24 | 17 | 12 | 0 | 2 | 25.6 | pass |
| `manoeuvre-a1000-s0-position` | 0.1251 | 9 | 28 | 28 | 0 | 0 | 26.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a1000-s1-full` | 0.1249 | 9 | 28 | 28 | 0 | 0 | 26.1 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a1000-s1-position` | 0.1251 | 9 | 28 | 28 | 0 | 0 | 26.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s0-full` | 0.1189 | 24 | 17 | 13 | 0 | 1 | 25.6 | pass |
| `manoeuvre-a3000-s0-position` | 0.1252 | 9 | 28 | 28 | 0 | 0 | 26.2 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s1-full` | 0.1245 | 10 | 29 | 29 | 0 | 0 | 26.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s1-position` | 0.1252 | 9 | 28 | 28 | 0 | 0 | 26.2 | fail: off_target_observations, target_track_impurity |

**development + synthetic pan (secondary)**

| Candidate | Track recall | Fragments | Off-target | Impurity | Swaps | Non-target | CV 1 s median px | PLAN-011 rule |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.1231 | 39 | 17 | 12 | 0 | 2 | 134.6 | — |
| `manoeuvre-a200-s0-position` | 0.1266 | 33 | 26 | 26 | 0 | 1 | 137.1 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a200-s1-full` | 0.1323 | 20 | 25 | 25 | 0 | 2 | 135.4 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a200-s1-position` | 0.1253 | 30 | 26 | 26 | 0 | 1 | 136.9 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a1000-s0-full` | 0.1294 | 24 | 13 | 10 | 0 | 1 | 134.6 | pass |
| `manoeuvre-a1000-s0-position` | 0.1357 | 10 | 25 | 25 | 0 | 0 | 135.0 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a1000-s1-full` | 0.1361 | 9 | 25 | 25 | 0 | 2 | 134.6 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a1000-s1-position` | 0.1356 | 10 | 25 | 25 | 0 | 0 | 134.9 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s0-full` | 0.1295 | 24 | 30 | 27 | 0 | 2 | 134.8 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s0-position` | 0.1363 | 8 | 25 | 25 | 0 | 0 | 134.6 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s1-full` | 0.1362 | 8 | 23 | 23 | 0 | 2 | 134.6 | fail: off_target_observations, target_track_impurity |
| `manoeuvre-a3000-s1-position` | 0.1363 | 8 | 25 | 25 | 0 | 0 | 134.6 | fail: off_target_observations, target_track_impurity |

## Selection

Two candidates pass the PLAN-011 rule on the development split: `manoeuvre-a1000-s0-full` and
`manoeuvre-a3000-s0-full` (both 24 fragments). The tie-break (fewest fragments, then highest track
recall) selects **`manoeuvre-a3000-s0-full`** (accel σ 3000 px/s², baseline size model, full gate).
On the development split it changes little: fragments 26 → 24, track recall 0.1154 → 0.1189.

The size-aware and position-gated candidates cut fragments to 8–14 and raise track recall to
about 0.125, but fail the off-target and impurity guards (about 22–29 against a bound of 20 and 15).
A diagnosis after the runs, which does not change the selection, found that every one of those
impure observations lies in the last 26 frames of a clip, where the approaching aircraft's box is
250–990 px wide, and the label lies inside the tracked box in 27–28 of 28 cases. The guards'
fixed 50 px gate counts a correctly held, very close aircraft as off-target. Re-scoring with a
size-relative gate would be a rule change after seeing results, so it is left to a separate,
reviewed planning change and an owner decision (recorded as an open issue, not acted on here).

Holdout: the selected candidate and `baseline` run once on the holdout split with this record as
the selection record (results below). Defaults stay `baseline.json` in every path; switching the
CLI, live or demo default is an owner decision (PLAN-011).

## Holdout (run once, after the selection above was written)

S2 is unavailable: six of its target labels lie outside the image (for example `v` = −19.6 and
1942.6 px at global frames 24540 and 24547), and the accepted AERIS-035 harness rejects them. The
harness is outside AERIS-033's write scope, so the holdout used S4, S6, M2, M4, M6 and M8, and
AERIS-034 (whose scope includes the harness) is to drop and count such labels and re-run the full
holdout. This was found before any holdout measure was computed.

| Run | Labelled frames | Track recall | Fragments | Off-target | Impurity | Swaps | Non-target | CV 1 s median px |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 7415 | 0.1283 | 21 | 5 | 3 | 0 | 0 | 26.9 |
| `manoeuvre-a3000-s0-full` | 7415 | 0.1310 | 21 | 7 | 6 | 0 | 1 | 27.1 |

`compare` on these files: the candidate passes the PLAN-011 rule; no sequence's track recall drops.
Under PLAN-011 the candidate may therefore be adopted, but changing any default is an owner
decision; `baseline.json` stays the default in the CLI, live and web paths.

## F-35 smoke check (unlabelled; not used for selection)

The hosted-demo clip (`f35-airshow-10fps.mp4`, 200 frames at 10 fps, hand-held pan) through the CLI
with `--profile fast-zoom`:

| Tracker | Tracks created | Frames with an observed confirmed track | Longest track (observations) |
|---|---|---|---|
| `baseline` | 55 | 62 | 10 |
| `manoeuvre-a3000-s0-full` (selected) | 6 | 153 | 91 |
| `manoeuvre-a3000-s1-full` (failed the QUT guards; shown for information) | 3 | 155 | 156 |

There are no labels, so this shows continuity, not correctness. The forecast-vs-later-detection
error is now scored on about 145 forecasts instead of 6 and is not comparable with the baseline's.

## Open issues

- The guard measures use a fixed 50 px gate, which penalises holding a very close aircraft. A
  size-relative guard needs a separate reviewed planning change and an owner decision before it
  could re-qualify the size-aware candidates; it is not applied here.
- Default change to `manoeuvre-a3000-s0-full`: owner decision.
