# ADR 034 — Tracker-guided full-resolution crop

Date: 2026-10-22. Task: AERIS-034. Plan: PLAN-011 r3 ([AERIS-034 behaviour], Q2, Q3, Q8).

## Decision

`python/aeris/detection/guided.py` adds, per frame, up to four model-input-sized (416 px for
YOLOX-Tiny) windows cut at native resolution around confirmed or lost tracks. Each window is
centred on the track's last emitted `TrackState` position propagated by its velocity to the
current timestamp (past information only), shifted to lie inside the frame. Windows are ordered by
the most recent observation, ties by `track_id`. Detections in a window that touch an interior
window edge are dropped as fragments, the rest are merged by the detector's NMS, and a crop box
with IoU ≥ 0.5 against a frame-pass box is dropped as a duplicate. Crop detections at the detector's
0.1 floor and above go to the tracker as `extend_only`: a third association pass offers them only
to confirmed or lost tracks still unmatched after the high- and low-score passes, so a crop can
never start or confirm a track. Camera-motion estimation still uses frame-pass detections only.

Paths: the CLI (`--guided-crops`), the live session and web server (`--guided-crops`, default
off) and the QUT harness (candidate key `guided_crops`). With crops on, the CLI runs the tracker
inside the decode loop (crops for frame *n* come from the states of frame *n − 1*); with crops off
it runs exactly the previous code path.

## Pre-registered comparison (written before the first development run)

- Baseline [Q2]: AERIS-033's chosen config `manoeuvre-a3000-s0-full`, `fast` profile, crops off
  (its development result is `benchmarks/qut_eval/manoeuvre-a3000-s0-full-dev.json`).
- One candidate: `guided-a3000-s0-full` (same, crops on, at most 4 per frame), listed in
  `configs/qut/candidates.json` (SHA-256 `4e5505390288db0506f88bfed8eaaeace255fce02be899c7901afcfcc43f706c` at the first run).
- Rule: PLAN-011 against that baseline on the pooled development split; measure 1 stays
  frame-pass only and a crop-inclusive detection recall is reported beside it; added ms/frame
  p50/p95 reported. Synthetic pan: secondary evidence.

## Harness changes in this task

- Out-of-image target labels (S2's last six rows) are dropped and counted
  (`target_labels_outside_image`) instead of aborting the run; the default reader still rejects
  them. The alignment rule still checks the last *published* target row.
- A result written outside `benchmarks/qut_eval/` gets its own raw-error file name, so scratch
  runs cannot replace a recorded result's raw file (AERIS-035 review finding 1, residual).

## Development results (fast profile, pooled)

**Development (selects)**

| Run | Det. recall (frame pass) | Det. recall incl. crops | Track recall | Fragments | Off-target | Impurity | Swaps | Non-target | CV 1 s median px | PLAN-011 rule |
|---|---|---|---|---|---|---|---|---|---|---|
| `manoeuvre-a3000-s0-full` (baseline, crops off) | 0.1231 | 0.1231 | 0.1189 | 24 | 17 | 13 | 0 | 1 | 25.6 | — |
| `guided-a3000-s0-full` | 0.1231 | 0.1168 | 0.1179 | 24 | 44 | 31 | 0 | 1 | 25.6 | fail: target_track_recall, off_target_observations, target_track_impurity |

Crops run: 1925 over 1408 frames; extend-only matches: 47. Added time per frame: p50 0.0 ms on every sequence (most frames have no confirmed or lost track), p95 up to 9.8 ms.
**Development + synthetic pan (secondary)**

| Run | Det. recall (frame pass) | Det. recall incl. crops | Track recall | Fragments | Off-target | Impurity | Swaps | Non-target | CV 1 s median px | PLAN-011 rule |
|---|---|---|---|---|---|---|---|---|---|---|
| `manoeuvre-a3000-s0-full` (baseline, crops off) | 0.1263 | 0.1263 | 0.1295 | 24 | 30 | 27 | 0 | 2 | 134.8 | — |
| `guided-a3000-s0-full` | 0.1263 | 0.1216 | 0.1308 | 24 | 35 | 32 | 0 | 2 | 134.3 | pass |

Crops run: 1914 over 1454 frames; extend-only matches: 52. Added time per frame: p50 0.0 ms on every sequence (most frames have no confirmed or lost track), p95 up to 9.9 ms.

## Outcome

**Negative result.** The guided-crop candidate fails the PLAN-011 rule on the development split: track
recall falls slightly (0.1189 → 0.1179) and off-target and impurity counts rise beyond the bound. The
crop-inclusive detection recall is *lower* than the frame-pass recall, because a native-resolution crop
often adds a second box near the target, which makes the unique-match test fail. Only 47 of 1925 crops
extended a track: on QUT most missed frames come before any track exists, when the target is a speck
the detector cannot see even at native resolution. Per PLAN-011 there is no holdout run for this
candidate, crops stay opt-in and off by default in every path, and nothing here changes a default.

## F-35 smoke check (unlabelled; not used for selection)

`--profile fast-zoom --tracker-config configs/tracker/manoeuvre-a3000-s0-full.json`, 200 frames at 10 fps:
crops off: 6 tracks, 153 frames with an observed confirmed track, longest 91 observations; crops on: 6
tracks, 156 frames, longest 94; 198 crops, 4 extend-only matches, added 9.8 ms p50 / 19.6 ms p95 per
frame. With crops off, `tracks.csv` and `predictions.jsonl` are byte-identical to the AERIS-033 run of the
same command made before this task's code changes.

## AERIS-033 selection on the full holdout (S2 now usable)

With out-of-image target labels dropped and counted (six in S2), `baseline` and the AERIS-033
selection `manoeuvre-a3000-s0-full` were re-run on all seven holdout sequences, using ADR 033 as
the selection record (files `baseline-holdout.json`, `manoeuvre-a3000-s0-full-holdout.json`).

| Run | Det. recall (frame pass) | Det. recall incl. crops | Track recall | Fragments | Off-target | Impurity | Swaps | Non-target | CV 1 s median px | PLAN-011 rule |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.1187 | 0.1187 | 0.1129 | 24 | 8 | 5 | 0 | 2 | 26.5 | — |
| `manoeuvre-a3000-s0-full` | 0.1187 | 0.1187 | 0.1154 | 24 | 15 | 14 | 0 | 2 | 26.9 | fail: off_target_observations, target_track_impurity |

**The AERIS-033 selection fails the PLAN-011 rule on the full holdout.** Track recall still rises and
fragments are equal, but off-target observations go from 8 to 15 (bound 13) and impurity from 5 to 14
(bound 10). S2 contributes most of the rise (off-target 3 → 8, impurity 2 → 8); S2 is the sequence in
which the aircraft passes over the camera. The six-sequence holdout recorded in ADR 033 passed; this
seven-sequence result supersedes it as the AERIS-033 holdout outcome. Under PLAN-011 the candidate stays
opt-in, defaults stay `baseline.json`, and the owner decides. No further diagnosis was run on holdout data.

## AERIS-033 review follow-ups handled here

Empty or non-object tracker configs and non-numeric values now raise `TrackerError` instead of silently
running the baseline or crashing; a boundary test pins the position gate to `gate_chi2_position`
(9.21); the web server resolves a relative `--tracker-config` once at startup.
