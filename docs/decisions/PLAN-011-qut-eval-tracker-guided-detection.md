# PLAN-011 — QUT evaluation harness, manoeuvre-robust tracker, tracker-guided crop

Date: 2026-10-22. Status: r3, approved for preparation. Review r1
(`orchestration/runs/coordination/PLAN-011-review-r1.md`) returned APPROVE WITH CHANGES with
[P1]..[P13], applied in r2. A fresh r2 review (`PLAN-011-review-r2.md`) found all thirteen resolved
and returned APPROVE WITH CHANGES with [Q1]..[Q8] (no blockers); r3 applies them (tags below),
following the PLAN-010 practice of applying post-approval changes without a further round.

## Problem

The hosted F-35 demo (`build/demo-work/runs/f35-airshow`, fast-zoom, 10 fps) finds the jet in
144 of 200 frames but the tracker splits the single jet into 55 tracks (median ~4 hits each).
A hand-held pan moves the jet a median 24 px (p90 53 px) between processed frames, while the
steady-state association gate is about 15 px. Forecasts made from 4-hit tracks are noise.
A scratch replay of the cached detections (not a repository artifact, no labels) showed
`accel_sigma_px_s2` 200 → 3000 reduces the count to 6 tracks and raises frames with an observed
confirmed track from 62 to 153 of 200. **Disclosure [P3]:** that value was seen on the F-35 clip
before this plan; any candidate near it is therefore not blind and must win on the development
split on its own. Misses are tiny distant specks, occlusion, vapour cones and underside views.
One unlabelled clip; nothing here is accuracy.

## Tasks

- **AERIS-032** QUT multi-sequence evaluation harness on all 15 published sequences (S1–S7
  single target, M1–M8 with other aircraft) at the AERIS-031 pinned revision `0bb9ece`. Measured
  size is about 29.4 MB (28.35 MB video, ~1 MB CSV) under a 40 MB cap [P13], into Git-ignored
  `data/raw/qut/`.
- **AERIS-033** manoeuvre-robust tracker motion model, selected with AERIS-032.
- **AERIS-034** tracker-guided full-resolution crop, selected with AERIS-032.

### Sources and pinning [P4]

Videos are pinned by the release's Git LFS SHA-256. Label CSVs and `target-info.csv` are plain Git
files: the release publishes only their Git blob SHA-1, which the harness verifies; it also
records each file's measured SHA-256. A sequence may have several other-aircraft CSVs (`M3_1/2`,
`M4_1..3`, `M5_1..3`, `M7_1/2`). `.DS_Store`, `._*` and `image-sequences.zip` are never fetched.
Other-aircraft rows outside the sequence's published global-frame bounds are dropped and counted.

### Pre-registered split (fixed before any variant is run)

- Development (tuning allowed): S1, S3, S5, S7, M1, M3, M5, M7.
- Holdout: S2, S4, S6, M2, M4, M6, M8. Run only for the unchanged baseline and for a candidate
  whose selection is already written down; never used to choose parameters. Every holdout run is
  recorded, including worse ones.
- Isolation [P12]: the manifest records each sequence's date and global-frame range and the harness
  checks for dev/holdout overlap on the same date. Known: M3 (dev, 22571–25044) and M4 (holdout,
  25040–27269) share five frames of one 2020-06-24 recording; all sequences come from one camera
  on two days. Recorded as a limitation, not removed. S5 was already seen in AERIS-031.

### Pre-registered measures [P1][P2]

Definitions. *Observation* = the assigned measurement centre of a track observed in that frame;
coasting (unobserved) states never count. *Gate* = the AERIS-031 50 px radius in original-frame
pixels (sequence resolution is recorded per sequence). *Unique match* = exactly one candidate in
the gate. *Labelled frame* = a frame with a target label. Score threshold = 0.3 (`TRACK_SCORE`,
the untuned CLI default); the tracker also receives detections from 0.1 to 0.3 in its low-score
pass (AERIS-027).

Per sequence and pooled. Pooling sums counts (frame-weighted); a count of sequences where a
candidate's target track recall is below baseline is also reported.

1. target detection recall — labelled frames with a unique match among detections scoring ≥ 0.3;
2. target track recall — labelled frames with a unique match among *confirmed* observations;
3. target fragments — distinct track IDs uniquely matched to the target; ID switches between
   consecutive matched frames (reported, not in the rule);
4. off-target observations [P1] — confirmed observations, in labelled frames, farther than the gate
   from every aircraft label in that frame (target and other aircraft);
5. target-track impurity [P1] — confirmed observations of a target-matched track ID, in labelled
   frames, farther than the gate from the target;
6. identity swaps [P1][Q1] — target-matched track IDs that, in some frame, have a confirmed
   observation uniquely matched to an other-aircraft label *and* farther than the gate from the
   target label of that frame (M sequences). Frames where a target and an other-aircraft label are
   within 2 × gate of each other are counted and reported as ambiguous;
7. non-target confirmed tracks — confirmed tracks never observed within the gate of any aircraft
   label. A false-alarm *proxy*: unlabelled aircraft, birds and label gaps inflate it;
8. forecast centroid error (median, p90, support) per model and horizon, issued at a frame where the
   issuing track is the target's unique confirmed match, scored against the target label at the
   target time, interpolated only between adjacent labelled frames at most 0.041 s apart (AERIS-031) [Q7]. Comparisons between variants are
   also given on the common subset of (sequence, issue frame, model, horizon) keys, with each
   side's unmatched counts [P11];
9. detection and tracking wall-clock ms/frame (p50 and p95 per frame where per-frame times exist)
   on the development machine; not a benchmark.

Every selection run uses stride 1 and source PTS timestamps; each result records both [Q5].

### Selection rule [P1][P2][P3]

`bound(b) = max(1.25 × b, b + 5)` (handles a zero baseline). A candidate passes against its
baseline on the pooled development split when: track recall ≥ baseline; fragments ≤ baseline;
off-target observations ≤ bound(baseline); impurity ≤ bound(baseline); identity swaps ≤ baseline;
non-target confirmed tracks ≤ bound(baseline). Impurity gets the same slack as off-target counts
[Q4]: a candidate that stops fragmentation keeps one track alive through manoeuvres, so the same
per-observation drift rate yields more impure observations in absolute terms; a hijack still shows
in identity swaps (zero slack) and in the per-sequence report. Ties and choice among passing candidates: fewest
fragments, then highest track recall, then fewest off-target observations, then the candidate
listed first in the task's candidate list. At most 12 candidates per task [Q7]: they are written to
`configs/qut/candidates.json` (with any synthetic-pan seed and amplitude) before the
first development run; the harness refuses a run whose variant is not on the list and stamps every
result with the list's SHA-256, so a list edited after the first run is visible as a hash change.
Every development result is recorded.

If no candidate passes on development [Q6], there is no holdout run, defaults stay unchanged and
the task reports the negative result.

Holdout consequence: the chosen candidate and the baseline each run once on the holdout. If the
candidate fails the same rule against the baseline on the holdout, it stays available but opt-in,
defaults stay unchanged, and the owner decides. Even when it passes, changing a default is an owner
decision recorded in the ADR [P8].

Detector profile [P5]: the selection baseline uses the accepted `fast` profile (YOLOX-Tiny, whole
frame, AERIS-006). `profiles.py` and the 2×2 tiling behind `fast-zoom` came from external commits
`79a7cb3` and `c480bf0` that no task owns; `fast-zoom` results are recorded as secondary evidence
because the hosted demo uses it, never as the selection baseline. Every result records profile,
detector config and weight hashes, tracker config hash and code commit.

### Camera panning [P10]

QUT is a fixed ground camera, so it does not exercise the hand-held pan behind the F-35 failure.
AERIS-032 adds an optional, development-only synthetic pan: each decoded frame is translated by a
smooth, seeded, known pixel offset (constant grey border) and labels are shifted by the same
offset (labels leaving the frame are dropped and counted). Results with the pan are a secondary
measure for AERIS-033, reported separately from the primary rule. Synthetic translation omits
rotation, blur and parallax; the real hand-held case remains unvalidated except for the
unlabelled F-35 smoke check, which is reported but never used for selection.

### AERIS-034 behaviour [P6]

- Baseline [Q2]: the tracker config AERIS-033 chose (or `baseline.json` if it chose none), `fast`
  profile, crops off, recorded with its hash — whether or not 033's choice became a default.

- Batch, live and evaluation paths interleave per frame: frame pass → tracker-guided crops →
  tracker step. The previous whole-clip detection pass stays the path when crops are off, and with
  crops off `tracks.csv` and `predictions.jsonl` must be byte-identical to the old path (tested).
- Crop centres come from the last emitted `TrackState` of each confirmed or lost track, propagated
  by its velocity to the current timestamp (past information only). Crops are model-input-sized at
  native resolution, clipped to the frame, at most 4 per frame (pre-registered), ordered by most
  recent observation.
- Crop detections may only *extend* existing confirmed or lost tracks; they never start or confirm
  a track (a new extend-only input to the tracker). Pinned details [Q3]: crop detections at or above
  the detector's 0.1 floor are kept; a crop box with IoU ≥ 0.5 against any frame-pass box is
  dropped as a duplicate; extend-only detections are offered only to tracks still unmatched after
  the high- and low-score passes; crop order ties break by `track_id`. Measure 1 stays frame-pass
  only; a crop-inclusive detection recall is reported beside it. Because crops around a false track can keep it
  alive, the off-target and non-target measures above apply.
- CameraMotion masks use frame-pass detections only, so camera-motion estimates are unchanged.
- Crop count and added ms/frame (p50/p95) are recorded. Live crops default **off**; enabling them by
  default needs the measured live cost and an owner decision. A test shows live mode with crops off
  produces the same track states as before [Q8].

### Shared files and AERIS-012 [P7][P9]

`python/aeris/cli.py` belongs to the unaccepted AERIS-012. Following AERIS-023..028 and TOOL-002,
033/034 may edit it minimally (flags, keyword arguments, manifest fields); logic lives in
`tracker.py` and `guided.py`. No 033/034 attempt runs while an AERIS-012 attempt is open (none is
open on 2026-10-22). The AERIS-012 attempt-03 reviewer must be told about these hunks. 033/034 do
not depend on AERIS-012. New tracker configs use the name pattern `configs/tracker/manoeuvre*.json`;
`baseline.json` stays outside every write scope. The CLI default tracker config stays `baseline.json`.

## Boundaries

- No change to `configs/tracker/baseline.json`, existing test expectations, gates, contracts or
  AERIS-007 baseline thresholds.
- QUT labels are approximate centroids: no AP, HOTA/IDF1 or calibrated accuracy is claimed.
- QUT frames reused in any public demo need the BSD-3-Clause notice [P13].
- No training, new weights, heavy dependency or paid compute.

Sources: [QUT release](https://open.qcr.ai/dataset/aircraft-collision-course/) (BSD-3-Clause),
[files at the pinned revision](https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/tree/0bb9ece0c70d56a56f0334dfdd867b0d7fce15f9).

## Addendum A (2026-10-22): config paths

AERIS-032 attempt 01 failed its `unit` check: `tests/unit/test_baseline.py`
(`test_no_threshold_values_are_committed_without_a_real_baseline`, AERIS-007) allows no JSON in
`configs/evaluation/` other than the regression schema, so no threshold file can appear there
before AERIS-022. The contract had placed the QUT manifest and candidate list in that folder.
That is a planning error, not an implementation one; the test is not changed. The manifest moves
to `configs/qut/sequences.json` and the candidate list to `configs/qut/candidates.json` in the
AERIS-032/033/034 write scopes, required artifacts and criteria. Nothing else changes. The
change is committed on its own (no write-scope file) so the AERIS-032 retry baseline overlays it.
