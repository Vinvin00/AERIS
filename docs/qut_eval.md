# QUT multi-sequence evaluation (AERIS-035)

A labelled-video check for detector and tracker changes. It replays all 15 published QUT
aircraft sequences, keeps every runtime output fixed, and only then compares them with the
publisher's approximate target centroids. Measures, the development/holdout split and the
selection rule are pre-registered in
[PLAN-011](decisions/PLAN-011-qut-eval-tracker-guided-detection.md). This is descriptive
evidence from one fixed ground camera, not a benchmark.

## Reproduce

```sh
.venv/bin/python scripts/qut_eval.py fetch                       # ~29 MB into data/raw/qut
.venv/bin/python scripts/qut_eval.py run --split dev --name baseline
.venv/bin/python scripts/qut_eval.py run --split dev --name baseline --pan
.venv/bin/python scripts/qut_eval.py compare benchmarks/qut_eval/baseline-dev.json benchmarks/qut_eval/OTHER-dev.json
```

`fetch` downloads only the files pinned in
[`configs/qut/sequences.json`](../configs/qut/sequences.json) at release
revision `0bb9ece`: videos by their published Git LFS SHA-256, label CSVs and `target-info.csv`
by their published Git blob SHA-1 (the release publishes no SHA-256 for them). Rights:
BSD-3-Clause per the [publisher](https://open.qcr.ai/dataset/aircraft-collision-course/); QUT
frames reused in a public demo need that notice. Media stays in Git-ignored `data/raw/qut/`.

`run --name` accepts only variants listed in
[`configs/qut/candidates.json`](../configs/qut/candidates.json); the
profile and tracker config come from that list, and every result records the list's SHA-256,
the code commit and whether the working tree was dirty, the detector config and weight hashes,
the tracker config hash, stride 1 and PTS timestamps. Each list entry also pins its tracker
config's SHA-256, so a listed config cannot be edited silently. Holdout runs need `--selection-record`
naming a decision record in `docs/decisions/` (its path and SHA-256 are stored), refuse the
synthetic pan, and never overwrite an earlier holdout result. A `--sequences` subset is written
under its own name so it never replaces a full-split result. Frame-pass detections are cached
in Git-ignored `data/processed/qut_cache/`, keyed by video hash, profile, detector config,
weights and pan, so tracker-only variants do not re-run the detector. Per-forecast errors for
`compare` go to Git-ignored `data/processed/qut_eval_raw/`; each report stores that file's SHA-256
and `compare` refuses a file another run has replaced.

## What is measured

Per sequence and pooled (counts summed, so pooling is frame-weighted). The gate is the AERIS-031
50 px radius in original-frame pixels (all sequences are 1280 × 720); an *observation* is the
assigned measurement centre of an observed track, never a coasting state; detections count at
score ≥ 0.3, while the tracker also uses 0.1–0.3 detections to extend existing tracks.

| Measure | Meaning |
|---|---|
| target detection recall | labelled frames with exactly one detection within the gate |
| target track recall | labelled frames with exactly one confirmed observation within the gate |
| target fragments / ID switches | track IDs that carried the target; ID changes between matched frames |
| off-target observations | confirmed observations in labelled frames farther than the gate from every aircraft label |
| target-track impurity | confirmed observations of a target-carrying track farther than the gate from the target |
| identity swaps | target-carrying tracks that, in some frame, uniquely match an other-aircraft label while outside the target's gate. Frames with no target label count too (the observation cannot be the target there), a stricter reading of PLAN-011. Labelled target frames with an other-aircraft label within 100 px are counted as ambiguous |
| non-target confirmed tracks | confirmed tracks never observed near any aircraft label (a false-alarm proxy) |
| forecast centroid error | median/p90/support per model and horizon, issued while the track is the target's unique match, against labels interpolated across at most one 25 fps frame gap |
| ms per frame | decode+detection and tracking+forecast p50/p95 wall-clock on this machine; not a benchmark |

`compare` applies the PLAN-011 rule (track recall ≥ baseline, fragments ≤ baseline, identity
swaps ≤ baseline, off-target, impurity and non-target counts ≤ max(1.25 b, b + 5)), lists
sequences whose track recall dropped, and gives forecast error on the common subset of
(sequence, issue frame, model, horizon) keys with each side's unmatched counts.

The optional `--pan` shifts every decoded frame by a seeded, smooth, known offset (three
sinusoids per axis, 0.05–0.5 Hz, peak 120 px, grey border) and shifts labels identically. It is
a development-only stand-in for a moving camera: translation only, no rotation, blur or parallax.

## Limitations

- Approximate single-point labels: no AP, HOTA/IDF1, false-alarm rate or calibrated accuracy.
  Off-target and non-target counts include unlabelled aircraft, birds and label gaps.
- Alignment of S1, S2 and S4 is inferred (the published bounds disagree with the decoded count
  by one frame; see ADR 032), so their labels carry ±1 frame (40 ms) uncertainty.
- One fixed camera on two days; dev M3 and holdout M4 share five frames of one recording; S5 was
  already used in AERIS-031. No confidence intervals.
- The `fast-zoom` profile depends on external, unaccepted tiling code (commits `79a7cb3`,
  `c480bf0`); it is secondary evidence only.
- QUT does not contain hand-held panning; the synthetic pan is a partial substitute.

## Results

Result files are in [`benchmarks/qut_eval/`](../benchmarks/qut_eval/).

## Development baseline

Recorded 2026-10-22 on the development split (S1, S3, S5, S7, M1, M3, M5, M7), uncommitted
working tree on branch `tracking-improvements` (each file records commit and dirty flag).
Pooled; constant-velocity forecast error is median px (support).

| Run | Labelled frames | Det. recall | Track recall | Fragments / switches | Off-target | Impurity | Swaps | Non-target | CV 0.5 s | CV 1.0 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` (fast) | 10792 | 0.123 | 0.115 | 26 / 18 | 15 | 10 | 0 | 1 | 12.6 (1225) | 25.3 (1163) |
| `baseline` + synthetic pan | 9807 | 0.126 | 0.123 | 39 / 32 | 17 | 12 | 0 | 2 | 44.8 (1160) | 134.6 (1096) |
| `baseline-fast-zoom` (secondary) | 10792 | 0.195 | 0.192 | 31 / 37 | 19 | 13 | 0 | 4 | 7.0 (2056) | 12.2 (1990) |

What this shows, descriptively:

- **Detection is the main limit on QUT.** Targets are labelled from when they are distant dots;
  the whole-frame detector finds the target in about one labelled frame in eight, and tiling
  raises that to about one in five. Track recall follows detection recall closely.
- **The synthetic pan shows a similar failure pattern to the F-35 clip:** target fragments rise
  from 26 to 39 and the median 1 s forecast error from about 25 px to about 135 px. The two runs
  are not on identical label sets: the pan excludes 985 labels whose shifted position leaves the
  frame (429 in S1, 455 in S3), so recalls and errors are only roughly comparable.
- Guard counts (off-target, impurity, swaps, non-target) are small at baseline, so the
  `max(1.25 b, b + 5)` bound mostly works through its `+ 5` term.
- Decode plus detection takes about 11 ms/frame (`fast`) and about 55 ms/frame (`fast-zoom`)
  p50 on the development machine. The same clips vary between runs (S3: 25.6 ms in the plain run,
  11.3 ms in the pan run); the cause was not investigated.

No holdout result exists for this baseline section. AERIS-033's selection is in
[ADR 033](decisions/033-manoeuvre-robust-tracker.md); its full seven-sequence holdout result, and
AERIS-034's guided-crop result, are in [ADR 034](decisions/034-tracker-guided-crop.md). Since
AERIS-034, target labels outside the image (six at the end of S2) are dropped and counted as
`target_labels_outside_image` rather than aborting the run, and a result written outside
`benchmarks/qut_eval/` gets its own raw-error file. Results include
`target_detection_recall_with_crops` beside the frame-pass detection recall (equal when no crops run).
