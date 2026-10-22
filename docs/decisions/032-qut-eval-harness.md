# ADR 032 — QUT multi-sequence evaluation harness

Date: 2026-10-22. Task: AERIS-035, which replaced AERIS-032 (PLAN-011 Addendum A). Plan: PLAN-011 r3.

## Decision

`scripts/qut_eval.py` evaluates a listed variant on the PLAN-011 development or holdout split of
the 15 pinned QUT sequences and writes `benchmarks/qut_eval/<name>-<split>[-pan].json`.

- **Separation.** `detect` and `track` see video only; `read_centroids` is called after the
  runtime outputs for a sequence are complete. Labels never reach the detector, tracker or
  forecasters.
- **Reuse.** Association (`unique_near`, the 50 px gate) and label interpolation come unchanged
  from the accepted AERIS-031 pilot. The tracker is the accepted AERIS-009/027 `KalmanTracker`
  with its low-score pass at 0.3; forecasters are the accepted AERIS-010 models.
- **Pinning.** The manifest pins videos by published LFS SHA-256 and CSVs by published Git blob
  SHA-1, with sizes, dates and global-frame bounds. `fetch` refuses a total above 40 MB and
  never touches other release files.
- **Frame alignment.** The first run found that `target-info.csv` bounds are not consistent:
  for 12 sequences `end_frame − start_frame + 1` equals the decoded frame count and the last
  target label is the last decoded frame (as AERIS-031 found for S5), but S1 and S2 decode one
  frame fewer and S4 one frame more. One rule fits all 15: *the last target label falls on the
  last decoded frame*. The manifest pins each sequence's decoded count and first decoded global
  frame from that rule (S4 starts at `start_frame − 1`; S1/S2 at `start_frame`); `alignment`
  marks S1, S2 and S4 as inferred with ±1 frame (40 ms) uncertainty. The runner refuses a
  decoded count other than the pinned one and a sequence where the rule fails. Excluding the three
  sequences instead would have changed the pre-registered split.
- **Other-aircraft labels** outside a sequence's bounds are dropped and counted, never shifted.
- **Candidates.** Variants come only from `configs/qut/candidates.json` (≤ 12 per
  task); each result carries the list's SHA-256 so a list edited after a task's first run shows.
- **Result integrity (attempt 02).** Each report stores the SHA-256 of its per-forecast error file
  and `compare` refuses a replaced file; subset runs get their own file name; holdout results are
  never overwritten; `--selection-record` must be a `docs/decisions/` record and its hash is stored;
  candidate entries pin their tracker config SHA-256.
- **Identity swaps.** Counted in any frame where a target-carrying track uniquely matches an
  other-aircraft label outside the target's gate, including frames with no target label (there the
  observation cannot be the target). This is stricter than the PLAN-011 wording, which only names
  frames with a target label. Ambiguous frames are counted over labelled target frames.
- **Detection cache.** Frame-pass detections are cached per (video, profile, config, weights,
  pan); it stores runtime outputs only. `--no-cache` bypasses it.
- **Synthetic pan.** A seeded, known per-frame translation applied before detection, with labels
  shifted identically; development split only.

## Alternatives rejected

- Calling `aeris.cli.run_pipeline`: AERIS-012 is unaccepted, the CLI writes a full artifact set
  per run, and it does not expose track states with measurement centres without parsing CSVs.
  The harness composes accepted modules directly, as AERIS-031 did. AERIS-034 adds the crop pass
  to the harness explicitly.
- Recording a holdout baseline now: PLAN-011 runs the holdout only for a written selection, so
  AERIS-035 records the development baseline only.

## Consequences

Detector and tracker changes in AERIS-033/034 are compared on labelled real video with
guard measures. The evidence stays approximate-label, one-camera and descriptive.
