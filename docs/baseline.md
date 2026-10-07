# Baseline manifest, metric report and threshold protocol (AERIS-007, offline half)

**There is no measured baseline.** The only labeled real data is three FGVC still images
whose frames are `partial` (one main aircraft boxed, not exhaustive; one image is a cabin
interior). The AERIS-005 evaluator excludes partial frames from every score, so AP, size-bin
recall, negative-frame errors and false-positive rates are *unavailable* on them. The real
measurement and frozen thresholds are the deferred task AERIS-022 (PLAN-003). This task
records what can be recorded honestly: reproducibility digests, the report contract and the
threshold-selection protocol.

## `benchmarks/baseline_manifest.json` (AC1)

Status `smoke_not_a_baseline`. It records, for the detector smoke run on the three-image
sample: the model (path, size, SHA-256, source URL, licence notes), the detector config
digest, the evaluator module digest and pinned versions (pycocotools, TrackEval), the
environment (OS, Python, numpy, OpenCV, ONNX Runtime, CPU provider, float32, 416x416),
dependency-lock digests, the dataset (dataset/annotation hashes, per-image SHA-256, size,
dimensions, annotation status, rights), the absent splits with a reason, the seed (none, with
reason), the code revision at creation and the smoke result (`no_complete_annotations`,
3 partial frames skipped, `ap50`/`ap50_95` null). Detection counts per image are provenance, not
accuracy. The `score_threshold` 0.5 in the evaluator block is a required API argument and is
explicitly **not** a validation-selected operating threshold.

`tests/unit/test_baseline.py` recomputes every digest against the files (model, images and
processed data only when the Git-ignored local copies exist), checks the pins against the
requirement files, and re-runs the real sample when the weights are present. Because the
evaluator digest is recorded, any edit to `python/aeris/evaluation/metrics.py` fails that test
until the manifest is regenerated and re-reviewed; that coupling is intentional.

Recompute the digests by hand:

```sh
shasum -a 256 models/onnx/yolox_tiny.onnx configs/detector/baseline.json \
  python/aeris/evaluation/metrics.py requirements/*-macos-arm64-py314.txt \
  data/processed/g1-bounded-sample/dataset.json data/processed/g1-bounded-sample/annotations.jsonl \
  data/raw/g1-bounded-sample/images/*.jpg
```

## Metric report (AC2)

`detection_metrics` (AERIS-005) already provides the report; this task wires and demonstrates
it rather than adding metric code. With complete labels it returns COCO AP50 and AP@[.50:.95],
custom original-pixel width bins `[0,16)`, `[16,32)`, `[32,64)`, `[64,inf)` each with support,
matched count and recall, negative-frame count and errors, TP/FP/FN, false positives per complete
frame and the number of skipped partial/unlabeled frames. With no complete labels it returns
`available: false`, `reason: no_complete_annotations`, null AP and the skipped counts. Tests
cover both paths and a mixed set (partial frames are excluded, never turned into false
positives). No number from the source plan is copied anywhere (FEASIBILITY.md treats them as
illustrative).

## Threshold-selection protocol (AC3)

`configs/evaluation/regression_schema.json` is the contract for the future regression config.
This repository holds **no** `regression.json` and no threshold values.

1. **Data first.** Complete-label, rights-cleared real data is split by recording/session
   or source group before frames (CONTRACTS.md). The development split is hashed and frozen; the
   holdout is hashed and never opened until release evaluation (`holdout_opened` must stay
   false).
2. **Repeatability.** Re-run the unchanged baseline at least twice (the schema requires
   `runs >= 2` and one seed per run; deterministic runs record a null-effect seed list) on the
   development split and record each metric's observed standard deviation and its support.
3. **Tolerance.** A tolerance may not be smaller than the observed standard deviation (rule R3)
   and needs a written rationale; metrics below `min_support` are unavailable.
4. **Freeze.** Thresholds, tolerances and rationales are committed with the freeze commit and time
   before any candidate comparison (`before_candidate_comparison` true).
5. **Fail closed.** A missing or unavailable metric fails the gate (`unavailable_policy: fail_closed`);
   gates are never relaxed to make a candidate pass.

Rules R1-R6 in the schema (draft configs carry no values; frozen configs need complete evidence)
are enforced in the tests by a small reference checker; AERIS-015 implements the production
enforcement and diagnostics.

## What AERIS-022 must supply

Rights-cleared real data with complete box labels and a grouped development/holdout split,
a measured baseline report on the development split, repeatability runs, and the frozen
`configs/evaluation/regression.json`.

## Limits

Smoke data only; one machine (macOS arm64, Python 3.14); the weight licence is unverified; the
code revision is a creation-time record; the reference checker for the schema's cross-field rules
lives in the tests, not in product code.
