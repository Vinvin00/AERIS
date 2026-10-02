# Evaluation protocol v1

This module evaluates canonical original-frame pixel records. It does not produce
real aircraft benchmark results until eligible real labels, identities, and predictions
exist. Synthetic unit fixtures exercise conversions and arithmetic only.

## Detection

`detection_metrics` invokes pycocotools 2.0.11 COCOeval for bbox AP50 and
AP@[.50:.95]. Category IDs are shifted by one for COCO; image IDs are unique per
sequence/frame. COCO's max detections are `[1, 10, max_detections]`, default 100.
`score_threshold` must be fixed on validation before evaluating another split.
At that threshold and a stated IoU threshold (default 0.5), one-to-one score-order
matching yields TP/FP/FN, precision, recall and F1. Ignored truth can absorb a
same-class overlapping detection. Complete empty frames count as negatives and
record error detections. FP/frame divides by all complete frames.

Custom bbox width bins in **original pixels** are `[0,16)`, `[16,32)`, `[32,64)`,
`[64,infinity)`. Their support and recall are separate from COCO's area bins.
Partial and unlabeled frames are excluded from all scores; skipped counts are
reported. This conservative policy prevents unannotated objects becoming false
positives or true negatives. A complete all-negative set with no categories has
no AP, though its negative count and zero errors remain available. No complete
annotations means truth-dependent detection metrics are unavailable.

FGVC's partial main-aircraft labels are ineligible for COCO AP and FP counting.
Its separate fixed-budget recall/localization smoke check is outside this adapter.
No threshold should be selected or changed using a sealed holdout.

## Tracking

`tracking_metrics` calls TrackEval 1.3.0 HOTA and Identity metrics. An eligible
sequence has complete timed annotations, nonignored objects with track IDs, and
at least one identity. Partial, untimed, or identity-free sequences are skipped.
Input `TrackObservation` IDs are sequence-local. Predicted-only positions are counted
but excluded from HOTA/IDF1, which score observed tracker output. Box IoU is the
TrackEval similarity, with a class mismatch assigned zero similarity. HOTA is the
mean over TrackEval's alpha thresholds; IDF1 uses its default 0.5 IoU threshold.
No eligible sequence yields null HOTA/IDF1 and an availability reason. Identity
switches, fragmentation and retention definitions need a later task before those
auxiliary metrics can be reported; this adapter does not fabricate them.

## Forecast

`forecast_metrics` scores declared models on the same eligible
(sequence, identity, issue time, horizon) keys. Issue truth must be observed on a
complete timed frame. Target truth is an exact observed sample or linear interpolation
between two observed complete samples of that identity within the configured maximum
gap (default 0.5 s). Exact target matching uses an absolute 1e-9 s tolerance
with zero relative tolerance, so large absolute timestamps cannot turn the issue
observation into future truth. It never extrapolates. Horizons must be strictly positive (a zero horizon would score the issue observation as its own future truth; it raises `EvaluationError`). Horizons default to 0.1, 0.25, 0.5, 1,
and 2 seconds. The function consumes completed predictions; runtime predictors
never receive truth or future observations through this API.

Per model and horizon, support counts eligible keys, matched counts a finite
forecast center, missed includes absent or unavailable centers, and coverage is
matched/support. RMSE and mean Euclidean error use matched keys only, always
accompanied by support and misses. Forecasts without eligible truth are counted
separately. If no eligible future truth exists, error scores are null. No confidence
interval is inferred from these small fixtures.

## Reproduction and limitations

Install the `evaluation` optional dependencies from `pyproject.toml`, or the
version-pinned local set in `requirements/evaluation-macos-arm64-py314.txt` with
`--no-deps` on the existing `.venv`. Run:

```sh
.venv/bin/python tools/check_tests.py tests/unit/test_metrics.py
```

The local version pins have no wheel hashes or cross-platform lock. Real evaluation
must record dataset/split hashes, threshold selection, evaluator versions and full
run manifest. Neither synthetic tests nor absent labels establish real quality.
