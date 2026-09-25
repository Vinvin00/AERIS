# Evaluation protocol (AERIS-005)

`aeris.evaluation.metrics` gives three evaluators over contract-v1 records:
`evaluate_detection`, `evaluate_tracking` and `evaluate_forecasts`. Each returns a
strict-JSON report. A number that cannot be computed is `null`, and its
`availability_reason` entry names why. Nothing unavailable is reported as zero or a
pass. The tests (`tests/unit/test_metrics.py`) run on the SYNTHETIC analytic fixtures
from AERIS-004. They exercise the evaluators and the protocol; they say nothing about
model quality.

## Standard evaluators and pins

| Evaluator | Pin | License | Use |
|---|---|---|---|
| pycocotools (COCO API) | 2.0.11 from PyPI | BSD-2-Clause | `COCOeval` bbox: AP50, AP@[.50:.95], per-image matches |
| TrackEval | official upstream commit `12c8791b303e0a0b50f753af204249e622d0281a` (2022-11-29) | MIT | HOTA, CLEAR, Identity metric classes and sequence combination; box IoU |

Both are verified at run time. `require_pinned_evaluators()` refuses other versions,
and every report records the installed versions.

**TrackEval and NumPy 2.** TrackEval at the pinned commit still uses `np.float` and
`np.int`, which NumPy 1.24 removed; NumPy only ever defined them as the builtins `float`
and `int`. The adapter restores exactly those two names while TrackEval runs and
removes them afterwards (`numpy_scalar_aliases`). The official source is not modified.

Parity was checked in a scratch environment. The same commit on NumPy 1.23.5 /
SciPy 1.11.4, with no shim, gave identical HOTA/CLEAR/Identity results on perfect,
missed and wrong-ID cases to NumPy 2.4.6 with the shim. Without the shim, NumPy 2 fails.
The PyPI package `trackeval` 1.3.0 is a third-party fork (kovalp/TrackEval). It also
requires `opencv-python`, which conflicts with this project's headless OpenCV, so it is
not used.

Install: `pip install -e '.[eval]'` (or `.[dev]`), or the hash lock
`requirements/eval-linux-x86_64-py311.txt` plus the VCS pin it documents.

## Detection (`evaluate_detection`)

| Rule | Behaviour |
|---|---|
| Frame eligibility | `unlabeled` frames are excluded; detections on them are counted (`detections_on_unlabeled_frames`) and never scored |
| AP | COCOeval AP50 and AP@[.50:.95] over `complete` frames only. Partial labels cannot score false positives, so AP is null without complete frames. Labelled objects with no detections give COCO's AP 0 |
| `ignore` objects | Become COCO crowd regions (`iscrowd=1`), COCO's own ignore mechanism. A detection matched to one is neither a true nor a false positive (`detections_matched_to_ignored_regions`) |
| Partial frames | A full-frame crowd region per category leaves unmatched detections unscored. Labelled objects there count toward `recall_including_partial` and the width bins |
| Threshold metrics | Precision, recall, F1, TP/FP/FN and false positives per complete frame need a caller-supplied, validation-selected `score_threshold` (inclusive). Without one they are null with a reason. They use COCOeval matches at one stated `iou_threshold` (default 0.5) on complete frames |
| Budget | `max_detections_per_frame` (1–100, default 100 = COCO's) keeps the highest scores. With the threshold it gives the fixed detection budget a recall-only (partially labelled) check needs (AERIS-002 N3) |
| Width bins | Contract bins on truth box width in original pixels: `[0,16)`, `[16,32)`, `[32,64)`, `[64,inf)`. They are not COCO area ranges. Each bin reports `support` (non-ignored labelled objects) and `recall`; zero support gives a null recall |
| Negative frames | `complete` frames with no objects: count, frames with a false positive, false positives per negative frame |
| Consistency | A detection on a frame without truth, with metadata different from its frame, or with a different class map raises `EvaluationError`. Classes the truth never labels score as false positives |

## Tracking (`evaluate_tracking`)

- **Eligible sequences.** Only identity-labelled ones: every frame is `complete` and
  every non-ignored object has a track ID. Others are listed in `ineligible_sequences`
  with a reason. Detector-only or partial data cannot score tracking, and with no
  eligible sequence `combined` is null.
- **Tracker input.** `TrackerBox(sequence_id, frame_index, track_id, bbox, observed)`.
  `observed=False` marks a predicted-only position. Those are excluded unless
  `include_predicted=True`, and always counted separately. Duplicate tracker IDs in a
  frame raise.
- **Distractors.** Tracker boxes matched (Hungarian, IoU >= 0.5) to ignored truth objects
  are removed before scoring, like MOTChallenge distractor preprocessing.
- **Metrics.** Similarity is TrackEval's box IoU. Metrics are TrackEval's classes:
  - HOTA, DetA, AssA and LocA are the mean over TrackEval's 19 localization thresholds.
  - MOTA and MOTP use CLEAR at IoU 0.5.
  - IDF1, IDP, IDR and IDTP/FN/FP use Identity at IoU 0.5.
- **Named definitions (TrackEval's):**
  - **IDSW**: a truth track's match moves to a different tracker ID from its previous
    match.
  - **Frag**: the number of times a truth track's tracked segments restart. This
    includes restarts after the object's own visibility gaps, so perfect tracking of a
    track hidden for two frames gives Frag 1.
  - **Retention**: MT, PT and ML are the truth tracks tracked for > 80 %, 20–80 %
    (inclusive) and < 20 % of their annotated frames.

## Forecasts (`evaluate_forecasts`)

- **Oracle-identity protocol.** Forecasts are `PredictionRecord`s keyed by ground-truth
  `track_id`, issue time and horizon. Forecasting is scored apart from association;
  end-to-end forecast scoring through tracker matches is not implemented here.
- **Eligible set.** It comes from truth alone, so it is the same for every model. Its
  SHA-256 digest is reported as `eligible_set_digest`, which comparisons must check.
  An eligible (track, issue, horizon) triple needs three things:
  - the track is annotated at the issue time;
  - truth at issue time + horizon can be interpolated linearly between two annotated
    observations at most `max_gap_s` apart (default 0.5 s);
  - the target lies within the labelled span. There is no extrapolation, and
    interpolated truth is evaluator-only.

  Untimed sequences are excluded with a reason.
- **Misses.** An absent forecast or a null centre counts as missed. Each horizon reports
  `eligible`, `scored`, `missed`, `coverage` (scored / eligible), `rmse_px` and
  `mean_error_px`. An error score is never shown without its coverage.
- **Unscored forecasts.** These are counted by reason: unknown track, issue time not an
  annotated observation, target beyond the labelled span, target in an over-long gap, or
  horizon not evaluated. A forecast issued before its track's first annotated
  observation is counted as a `future_leakage_suspect`.
- **Uncertainty.** For forecasts with a covariance, the report gives empirical coverage
  of Gaussian ellipses at 50/90/95 % (chi-square(2) quantile −2 ln(1−p)) and the mean
  ellipse area. Degenerate covariances are counted, not scored. Covariance coverage is
  an empirical check, not proof of calibration.
- **Trajectory errors.** ADE/FDE are computed over issue points whose every evaluated
  horizon is eligible and scored. FDE uses the longest horizon.
- **Intervals.** Per-horizon RMSE intervals come from a sequence bootstrap: whole
  sequences are resampled, never correlated frames (seeded, 1000 resamples, 95 %). They
  need at least `MIN_SEQUENCES_FOR_INTERVALS = 10` sequences with scored forecasts;
  below that they are null with a reason. This is the minimum `docs/dataset.md` asked
  AERIS-005 to set.

## Limitations

- **Real data.** Detection, tracking and forecast numbers on real data do not exist yet.
  AERIS-007 needs a real labelled baseline.
- **Forecast protocol.** Forecast evaluation uses ground-truth identities. Scoring
  forecasts end to end through a tracker needs an association protocol that is not
  implemented.
- **TrackEval shim.** It is process-global while it runs and is not thread-safe.
  TrackEval's dataset loaders and plotting are unused.
- **COCO semantics.** COCO crowd IoU (intersection over detection area) applies to
  ignored objects. This is COCO's standard behaviour, and it is more permissive than
  ordinary IoU.
- **Bootstrap interval.** The percentile bootstrap is a simple default. The minimum
  sequence count is a policy choice, not a statistical guarantee.
