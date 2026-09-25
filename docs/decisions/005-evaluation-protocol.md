# ADR 005 — Standard evaluators and evaluation protocol

Status: proposed for AERIS-005 review. Date: 2026-09-25.

Decision:

- **COCO.** Detection AP comes from pycocotools 2.0.11 `COCOeval` (bbox), over complete
  frames only. AERIS `ignore` objects map to COCO crowd regions. Partial frames get a
  full-frame crowd region, so unmatched detections there are unscored. Threshold
  metrics need a caller-supplied score threshold and use COCOeval matches at one stated
  IoU. Width bins are the contract's original-pixel bins, with explicit support.
- **Tracking.** Metrics come from the official TrackEval at commit `12c8791b` (MIT),
  installed as a VCS pin. The adapter calls its HOTA, CLEAR and Identity classes and its
  box IoU directly on identity-labelled complete sequences. Distractor removal is one
  joint Hungarian assignment over labelled and ignored truth (IoU < 0.5 zeroed), exactly as
  in TrackEval's MOTChallenge preprocessing. TrackEval still references
  `np.float`/`np.int`, removed in NumPy 1.24. The adapter restores exactly those builtin
  aliases while TrackEval runs and removes them afterwards; the upstream source is
  unmodified. Parity against the same commit on NumPy 1.23 without the shim was
  identical.
- **Forecasts.** Evaluation follows an oracle-identity protocol with an eligible set
  derived from truth alone, and publishes that set's digest. Truth is interpolated
  linearly within `max_gap_s`, with no extrapolation. Misses stay visible through
  `missed` and `coverage`. Interval coverage uses chi-square(2) ellipses, and
  sequence-bootstrap intervals need at least 10 sequences.
- **Pins.** Pinned versions are enforced at run time and recorded in every report. The
  check covers the pycocotools version, the TrackEval commit pip recorded, and TrackEval's
  installed files against pip's RECORD hashes. The `dev` extra carries the evaluators, so
  CI installs them.

Alternatives rejected:

- The PyPI `trackeval` 1.3.0. It is a third-party fork, and it pulls `opencv-python`,
  which conflicts with headless OpenCV, plus matplotlib and pillow.
- Pinning NumPy < 1.24. That conflicts with the project's NumPy 2 dependency.
- Vendoring a patched TrackEval. That modifies the standard evaluator's source.
- py-motmetrics. It has no HOTA, which the contract requires.

Consequences: AERIS-007 supplies the validation-selected score threshold, and it and
later tasks report metrics only through these adapters. End-to-end forecast scoring
through tracker association needs a separate protocol decision. Upgrading either
evaluator is a deliberate change: re-run `tests/unit/test_metrics.py` and record the
new pin here.
