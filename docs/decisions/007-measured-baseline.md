# ADR 007 — Measured baseline on AOT and the frozen development gate

Status: proposed for AERIS-007 attempt 2 review. Date: 2026-09-27.

Context: AERIS-006 fixed a pretrained YOLOX-S detector and its CPU ONNX path. AERIS-007
must measure it on real labelled data, freeze a development regression split, and set
regression thresholds from measured repeatability, with a separate final holdout. The
owner's video pilot does not exist. On 2026-09-27 the owner asked for labelled data found
online, with more budget if needed.

Decision:

- **Data.** The Airborne Object Tracking dataset (AOT, Amazon Prime Air, training set,
  CDLA-Permissive-1.0, public S3). It is real video of airplanes and helicopters with
  boxes, track IDs and capture timestamps, including frames with no objects. Nothing
  from it is committed; frames are fetched by key and checked against the S3 ETag (MD5)
  and a recorded SHA-256. The ground-truth CSVs are pinned by SHA-256.
- **Rejected sources.**
  - TartanAviation: fixed airport cameras, RGB and tracked aircraft, so it is the closer
    domain. Its servers use ports 8080/8081, which this environment's proxy cannot reach.
  - FGVC-Aircraft: one main aircraft per image and no negatives, so it cannot give AP
    or false positives (docs/dataset.md).
  - COCO val2017: the detector's own validation set. It has no video, no source groups
    and no small, distant aircraft.
- **Sequences.** AOT parts 1–2 are camera 0 and part 3 is camera 1 of the same flights.
  A sequence is a flight and a camera.
- **Class policy.** `Airplane` and `Helicopter` map to AERIS class 0. About 81 % of AOT
  aircraft boxes are nominal 6 x 6 or 16 x 16 markers around point-like targets, not
  extents. Sides are compared within 1e-6 px, because the CSV's `right = left + width`
  carries floating-point noise. Those aircraft, and `Airborne` (unidentified) and `Drone`
  objects, become ignore regions (COCO crowd regions). Birds and flocks are left out, so
  a detection on one is a false positive. Frames are `complete`.
- **Splits.** The unit is the capture day of a flight (both cameras share it). Days are
  ordered by a salted SHA-256: a quarter holdout, a quarter validation, the rest
  development.
  - The holdout frame list is frozen by digest. Its frames are not downloaded or
    evaluated before AERIS-018.
  - Frames are chosen from labels only (`configs/evaluation/aot_benchmark.json`): every
    10th frame, capped per sequence by kind (measured-extent aircraft, empty, other).
    The mix is enriched for scored aircraft compared with the dataset's frame
    distribution.
- **Protocol.** The AERIS-006 detector, unchanged, on ONNX Runtime CPU. Evaluation uses
  AERIS-005's `evaluate_detection` (pinned pycocotools) at IoU 0.5. The score threshold
  is the grid value (0.01 to 0.99) with the highest F1 on validation. The development
  split is scored once at that threshold.
- **Repeatability and tolerances.**
  - The baseline is run twice: 1 thread, and 4 intra-op threads in a fresh process.
  - Each gated metric's tolerance is max(2 x the observed run-to-run difference, the
    metric's resolution floor). The floor is 0.001 for AP, COCO's reporting resolution,
    and one count for count metrics.
  - Width bins with fewer than 30 scored objects, and negative-frame metrics with fewer
    than 30 frames, are not gated. They are reported as unavailable, never as passing.
  - Capture-day bootstrap intervals are reported as context, not used as tolerances.
- **Freezing.** The selection, class policy, threshold rule and tolerance rule were
  committed (`3985a8f`, benchmark v1) before the detector ran on any AOT frame. The
  measured threshold, baseline values and tolerances are frozen in
  `configs/evaluation/regression.json` before any candidate is evaluated.
- **Revision v2 (after v1 results were seen).** The attempt-1 review found that v1
  compared marker sides exactly, so 7,697 markers with sides like 16.000000000000114 px
  were scored as aircraft. Of those, 16 were in development, 11 in validation and 19 in
  the holdout.
  - v2 compares within 1e-6 px. No other rule value changed: split, caps, threshold
    rule, tolerance values and floors, evaluator, IoU and detector. Three clauses were
    added to say how the rules apply: the rounding margin, the environment scope and the
    fixed-threshold candidate protocol.
  - The frames were reselected and the pipeline re-run.
  - The v1 outputs are kept in `benchmarks/runs/aeris-007-v1/` (not in Git).
  - The same review found that `gate_check` failed exactly one-count-worse candidates
    through floating-point rounding. It now uses a 1e-9 relative margin.
- **Candidate protocol.** A candidate is scored on the development frames with the same
  evaluator, IoU 0.5 and the frozen score threshold from `regression.json`. It does not
  re-select a threshold; one that needs its own threshold is a different protocol
  needing a reviewed change. Tolerances hold in the measured environment. Elsewhere, the
  baseline is re-run first, and repeatability is re-measured if its detections differ.

Consequences:

- Measured on 2026-09-27 (docs/baseline.md), benchmark v2:
  - validation chose threshold 0.16;
  - development AP50 is 0.067 and recall 0.083 (44 of 531), with recall 0.451 for
    aircraft at least 64 px wide and 0 below 16 px;
  - 1-thread and 4-thread runs gave bit-identical detections, so the tolerances are
    the resolution floors.

  Two diagnostics (a false-positive breakdown and IoU 0.2) were added after these
  results were seen. They are labelled as such, and the gate is unchanged.
- The benchmark is a domain proxy. AOT is grayscale video from a moving aircraft; the MVP
  targets a fixed RGB ground camera. The baseline measures the pretrained detector on
  real aircraft video, but not in the target setting. The owner's pilot or a reachable
  fixed-camera source (TartanAviation) is still needed for target-domain claims.
- COCO has no helicopter class, and most AOT aircraft are helicopters. Recall is also
  reported per category, so this is visible.
- Point-like targets are excluded from scoring. Evidence about aircraft below a few
  pixels needs point-based evaluation, which is not a standard COCO metric and is not
  claimed here.
- A candidate model or setting (AERIS-015, AERIS-017) must use the same frames,
  evaluator, IoU and frozen threshold, and pass `configs/evaluation/regression.json`.
  The holdout stays sealed until release evaluation.
- COCO crowd matching divides by the detection's area. A future tiny-target candidate
  that boxes a marked aircraft much more loosely than the marker (above about twice its
  area) is charged a false positive. AERIS-017 must account for this.
