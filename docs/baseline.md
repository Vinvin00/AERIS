# Measured baseline and development gate (AERIS-007)

Status: attempt 2 (benchmark `aeris-007-aot-v2`), awaiting independent review.

This is the first measurement of the AERIS-006 detector (pretrained YOLOX-S, COCO
`airplane` → AERIS class 0, ONNX Runtime CPU) on real labelled aircraft video. It also
freezes the development regression set and the regression thresholds for later
candidates. Decisions and alternatives are in ADR 007.

**What this is not.** It is not a result in the MVP's target setting (a fixed RGB ground
camera). AOT is grayscale video filmed from an aircraft. The numbers describe this frozen
frame set under the rules below, not aircraft detection in general.

## Data: Airborne Object Tracking (AOT)

| | |
|---|---|
| Source | Amazon Prime Air, AOT training set ([registry](https://registry.opendata.aws/airborne-object-tracking/), [dataset notes](https://gitlab.aicrowd.com/amazon-prime-air/airborne-detection-starter-kit/-/blob/master/docs/DATASET.md)) |
| Licence | [CDLA-Permissive-1.0](https://cdla.dev/permissive-1-0/) |
| Storage | public S3 bucket `airborne-obj-detection-challenge-training`; nothing committed |
| Frames | 8-bit grayscale PNG, 2448 x 2048, about 10 fps, camera on an aircraft |
| Ground truth | `part{1,2,3}/ImageSets/groundtruth.csv`, pinned by SHA-256 and S3 ETag (MD5) in `configs/evaluation/aot_benchmark.json` |

Measured facts from the pinned ground truth (2026-09-27):

- 3,438 flights over 61 capture days. Parts 1 and 2 are camera 0 and part 3 is camera 1
  (each part's `groundtruth.json` names its camera). 716 flights appear in part 3 as well
  as in part 1 or 2, with different frames and timestamps. A sequence is therefore a
  flight and a camera.
- The parts order their CSV columns differently (part 2 puts `id` second). The reader
  matches columns by name and requires exactly the same set.
- Of 2,013,970 airplane and helicopter boxes, 1,628,467 (81 %) are nominal markers
  around point-like targets, not extents. They have a side of 6 or 16 px, with the other
  side no larger than 16. Of those, 7,697 are off by floating-point noise below 1e-6 px:
  the CSV computes `right = left + width`, giving widths like 16.000000000000114. The
  remaining 385,503 boxes have measured extents: median width 26.2 px, and 45,946 are
  at least 64 px wide. A further 52 box sides differ from 6 or 16 px by 1e-6 to 1e-3 px,
  and 2,330 by 1e-3 to 0.05 px. Those count as measured.
- No box lies outside the frame or is empty.

## Class policy (`aeris-aircraft-v1`)

| AOT category | AERIS |
|---|---|
| `Airplane`, `Helicopter` with a measured box | class 0, scored |
| `Airplane`, `Helicopter` with a nominal box | class 0, `ignore` region |
| `Airborne` (unidentified), `Drone` | class 0, `ignore` region |
| `Bird`, `BIrd`, `Flock` | left out, so a detection on one is a false positive |

An `ignore` region is a COCO crowd region: a detection matched to it is neither a true
nor a false positive. COCO matches crowd regions by intersection over the *detection's*
area, so at IoU 0.5 a detection more than about twice the marker's area is not matched.
It counts as a false positive. On this baseline no detection at the threshold overlaps
an ignore region, but a future tiny-target candidate could be charged this way. AOT labels no occlusion or truncation, so those contract flags are
`false`, meaning "not labelled". Every frame is `complete`. The dataset labels every
airborne object annotators found, planned or not, but its documentation makes no
completeness guarantee (see limitations). Timestamps are AOT capture times in seconds
from the sequence's first frame (`timestamp_source = "provided"`).

## Splits and frame selection (frozen before any detector run)

The definition is `configs/evaluation/aot_benchmark.json`. Version v1 was committed in
`3985a8f` before the detector ran on any AOT frame.

**Revision v2.** It was made after the v1 results were seen, prompted by the attempt-1
review's finding F1. v1 compared marker sides exactly, so markers with float-noise sides
were scored as aircraft: 16 in development, 11 in validation and 19 in the holdout, all
16–32 px wide. v2 compares within 1e-6 px; nothing else in the definition changed. The
frames were reselected and every step re-run. The v1 outputs are kept in
`benchmarks/runs/aeris-007-v1/` (not in Git). The v1 figures are in the attempt-1 report
and are superseded.

- **Unit.** The capture day of a flight: the UTC date of its earliest timestamp over both
  cameras. Days are ordered by `sha256("aeris-007-aot-split-v1:" + day)`. The first
  quarter is holdout, the next quarter validation, the rest development. No day, flight
  or frame is shared between splits (checked).
- **Sequences.** Within a split, sequences are taken round-robin over days: 90
  development, 60 validation and 90 holdout.
- **Frames.**
  - Candidates are every 10th frame.
  - Per sequence, the selection keeps up to 15 frames with a measured-extent aircraft,
    4 frames with no labelled object, and 3 other frames (nominal markers, unknown
    objects or birds only). Each kind is taken in salted-hash order.
  - This is label-only selection. It enriches the set for scored aircraft compared with
    AOT's natural frame mix.
- **Holdout.** Sealed. Its frame list is frozen by digest in the manifest. Its frames
  were not downloaded or evaluated, and it never drives thresholds.

## Selected frames

| Split | Days | Sequences | Frames | Scored aircraft (airplane / helicopter) | Empty frames | Ignored objects |
|---|---|---|---|---|---|---|
| Development | 31 | 90 | 1,114 | 544 (193 / 351) | 351 | 280 |
| Validation | 15 | 60 | 775 | 395 (247 / 148) | 235 | 304 |
| Holdout (sealed) | 15 | 90 | 1,039 | 482 by label count (not evaluated) | 325 | — |

Scored aircraft by width in original pixels:

| Split | [0,16) | [16,32) | [32,64) | [64,inf) |
|---|---|---|---|---|
| Development | 47 | 282 | 144 | 71 |
| Validation | 19 | 181 | 117 | 78 |

The development set also has 22 frames whose only labelled objects are birds. They count
as negative frames: 351 empty frames plus these 22 give the evaluator's 373.

Downloaded: 1,889 frames (4,682,154,560 bytes) plus the three ground-truth CSVs
(946,656,956 bytes). That is within the task budget of 8 GB. The holdout was not
downloaded.

## Protocol

- **Detector.** `configs/detector/baseline.json` as accepted in AERIS-006, unchanged:
  the AERIS opset-17 ONNX export (SHA-256 `e68784f4…`), ONNX Runtime 1.30.0
  `CPUExecutionProvider`, score floor 0.01 and per-class NMS at IoU 0.65. Grayscale PNGs
  are decoded as three equal BGR channels.
- **Evaluator.** `evaluate_detection` (pycocotools 2.0.11), IoU 0.5, at most 100
  detections per frame.
- **Threshold.** Chosen on validation only: the grid value (0.01–0.99) with the highest
  F1, ties to the higher value. The result is **0.16** (validation F1 0.199). The
  development split was scored once at 0.16.
- **Runs.**
  - The baseline ran with 1 thread from a clean tree at `773e899`.
  - The repeat ran with 4 intra-op threads, in a fresh process, from a clean tree at
    `2f249e5`, which differs only in documentation.
  - Median wall time per frame, including PNG decode, was 0.28 s and 0.14 s on a shared
    4-core Xeon VM. These are timings for the budget, not performance claims.

## Results (measured 2026-09-27)

Development split, frozen threshold 0.16, IoU 0.5:

| Metric | Value | Day-bootstrap 95 % interval (31 days) |
|---|---|---|
| AP50 | 0.065 | 0.035 – 0.104 (200 resamples) |
| AP@[.50:.95] | 0.027 | 0.014 – 0.047 (200 resamples) |
| Precision | 0.361 (44 TP, 78 FP) | 0.200 – 0.535 |
| Recall | 0.081 (44 of 544) | 0.048 – 0.117 |
| False positives per negative frame | 0.067 (373 frames, 12 with a false positive) | 0.005 – 0.174 |
| Recall, width [0,16) | 0.000 (support 47) | 0.000 – 0.000 |
| Recall, width [16,32) | 0.007 (support 282) | 0.000 – 0.018 |
| Recall, width [32,64) | 0.069 (support 144) | 0.020 – 0.135 |
| Recall, width [64,inf) | 0.451 (support 71) | 0.281 – 0.631 |

Validation split, for reference (it chose the threshold, so it is not an independent
estimate): AP50 0.119, AP@[.50:.95] 0.072, precision 0.603, recall 0.119 (47 of 395),
0.027 false positives per negative frame (255 frames). Recall at [64,inf) is 0.551
(support 78).

**Repeatability.** Both runs gave bit-identical detections on both splits:
1,451 + 885 detections, no frame with a different count, and a maximum score and box
difference of 0.0. The repeat run's validation threshold is also 0.16. Every
development metric difference is therefore 0.

## Diagnostics

These are not COCO metrics and are not gated.

- **Per category** (development, the other category as ignore regions):
  - airplanes: recall 0.078 (193), AP50 0.047;
  - helicopters: recall 0.083 (351), AP50 0.057.

  COCO has no helicopter class. The detector still finds some helicopters as `airplane`.
- **Missed aircraft covered by another COCO class.** Of the 500 scored aircraft with no
  aircraft detection at IoU ≥ 0.5 and score ≥ 0.16, an unmapped class at or above 0.16
  covers 15: `person` 7, `kite` 5, `bird` 3. The argmax class rule (follow-up 006 A2 F1)
  explains only a small part of the misses on this set.
- **Where false positives lie**, measured after the first results were seen. There
  are 77 detections at or above 0.16 with no scored aircraft at IoU ≥ 0.5 (simple IoU,
  not COCO's greedy matching). Of these:
  - 19 overlap a scored aircraft (localisation mismatch);
  - 25 are on negative frames;
  - 33 are elsewhere on frames with objects;
  - none overlaps an ignore region;
  - 73 are at least 64 px wide.

  Visual inspection (`benchmarks/runs/aeris-007/fp_tp_sheet.jpg`, not in Git) shows two
  patterns. Some are large boxes on clouds or the horizon. Others are detections that
  include a helicopter's rotor or tail, around a label box drawn on the fuselage only.
- **Looser IoU**, also measured after the first results were seen, same frozen
  threshold. At IoU 0.2, development recall is 0.094 (51 of 544) and precision 0.418.
  [64,inf) recall is 0.521. So box-extent convention differences account for 7
  aircraft; tiny targets account for most of the misses.

## Frozen development gate (`configs/evaluation/regression.json`)

A candidate is evaluated on the same development frames with the same evaluator,
IoU 0.5 and score threshold 0.16. It fails when a gated metric is worse than the
baseline by more than its tolerance.

| Metric | Baseline | Tolerance | Basis |
|---|---|---|---|
| AP50 | 0.0648 | 0.001 | AP resolution floor (repeat difference 0) |
| AP@[.50:.95] | 0.0272 | 0.001 | AP resolution floor |
| Precision | 0.3607 | 1/122 = 0.0082 | one detection of 122 at the threshold |
| Recall | 0.0809 | 1/544 = 0.0018 | one object |
| False positives per negative frame (lower is better) | 0.0670 | 1/373 = 0.0027 | one frame |
| Recall [0,16) | 0.000 | 1/47 | one object |
| Recall [16,32) | 0.0071 | 1/282 | one object |
| Recall [32,64) | 0.0694 | 1/144 | one object |
| Recall [64,inf) | 0.4507 | 1/71 | one object |

Every bin has at least 30 scored objects, so all are gated; a bin below 30 would be
reported as not gated. The tolerances are tight because the pipeline is deterministic
on this CPU. The day-bootstrap intervals show that the set cannot tell apart candidates
whose true difference is a few points of recall. So a passing candidate is "no
measurable regression on this frozen set", not "equally good in general".
`aeris.evaluation.baseline.gate_check` applies the gate. The baseline passes its own
gate (tested).

## Reproduce

```
python -m aeris.evaluation.baseline select    # needs data/raw/aot/part{1,2,3}/groundtruth.csv (pinned)
python -m aeris.evaluation.baseline fetch     # 1,889 frames, ETag + SHA-256 checked
python -m aeris.evaluation.baseline run --threads 1 --run-id baseline
python -m aeris.evaluation.baseline run --threads 4 --run-id repeat
python -m aeris.evaluation.baseline report
AERIS_REQUIRE_REAL_BASELINE=1 python tools/check_tests.py tests/unit/test_baseline.py
```

The ground-truth CSVs come from `<bucket_url>/<key>` in the benchmark config. Each run
writes to `benchmarks/runs/aeris-007/` (Git-ignored). `benchmarks/baseline_manifest.json`
records every frame's key, ETag MD5 and SHA-256, the annotation, detection and results
digests, the model and config digests, the code revisions of both runs, the dependency
lock digests and the environment.

The REAL tests re-derive the committed numbers from those files:

- frame hashes;
- selection digests;
- annotation digests;
- the development metrics from the recorded detections;
- the validation threshold;
- an exact re-run of the detector on five development frames.

## Limitations

- **Domain.** AOT is grayscale, filmed from a moving aircraft, over 2019 flights. The
  MVP targets a fixed RGB ground camera. These numbers are not target-domain evidence.
  The owner's pilot or TartanAviation (unreachable here) is still needed.
- **Composition.** Frames are enriched for scored aircraft. Precision and AP would be
  lower on AOT's natural frame mix, which has many more empty frames.
- **Point-like targets are excluded.** About 80 % of AOT aircraft labels are nominal
  markers. The set says nothing about aircraft only a few pixels wide.
- **Label completeness.** AOT labels every airborne object its annotators found, but
  its documentation makes no completeness guarantee. An unlabelled aircraft would count
  as a false positive. The false-positive sheet shows no such case among the sampled
  boxes, but not every false positive was inspected.
- **Box convention.** AOT helicopter boxes often cover the fuselage only. At IoU 0.5
  this costs 7 development aircraft compared with IoU 0.2 (a diagnostic added after the
  results were seen; the gate stays at 0.5).
- **No occlusion or truncation labels.** Those flags are `false`, meaning "not
  labelled".
- **Intervals.** The day bootstrap treats days as independent. The AP intervals use
  200 resamples because each one reruns COCOeval.
- **Timings** come from a shared cloud VM and are not a performance benchmark
  (AERIS-016).
- **Holdout** counts come from labels only. The holdout was not fetched or evaluated.
