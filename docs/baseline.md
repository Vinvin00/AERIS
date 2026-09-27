# Measured baseline and development gate (AERIS-007)

Status: attempt 1, awaiting independent review.

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
- Of 2,013,970 airplane and helicopter boxes, 805,174 are exactly 16 x 16 and 798,357
  exactly 6 x 6. A further 17,239 have a 6 or 16 px side, with the other side no larger
  than 16. These are nominal markers around point-like targets, not extents. 393,200
  boxes have measured extents: median width 25.8 px, and 45,946 are at least 64 px wide.
- No box lies outside the frame or is empty.

## Class policy (`aeris-aircraft-v1`)

| AOT category | AERIS |
|---|---|
| `Airplane`, `Helicopter` with a measured box | class 0, scored |
| `Airplane`, `Helicopter` with a nominal box | class 0, `ignore` region |
| `Airborne` (unidentified), `Drone` | class 0, `ignore` region |
| `Bird`, `BIrd`, `Flock` | left out, so a detection on one is a false positive |

An `ignore` region is a COCO crowd region: a detection matched to it is neither a true
nor a false positive. AOT labels no occlusion or truncation, so those contract flags are
`false`, meaning "not labelled". Every frame is `complete`. The dataset labels every
airborne object annotators found, planned or not, but its documentation makes no
completeness guarantee (see limitations). Timestamps are AOT capture times in seconds
from the sequence's first frame (`timestamp_source = "provided"`).

## Splits and frame selection (frozen before any detector run)

The definition is `configs/evaluation/aot_benchmark.json`, committed in `3985a8f` before
the detector ran on any AOT frame.

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

RESULTS_PLACEHOLDER
