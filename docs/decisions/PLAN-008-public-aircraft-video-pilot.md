# PLAN-008 — bounded public aircraft-video pilot

Date: 2026-10-15

## Decision

Add AERIS-031 as a separate, ready evidence task after AERIS-005/006/008/009/010.
It may acquire only QUT's compressed S5 video and corresponding target-centroid
CSV, capped at 2 MB, into Git-ignored `data/raw`. Its outputs are a pinned source
manifest, reproducible runner, actual model observations, and an honest account
of unavailable metrics. No dataset or model is bundled in Git.

## Rationale and boundary

The 2026-09-24 AERIS-002 audit predates this public source. QUT publishes a
BSD-3-Clause fixed, grounded camera aircraft dataset and compressed per-sequence
videos. Its label README calls the target positions *approximate* centroids and
the CSV includes frame numbers and range. These labels can support a narrow
point-error audit if the video-frame offset and target association are verified.
They do not supply exhaustive boxes, pose, calibration or all aircraft identities.
Therefore this pilot cannot activate or satisfy AERIS-021/022, set thresholds,
or validate ENU bearings, live behavior or 3D position.

The accepted detector, tracker and forecaster APIs can be composed in the task's
own script, avoiding dependency on the still-unaccepted AERIS-012 CLI. The
published video and label source may have indexing or encoding differences; the
task must detect and report them rather than silently guessing an offset.

Primary sources: [QUT release](https://open.qcr.ai/dataset/aircraft-collision-course/),
[dataset files](https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/tree/main),
[target label description](https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/resolve/main/Target%20Labels/readme.txt).

This adds a task; it does not change existing acceptance gates, contracts,
baseline thresholds or deferred-task activation. A separate reviewer must
inspect this plan before AERIS-031 is prepared.
