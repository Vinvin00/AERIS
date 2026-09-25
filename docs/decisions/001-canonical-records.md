# ADR 001 — Canonical record representation

Status: implemented for AERIS-001, pending independent task review. Date: 2026-09-24.

Use standard-library frozen dataclasses in `python/aeris/data/contracts.py` with
schema version 1 on frame, annotation, detection and prediction envelopes. BBox and
ground-truth objects inherit the enclosing envelope's version. JSON round trips are
strict: every field is required, unknown fields and duplicate keys are errors, and
nonfinite values and booleans as numbers are forbidden. Sequences and identity-bearing
fields are nonempty whitespace-free strings; class IDs and frame indices are
nonnegative integers. Width and height are positive integers. Geometry uses finite
numbers without rounding the supplied values.

`availability_reason` maps exactly each null field to a nonempty reason. It is copied
and made read-only, as are nested coordinate/matrix sequences. Known values cannot
retain stale unavailability reasons. Camera ID, still-image time, and unannotated track
identity therefore remain explicit unknowns. A prediction may have an unavailable
center and covariance, or a known center with unavailable covariance.

Frame timestamps use `decode_pts`, `inferred_fps`, `provided`, or `unavailable` as the
source vocabulary. Only unavailable timestamps are null. `validate_sequence` checks
one sequence in supplied order: frame indices and available times strictly increase,
indices may skip, and timed and untimed records cannot mix. Annotation class-map
versions and each known track's class remain consistent. IDs can repeat in different
sequences; duplicate track or annotation identities within a frame are forbidden.
Annotation IDs are frame-local. All-null times support image collections but confer
no temporal measurement capability.

Frame image checksums use lowercase SHA-256 hex. This task validates digest syntax;
it does not inspect media, compute the digest, or prove provenance. Image paths are
nonempty strings, not proof that the file exists. Class-map version and model provenance
are explicit references; model manifests and class registry membership are validated
by subsequent tasks. The class-map reference is on annotations/detections, where class
IDs are used. No class names or numeric IDs are guessed by these records.

Ground-truth objects contain no detector score. Detection records contain no labels
or truth flags. `DetectionRecord.validate_frame` checks sequence/index/time/dimensions
when attaching a detection to its frame. Annotation envelopes distinguish complete
empty negative frames from partial empty or unlabeled frames; unlabeled frames cannot
contain truth objects. Downstream code must use `is_negative`, not list emptiness.
These structural boundaries do not authenticate a producer or prevent malicious code
from supplying truth as a detection.

BBox extents are half-open original-frame pixels; zero area, negative extents and
out-of-frame boxes fail. Prediction centers may leave the frame, so they have finite
coordinates but no image clipping. Prediction covariance is a symmetric 2×2 matrix
in px², with nonnegative diagonal and normalized determinant >= -1e-12 (a numerical
PSD tolerance); symmetry is exact. This validates matrix shape/algebra, not calibrated
real-world uncertainty. Forecast horizons are the five shared-contract values, and
target time must exceed issue time and equal issue time plus horizon within 1e-9 s
absolute tolerance. Extremely large timestamps that cannot represent the horizon
are rejected rather than producing a zero-time prediction.

Runnable offline example:

```sh
.venv/bin/python tools/check_tests.py tests/unit/test_contracts.py
```

The fixtures test contracts only. No detector, tracker, dataset loader, metrics,
physical geometry, training, or real-world performance is implemented here.
