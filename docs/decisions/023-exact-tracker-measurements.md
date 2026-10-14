# AERIS-023 — Exact tracker measurements for forecasts

Status: implementation for independent review. Date: 2026-10-14.

The CLI previously reconstructed a track's measurement by choosing the nearest
same-class detection to its filtered state. That choice can differ from the tracker's
gated Hungarian assignment, particularly when nearby objects have different sizes.
It could also fall back to the filtered state and present it as a fresh detector
measurement. Both paths distort a forecaster's input history.

Each observed `TrackState` now carries the center of the exact deduplicated detection
used for its birth or update. A coasted state carries null. The CLI sends that value
to both causal forecasters and treats an observed state without it as an error.
The tracker still estimates velocity in original-frame px/s from actual timestamp
deltas. No physical speed, real tracking quality or forecast accuracy is inferred.

Tests cover an association where a wrong detection has the nearer center, duplicate
suppression, coasting, and a CLI spy that distinguishes the assigned center from
the raw nearest-center reconstruction. Existing irregular-timestamp and causality
tests remain required gates. Measurement noise is still the untuned fixed 1 px
standard deviation; this task does not calibrate uncertainty.
