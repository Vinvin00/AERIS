# ADR 011 — Fixed-camera ray geometry with explicit transforms and suppression

Status: proposed for AERIS-011 review. Date: 2026-10-05.

Use OpenCV only for the iterative undistortion and forward projection; verify it in tests
against an independent NumPy Brown-Conrady forward model and closed-form pinhole values
from the AERIS-004 fixture. Crops, resizes and padding are an explicit `ImageTransform`
object, never implicit. Angular extent is the angle between mid-edge rays and heading is
`atan2(-vy, vx)`, as CONTRACTS.md specifies. Angular claims are suppressed (value `None`,
reason recorded) without calibration, after camera motion, or on a zoom mismatch; size
change is only ever an apparent-size trend with an uncertainty-aware label, so it cannot be
read as range or approach. Fisheye and rational models are rejected rather than approximated.
Thresholds are untuned defaults; no real calibration exists to validate against.
