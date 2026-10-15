# AERIS-029 — Additive surveyed-site bearing contract

Status: implementation for independent review. Date: 2026-10-15.

## Decision and migration

Add `FixedCameraPose` and `WorldBearingRecord` version 1 as a separate optional
output family. Preserve every existing frame, track and pixel prediction field,
coordinate convention and JSON schema. Consumers must opt in to bearing records;
they must not reinterpret a pixel center as an ENU position. This reviewed task is
the contract-change task required by `docs/CONTRACTS.md`.

The ENU reference is a named local site-survey origin; its geodetic coordinates
are outside this contract. The pose supplies the camera origin in metres relative
to that reference and maps camera right/down/forward axes into east/north/up. It
records a UTC survey time and a Boolean operator assertion that the mount remains
fixed. This assertion is necessary input, not a measured stability guarantee.
Each conversion also requires the current zoom and an explicit per-frame
`camera_stable` assessment. `FrameContext.camera_moved=False` is its default and
cannot certify stability by itself; missing assessment returns null.

Available records contain only a unit direction, azimuth/elevation, camera origin
and timestamps. Azimuth is clockwise from north in `[0,360)` degrees and elevation
is above the horizon in `[-90,90]` degrees. Observation horizon is zero; a
forecast target time is issue time plus a positive horizon. A forecast conversion
accepts a causal predicted original-frame pixel center and does not access future
frames. No range, target 3-D position, aircraft yaw or calibrated angular interval
is returned. A missing input produces null geometry with a reason.

## Validation and next step

Analytic pinhole and distorted-camera tests cover coordinate conversion, times,
record round trips and rejection of invalid poses and unavailable cases. They do
not establish physical angular accuracy. AERIS-030 is the separate live-loop
binding; it waits for the scene-relative forecast prerequisites. Real calibration,
surveyed reference bearings and a physical camera test remain necessary before
calling a displayed value a validated live bearing.
