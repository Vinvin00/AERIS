# AERIS-024 — Forecast measurement timing and noise

Status: implementation for independent review. Date: 2026-10-14.

Two inputs made the 1 s forecast direction in the live demo unstable.

1. **Timestamps.** The live loop stamped every frame with the monotonic clock at the
   moment it was processed. For a paced file that time includes variable inference and
   pacing delays, so a target moving at constant speed appeared to speed up and slow
   down. Paced files now use media time, `(position - 1) / fps`, where `position` counts
   decoded and skipped frames. Cameras and streams keep monotonic arrival time. The
   strictly-increasing guard previously moved the first frame from 0 to 1e-6 s; it now
   starts from -1 so the first media time is exact.
2. **Measurement noise.** Both forecasters received a fixed 1 px standard deviation,
   far below the jitter of a detector box center. The filter then followed each noisy
   center. The batch CLI and the live loop now use the tracker's own box-relative
   position noise, `max(r_floor_px, r_position_frac * max(w, h))`, from
   `configs/tracker/baseline.json`. The forecaster takes one scalar per update, so the
   larger box side is used.

Both values are untuned engineering choices. No real forecast accuracy, direction
error or calibration claim follows; that needs labeled real video (AERIS-021).
A variable-frame-rate file is timed at its nominal fps in the live loop; the batch CLI
still uses decoder PTS. Tests: `tests/ui/test_web.py::LiveTimingTests`.
