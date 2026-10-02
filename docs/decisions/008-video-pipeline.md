# ADR 008 — ffprobe PTS with marked fallback; staged, timed detector interface

Status: proposed for AERIS-008 review. Date: 2026-10-02.

OpenCV's `CAP_PROP_POS_MSEC` derives time from frame index and nominal fps in the
FFmpeg backend, so it cannot reveal variable frame rate and must not be called PTS.
Decode PTS therefore come from `ffprobe`, validated strictly (finite, nonnegative,
strictly increasing, count equal to the decoded frames). If unavailable, timestamps
are inferred from nominal fps and marked `inferred_fps` with a reason, or refused with
policy `pts`. PyAV was not added to avoid another binary dependency; the cost is an
external `ffprobe` and one extra pass over the container.

The detector is a three-method stage interface so preprocess, inference and
postprocess can be timed separately without editing the accepted AERIS-006 module.
Stride skipping, truncation, and failures are explicit in the result. Mixed timestamp
sources are rejected by a helper here; changing `validate_sequence` would be a
contract change and is left to a separate reviewed task.
