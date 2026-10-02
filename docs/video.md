# Timestamp-aware video pipeline (AERIS-008)

`python/aeris/video/pipeline.py` decodes a recorded video with OpenCV, runs the
AERIS-006 baseline in three timed stages (or any object with the same `preprocess /
infer / postprocess` methods), and can write an overlay video. It is plumbing: no
tracking, forecasting, CLI or quality claim (the CLI is AERIS-012).

## Timestamps (AC1)

- **Source of truth:** container presentation times from `ffprobe`
  (`best_effort_timestamp_time`, display order, video stream 0), recorded as
  `timestamp_source = "decode_pts"`. They are used as-is, including variable frame
  rate and a nonzero start; nothing is shifted or resampled.
- **Fallback:** if `ffprobe` is missing or reports no PTS and the policy is
  `pts_or_inferred` (default), timestamps are `frame_index / nominal_fps` and marked
  `inferred_fps` with a reason in the result. Policy `pts` refuses instead.
- **Failures (`VideoError`):** non-finite, negative, zero-delta or decreasing
  timestamps; ffprobe frame count different from the decoded count whenever decoding
  reaches the end of the video (also with `max_frames`; a truncated read only requires
  decoded <= ffprobe frames); an `ffprobe` timeout, crash or bad output counts as PTS
  unavailable (fallback or refusal by policy); decoded frames beyond ffprobe's list; unusable frame rate when
  inferring.
- `require_single_timestamp_source` rejects a mixed `decode_pts`/`inferred_fps`
  sequence (follow-up from AERIS-001). The pipeline itself produces one source per
  video; the helper is for callers that merge sequences. `contracts.validate_sequence`
  was not changed (contract change needs its own task).

## Timings and skips (AC2)

Per processed frame: `decode, preprocess, inference, postprocess, render, total`
in seconds (`perf_counter`; `total` is the sum of the other five stages for that
frame, excluding ffprobe and pipeline bookkeeping). The result also carries
per-stage p50/p95/total with the first `warmup_frames` (default 1) excluded and the
sample count, plus whole-run `wall_clock_s`. `stride=n` processes every n-th frame;
the others are decoded, counted in `skipped_frames["stride"]`, and keep their
timestamps (they are still written unannotated to the overlay). Disabled rendering
reports render time 0 with an explicit note. Timings are one-machine wall clock, not
a benchmark. Decode time of stride-skipped frames is not in any per-frame stage; it is only inside `wall_clock_s`. There is no queueing or dropped-frame model for offline files.

## Failure behaviour

Missing/corrupt file, no decodable frames, dimension change mid-stream, non-color
frames, detector exceptions (wrapped with the frame index), detector records that do
not match the frame (index, timestamp, size), an unopenable or unreadable output, and
invalid options all raise `VideoError`. The overlay is read back; its first-frame dimensions and total frame count must
equal the input's. Overlay video uses `mp4v` at the nominal fps, so variable frame
rate timing is not preserved in the rendered file.

## Evidence (AC3)

- `tests/unit/test_video.py` generates a 12-frame 64x48 clip, runs a deterministic
  fake detector (a box at `x = 5 + 2*frame`), and checks output dimensions, record
  fields, timings, stride skips, render read-back, truncation and every failure path
  above, including a variable-frame-rate clip built with `ffmpeg` (frames dropped, so
  the PTS deltas are irregular) and mocked zero/negative/miscounted PTS.
- A local-only test pans a real FGVC aircraft photo (`1143132`, six 640x400 crops,
  10 fps) through the real YOLOX baseline. Summary: `tests/fixtures/video/
  pan_run_summary.json` (one detection per frame, decode_pts source, per-stage
  timings). It verifies plumbing and coordinates, not accuracy. It skips when the
  Git-ignored weights or image are absent, as in CI.
- No owned or rights-cleared real video exists yet; the "real sample" is a clip
  derived from one still photograph and is labelled that way.

## Limits

- `ffprobe` is an external tool, not a pinned Python dependency; tests that need it
  skip when it is missing, and the fallback path is tested by mocking it away.
  Tested on macOS arm64, Python 3.14, OpenCV 5.0.0, ffmpeg from Homebrew.
- Frames are decoded sequentially by OpenCV's FFmpeg backend; its own frame-index to
  PTS agreement is checked only through the count, not frame by frame.
- Memory is O(1) frames except `detections`/`frames` lists in the returned dict.
