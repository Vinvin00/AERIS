# ADR 021 — Refuse `--render` for odd input dimensions

Status: proposed for AERIS-021 review. Date: 2026-09-27.

Context: Codex's supplementary macOS review of the accepted AERIS-008 source (`b00c99c`,
PR #1 comment `5859280428`) found finding **C008-1** (major): AERIS-008's `--render`
overlay silently reported the requested frame size in `run.json.overlay` even when
OpenCV's `mp4v` writer had actually written a smaller file. Their reproducer used a real
FFV1-encoded 65x49 Matroska input on macOS; the coordinator independently reproduced the
same rounding directly against `cv2.VideoWriter` on Linux (measured: 65x49 -> 64x48,
65x48 -> 64x48, 64x49 -> 64x48, 63x47 -> 62x46; 64x48 unaffected), confirming it is
general OpenCV `mp4v`-writer behaviour (rounding down for 4:2:0 chroma subsampling), not
a platform artefact. AERIS-008 is accepted with all three of its attempts used, so this
is a separate follow-up task (AERIS-021) rather than a reopened attempt.

Decision:

- **Refuse rendering an odd width or height.** `Overlay.__init__` now raises
  `PipelineError` before opening the writer (so no file is created) when the decoded
  frame's width or height is odd. The message names the offending size and points to
  running without `--render`. Headless processing (`frames.jsonl`, `detections.jsonl`,
  `timings.json`, `run.json`) is unaffected by input dimension parity; only the optional
  overlay is refused.
- **Test by decoding the written file, not the writer's return value.** The AERIS-008
  regression for the MP4 overlay path (`test_mp4_timestamps_detections_and_overlay`) now
  also asserts that `run.json.overlay`'s width/height equal what `cv2.VideoCapture`
  actually decodes from `overlay.mp4`, not just its frame count — closing the general
  form of C008-1, not only the odd-dimension case. A dedicated test exercises odd width,
  odd height and both odd, and confirms the same input processes fine without `--render`.
- **Odd-dimension test inputs come from the fake-capture seam, not a real encode.**
  OpenCV's own `VideoWriter` cannot be made to emit a genuinely odd-dimension file in
  this build — it performs the identical rounding on write, for every fourcc tried
  (`mp4v`, `MJPG`, `FFV1`, `HFYU`, `RGBA`), across both `.mp4`, `.avi` and `.mkv`
  containers. Codex's own reproducer needed the `ffmpeg` CLI with an explicit
  `-pix_fmt bgr0` to force a genuinely odd-dimension file, which this sandbox has no
  `ffmpeg` binary to reproduce. The tests instead feed an odd-shaped NumPy array directly
  through `FakeCapture` (already used elsewhere in `tests/unit/test_video.py` for cases
  OpenCV cannot write), which exercises `Overlay` exactly as a genuinely odd real file
  would, without needing one.

Alternatives rejected:

- **Pad to the next even size and record the actual encoded geometry.** This would keep
  `--render` working for any input, at the cost of a coordinate transform between the
  frame's true geometry and the overlay's padded canvas that every future consumer of
  `run.json.overlay` would need to know about. Since the overlay is explicitly a display
  aid — true detection and frame geometry always live in `frames.jsonl`/
  `detections.jsonl`, unaffected by this change — refusing costs nothing but the overlay
  video itself, and keeps `run.json.overlay`'s width/height a plain, load-bearing fact
  rather than a value that needs a transform to interpret. This can be revisited if a
  real workload needs odd-dimension overlays.
- **Verifying the actual written geometry after `close()` by reopening the file.** More
  general (it would also catch encoder quirks unrelated to dimension parity), but
  `close()` runs inside the pipeline's `finally` block; an exception raised there could
  mask an exception already being handled from earlier in the loop, or run before
  `run.json`'s failure path can record it, breaking the "run.json is written even when a
  run fails" guarantee. The upfront, known-cause check in `__init__` avoids that risk.

Consequences:

- No accepted AERIS-008 evidence is affected: the Haneda clip is 1920x1080 (both even),
  and no other accepted run used `--render` on an odd-dimension input.
- A future codec change to `OVERLAY_FOURCC` that has different alignment requirements
  (for example, some encoders need dimensions divisible by 16) is not covered by this
  even/odd check; the comment in `Overlay.__init__` names the measurement it is based on
  so a future change can re-verify it.
