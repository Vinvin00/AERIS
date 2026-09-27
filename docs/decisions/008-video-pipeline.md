# ADR 008 — Timestamp-aware video pipeline on OpenCV's FFmpeg backend

Status: proposed for AERIS-008 attempt 2 review. Date: 2026-09-27.

Context: AERIS-008 connects recorded video to the AERIS-006 detector. Frame times must
come from the container's presentation timestamps, or from an explicitly marked
fallback, and must strictly increase. Stage timings and skipped frames must be recorded.
The owner asked on 2026-09-27 for real data found online.

Decision:

- **Decoder: OpenCV's FFmpeg backend** (`cv2.VideoCapture(path, cv2.CAP_FFMPEG)`),
  already a project dependency through `opencv-python-headless` 5.0.0.93. Its bundled
  FFmpeg is distributed under LGPL-2.1 (the wheel's `LICENSE-3RD-PARTY.txt`) and
  includes no x264 or x265. No new dependency is added.
- **Timestamps.** After each `read()`, `CAP_PROP_POS_MSEC` gives FFmpeg's best-effort
  presentation timestamp of that frame. It is a real per-frame time only when the
  container stores one.
  - `pts` mode therefore accepts only Matroska/WebM, MP4/QuickTime and MPEG-TS,
    identified by magic bytes.
  - It refuses AVI (index × rate), Ogg (frame-count granules), raw elementary streams and
    unknown files, with the `inferred_fps` hint. The attempt-1 review showed FFmpeg
    inventing 25 fps times for a raw MJPEG stream.
  - Frame time is (that value − the first frame's value) / 1000 s. The records label it
    `decode_pts`.
  - A missing, non-finite or negative value fails, and so does a zero or negative
    delta (`TimestampError`). Every decoded frame is checked, including frames skipped
    by stride.
  - On the real clip, every frame time equals the Matroska block time read without
    FFmpeg.
  - The only fallback is explicit: `--timestamps inferred_fps --fps F` gives
    `frame_index / F`, labelled `inferred_fps`. An fps is never used silently in PTS
    mode.
- **Stages.** The AERIS-006 detector runs split into `det.preprocess`, the ONNX model
  and `det.postprocess`, so each stage is timed. A test checks that this split gives
  byte-identical records to the accepted `det.detect` on the same frames.
  - Per processed frame, the pipeline records decode, preprocess, inference,
    postprocess and render wall time (`time.perf_counter`).
  - `total` is one span from the start of the frame's decode to the end of its
    processing. `other` is the rest: timestamp check, pixel hash and record building,
    which attempt 1 left untimed.
  - Throughput is processed frames over the loop's wall time.
  - Summaries give count, mean, P50, P95, P99 and max, excluding one warm-up frame by
    default. Decode time is also summarised over every decoded frame.
  - Frames decoded but not processed are counted by reason (`stride`), and stopping at
    `max_frames` is recorded.
  - An early end is warned about: fewer frames decoded than the container reports.
- **Outputs.**
  - `frames.jsonl` holds contract `FrameRecord`s, `unlabeled`, with the SHA-256 of the
    decoded pixels as the checksum. `detections.jsonl` holds contract
    `DetectionRecord`s.
  - `timings.json` and `run.json` record the input digest, timestamp policy and delta
    summary, counts, detector provenance and providers, environment, code revision,
    output digests, and status or error. `run.json` is written even when a run fails.
  - With `--render`, `overlay.mp4` (mp4v) is written at a constant rate, drawing
    detections with score ≥ 0.3. This is a display choice only.
- **Real sample.**
  - Clip: Wikimedia Commons "Landing aircraft - Haneda - 2021 3 31.webm" by Nesnad,
    CC BY-SA 4.0.
  - Size: 3,381,887 bytes, 1920 x 1080, 392 frames, 1 ms time base.
  - Pinned in `tests/fixtures/video/real_samples.json` by the Commons SHA-1 and a
    SHA-256.
  - `aeris.video.samples` re-checks the licence and hashes through the Commons API
    before storing a copy in `data/raw/video/`. The file is not committed.
- **Fake detector.** `FakeDetector` is a SYNTHETIC deterministic stand-in with a
  documented box formula and an optional `fail_at`. It exercises the pipeline without a
  model.

Alternatives rejected:

- **PyAV 18.1.0.** It exposes exact PTS as rationals. But its manylinux wheel bundles
  libx264 and libx265, which are GPL, and no project licence has been chosen (AGENTS.md:
  do not choose one by accident). It would also add a 35.8 MB dependency. It can be
  revisited once the owner chooses a licence.
- **Inferring time from frame index and nominal fps by default.** It hides
  variable-frame-rate timing and dropped frames. The real sample's 33/34 ms deltas show
  the difference.
- **Committing tiny test videos.** AGENTS.md keeps videos out of Git, so synthetic
  videos are written at test time with OpenCV's `VideoWriter`.

Consequences:

- Time precision follows the container's time base (1 ms for the WebM sample), as
  reported by OpenCV in milliseconds.
- AVI and other containers without per-frame times need the explicit `inferred_fps`
  mode. That is a deliberate refusal to label a derived time as a presentation time.
- OpenCV's `VideoWriter` cannot write variable-frame-rate files. VFR, duplicate,
  backward and missing timestamps are therefore tested through the timestamp policy and
  a fake capture, and on the real clip's uneven deltas, rather than through synthetic
  VFR files.
- The real sample pans with the aircraft. It exercises decoding, timing and output
  geometry, not the fixed-camera assumption of later geometry tasks.
- Timings come from a shared cloud VM. They are records, not performance claims; CPU
  performance is AERIS-016.
