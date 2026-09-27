# Timestamp-aware video pipeline (AERIS-008)

Status: attempt 2, awaiting independent review.

`aeris.video.pipeline` decodes a recorded video, gives every frame a validated timestamp,
runs a detector on the frames it processes, times each stage, and writes contract records
and a run manifest. Decisions and alternatives are in ADR 008. This is plumbing: it
measures no detector quality.

```
python -m aeris.video.samples fetch     # the pinned real clip -> data/raw/video/
python -m aeris.video.pipeline run INPUT --out DIR [--detector onnx|fake] [--threads N]
       [--timestamps pts|inferred_fps] [--fps F] [--every N] [--max-frames N] [--render]
```

## Decoding and timestamps (AC1)

Frames are decoded with OpenCV's FFmpeg backend: `cv2.VideoCapture(path, cv2.CAP_FFMPEG)`
from `opencv-python-headless` 5.0.0.93, whose bundled FFmpeg is LGPL-2.1. After each
`read()`, `CAP_PROP_POS_MSEC` is FFmpeg's best-effort presentation timestamp of that
frame.

That value is a real presentation timestamp only when the container stores per-frame
times. Otherwise FFmpeg derives one from a nominal frame rate. The attempt-1 review showed
that a raw MJPEG stream got invented 0/40/80 ms times (FFmpeg's default 25 fps). So
`pts` mode checks the container's magic bytes first:

| Container (magic bytes) | Per-frame times | `pts` mode |
|---|---|---|
| Matroska / WebM (`1A 45 DF A3`) | block timecodes | accepted |
| MP4 / QuickTime (`ftyp`, `moov`, `mdat`, `wide`, `free`, `skip` at byte 4) | sample tables | accepted |
| MPEG-TS (`0x47` sync at bytes 0, 188, 376) | PES PTS | accepted |
| AVI (`RIFF…AVI `) | none: index × rate | refused |
| Ogg (`OggS`) | granule positions (frame counts) | refused |
| anything else (raw streams, unknown) | none | refused |

A refused file stops the run before any output is written. The message names the
container and points to the explicit fallback.

| Mode | Frame time | `timestamp_source` | Fails when |
|---|---|---|---|
| `pts` (default) | (frame value − first frame's value) / 1000 s | `decode_pts` | the container is not listed as accepted, a value is missing, non-finite or negative, or any delta is ≤ 0 |
| `inferred_fps` (explicit) | `frame_index / fps` | `inferred_fps` | no explicit finite fps > 0 is given |

- An fps passed in `pts` mode is refused rather than ignored.
- Every decoded frame is checked, including frames skipped by `--every`. So a duplicated
  or backward timestamp anywhere stops the run.
- OpenCV reports 0 ms for a frame without a timestamp. Within an accepted container, that
  shows up as a zero or negative delta and fails too.
- One run uses one timestamp policy, so all its frame records share one
  `timestamp_source`. Follow-up 001 F4, where `validate_sequence` accepts mixed sources,
  belongs to the owner of `python/aeris/data/`.

On the real clip, every frame time equals the Matroska block time read directly from the
file without FFmpeg (a REAL test).

## Stages, timings and skipped frames (AC2)

For each processed frame, `timings.json` records wall-clock seconds
(`time.perf_counter`):

| Field | What it covers |
|---|---|
| `decode` | `cap.read()` |
| `preprocess` | letterbox (`aeris.detection.baseline.preprocess`) |
| `inference` | the ONNX model (or the fake detector) |
| `postprocess` | decode, threshold, NMS, class map, inverse letterbox |
| `render` | overlay drawing and writing (`null` without `--render`) |
| `other` | `total` minus the named stages: timestamp check, SHA-256 of the pixels, record building |
| `total` | one wall-clock span from the start of the frame's decode to the end of its processing |

- **Summaries.** Count, mean, P50, P95, P99 (NumPy linear percentiles) and max, excluding
  the first processed frame as warm-up (the `warmup_frames` argument of `run()`,
  default 1). Decode time is also summarised over every decoded frame.
- **Throughput.** `processed_frames_per_second` is processed frames over `loop_wall_s`,
  the wall time of the whole decode loop. That includes warm-up, decoding of
  stride-skipped frames and overlay writing.
- **Counts.** `run.json` counts decoded frames, processed frames, frames skipped by reason
  (`stride`), whether the run stopped at `--max-frames`, and detections. It warns when the
  stream ends before the container's reported frame count.
- **Detector split.** The ONNX detector is the accepted AERIS-006 baseline split into its
  three stages. A REAL test checks that the split gives byte-identical detection records
  to `det.detect`.

## Outputs

| File | Content |
|---|---|
| `frames.jsonl` | contract `FrameRecord` per processed frame: sequence ID (input file stem), decoded frame index, timestamp and source, size, `<file>#frame=<i>`, SHA-256 of the decoded BGR pixels, `annotation_status = "unlabeled"` |
| `detections.jsonl` | contract `DetectionRecord`s in original-frame pixels (class map `aeris-aircraft-v1`), with model provenance |
| `timings.json` | per-frame stage times, summaries, loop wall time and throughput |
| `run.json` | status and error, input (path, size, SHA-256, container, reported properties), timestamp summary, counts, settings, detector (provenance, ONNX digest, providers, threads, precision, input size), overlay properties, warnings, environment, code revision with dirty paths, argv, output digests |
| `overlay.mp4` | with `--render`: processed frames with detections scoring ≥ 0.3 drawn (display only), mp4v at nominal fps / `every`; true times are in `frames.jsonl` |

**Errors.**
- **Before output:** bad arguments, a missing or unopenable input, and a container
  without per-frame times raise `PipelineError` before anything is written.
- **During the loop** (timestamp, frame size change, detector error):
  - the pipeline writes `run.json` with `status = "failed"` and the error naming the
    frame, plus the records produced so far;
  - a `PipelineError` or `TimestampError` is then re-raised unchanged;
  - any other exception is raised as a `PipelineError`, with the original as its
    `__cause__`.

## Real sample (AC3)

| | |
|---|---|
| File | Wikimedia Commons [Landing aircraft - Haneda - 2021 3 31.webm](https://commons.wikimedia.org/wiki/File:Landing_aircraft_-_Haneda_-_2021_3_31.webm) |
| Author, licence | Nesnad, CC BY-SA 4.0 (checked through the Commons API) |
| Pins | 3,381,887 bytes; Commons SHA-1 `c201872a…`; SHA-256 `2288c9af…` (`tests/fixtures/video/real_samples.json`) |
| Measured | WebM, 1 ms time base, one video track, 1920 x 1080, nominal 29.97 fps, 392 frames, block times 0–13,046 ms in 33 ms (248) or 34 ms (143) steps |
| Content | ground-level view of an airliner landing; the camera pans with it (not fixed) |

**Provenance.**

1. The copy in `data/raw/video/` was first downloaded on 2026-09-27 with `curl` from
   `upload.wikimedia.org`. A Commons API query had just recorded the licence and
   SHA-1, and the download matched that SHA-1.
2. A first run of `aeris.video.samples fetch`, with no local copy, got HTTP 429 from the
   Commons API. The verified copy was then placed where the tool reuses it by its pinned
   SHA-256.
3. At 15:29 UTC the tool fetched the file end to end into a fresh directory: API
   licence check (`cc-by-sa-4.0`), download, SHA-1 and SHA-256. The result was
   byte-identical to the stored copy (`orchestration/runs/AERIS-008/attempt-02/evidence/fetch-2026-09-27.txt`).

The file is Git-ignored and never committed.

**Full-clip run** (2026-09-27, ONNX detector, 1 thread, `--render`, clean tree at
`dea38e5`, shared 4-core Xeon VM; evidence in
`orchestration/runs/AERIS-008/attempt-02/evidence/haneda-onnx-full/`, not in Git):

- **Counts:** 392 frames decoded and processed, none skipped; `decode_pts` timestamps
  (Matroska); no warnings.
- **Detections:** 4,111 detection records at or above the detector's 0.01 floor. Every
  frame has at least one aircraft detection scoring ≥ 0.3 (maximum 0.93). This is not a
  labelled evaluation.
- **Overlay:** 1920 x 1080, 392 frames.

Stage times in ms per frame, 391 frames after warm-up:

| Stage | P50 | P95 | P99 | Max |
|---|---|---|---|---|
| decode | 2.0 | 3.7 | 5.7 | 25.5 |
| preprocess | 1.8 | 2.3 | 2.9 | 3.9 |
| inference | 228.2 | 259.7 | 273.6 | 295.6 |
| postprocess | 3.5 | 4.7 | 5.1 | 9.3 |
| render | 14.7 | 21.0 | 24.6 | 34.5 |
| other | 17.7 | 23.4 | 27.9 | 29.4 |
| total | 270.0 | 306.3 | 323.2 | 339.2 |

The loop took 107.8 s for 392 frames: 3.64 processed frames per second, far below real
time at 29.97 fps. The `other` row (mostly the 1080p pixel hash) was untimed in attempt
1, whose 3.9 fps figure was overstated. These are records from a shared VM, not a
performance claim (AERIS-016).

## Tests (`tests/unit/test_video.py`)

**Unit tests (SYNTHETIC)** use videos written at test time with OpenCV `VideoWriter` into a
temporary directory; nothing is committed. They cover:

- **Timestamp policy:**
  - PTS seconds and offsets;
  - variable deltas kept;
  - zero, negative and OpenCV-style missing (0 ms) deltas fail at the named frame;
  - None/NaN/inf/negative values fail with a hint;
  - explicit `inferred_fps`, with bad fps values refused;
  - fps refused in `pts` mode.
- **Fake capture seam** (Matroska-tagged input):
  - a duplicate timestamp stops the run and `run.json` records it;
  - a frame size change fails;
  - an early end is warned about, and `--max-frames` is not treated as one;
  - the `inferred_fps` run;
  - exact per-stage, `other`, `total`, loop-wall and throughput values under a scripted
    clock.
- **Containers with the fake detector:**
  - MP4 timestamps at 100 ms and 80 ms steps;
  - `FakeDetector` boxes, scores and gaps against its formula;
  - the frame checksum;
  - overlay size and frame count;
  - output digests;
  - stride and skipped counts;
  - detector failure recorded;
  - AVI and a raw MJPEG stream refused in `pts` mode with the hint, while AVI runs with
    explicit `inferred_fps` (0, 0.2, 0.4, 0.6 s);
  - magic-byte detection for every listed container;
  - missing, unknown, empty and broken-Matroska inputs;
  - bad arguments;
  - overlay writer failure;
  - the CLI.
- **Sample fetcher:** fetch and reuse, and refusals when the licence is not allowed, the
  licence changed, or the SHA-1 or bytes differ; the pinned manifest.

**REAL tests** (`AERIS_REQUIRE_REAL_VIDEO=1` makes skips fail):

- **Container times:** the pinned clip decodes to 392 frames at 1920 x 1080.
  Timestamps are whole milliseconds with only 33 and 34 ms steps. Every one equals the
  file's Matroska block time, read by an independent EBML walker in the test.
  `frame_index / 29.97` would end at 13,046.37 ms, not 13,046 ms.
- **Tampered copies:** frame 10's block time is set equal to frame 9's, or 5 ms before
  it. Both stop the run at frame 10 with a 0.000 ms or −5.000 ms delta, including when
  frame 10 is skipped by stride; `run.json` says failed. A 20 ms later time is kept.
- **Truncated clip:** a copy cut to a third still runs and warns about the early end.
- **ONNX stages:** 10 frames (every 40th) with the real detector give byte-identical
  records to `det.detect`. All boxes are inside the frame, the overlay is 1920 x 1080
  with 10 frames, and all seven timing summaries are present.

## Limitations

- **Timestamps.**
  - OpenCV reports FFmpeg's best-effort presentation timestamp in milliseconds. Within
    accepted containers this equals the stored frame times; it was checked against
    Matroska block times on the real clip, not on MP4 or MPEG-TS real files.
  - The container check uses magic bytes, not a full parse. A damaged file that starts
    correctly is caught only by the open, decode or delta checks.
- **Test coverage.** Variable-frame-rate *files* are not generated in tests (OpenCV
  cannot write them). Uneven timing is covered by the policy tests, the fake capture,
  the tampered real copies and the real clip's 33/34 ms steps.
- **Overlay.** It is constant-rate, for viewing. Only `frames.jsonl` carries true times.
- **Real sample.** It pans, so it exercises decoding, timing and output geometry, not the
  fixed-camera assumption. It is CC BY-SA 4.0 and is fetched, not redistributed.
- **Processing model.** Frames are processed sequentially on one thread. There is no
  queueing or frame dropping under load, so "skipped" means stride only.
- **Timings** are wall-clock on a shared VM, not a benchmark. The full unit check now
  runs in about 90 s against its 120 s timeout (a governance follow-up).
