# Timestamp-aware video pipeline (AERIS-008)

Status: attempt 1, awaiting independent review.

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
frame: the frame's PTS when the stream has one.

| Mode | Frame time | `timestamp_source` | Fails when |
|---|---|---|---|
| `pts` (default) | (frame value − first frame's value) / 1000 s | `decode_pts` | a value is missing, non-finite or negative, or any delta is ≤ 0 |
| `inferred_fps` (explicit) | `frame_index / fps` | `inferred_fps` | no explicit finite fps > 0 is given |

- OpenCV reports 0 ms for a frame without a timestamp, so such a stream fails at its
  second frame in `pts` mode. The error suggests the explicit fallback.
- An fps passed in `pts` mode is refused rather than ignored.
- Every decoded frame is checked, including frames skipped by `--every`. So a
  duplicated or backward timestamp anywhere in the stream stops the run.
- `run.json` records the first timestamp and the minimum, median and maximum deltas.

## Stages, timings and skipped frames (AC2)

For each processed frame, `timings.json` records wall-clock seconds
(`time.perf_counter`) for:

- `decode` (`cap.read()`);
- `preprocess` (letterbox, `aeris.detection.baseline.preprocess`);
- `inference` (the ONNX model, or the fake detector);
- `postprocess` (decode, threshold, NMS, class map, inverse letterbox);
- `render` (overlay drawing and writing, or `null` without `--render`);
- `total`, the sum.

Summaries give count, mean, P50, P95, P99 (NumPy linear percentiles) and max, excluding
the first processed frame as warm-up (the `warmup_frames` argument of `run()`, default 1). Decode
time is also summarised over every decoded frame. `run.json` counts:

- decoded and processed frames;
- skipped frames by reason (`stride` for frames decoded but not processed);
- whether the run stopped at `--max-frames`;
- detections.

It also warns when the stream ends before the frame count the container reports.

The ONNX detector is the accepted AERIS-006 baseline split into its three stages. A REAL
test checks that the split gives byte-identical detection records to `det.detect` on the
same frames.

## Outputs

| File | Content |
|---|---|
| `frames.jsonl` | contract `FrameRecord` per processed frame: sequence ID (input file stem), decoded frame index, timestamp and source, size, `<file>#frame=<i>`, SHA-256 of the decoded BGR pixels, `annotation_status = "unlabeled"` |
| `detections.jsonl` | contract `DetectionRecord`s in original-frame pixels (class map `aeris-aircraft-v1`), with model provenance |
| `timings.json` | per-frame stage times and summaries |
| `run.json` | status and error, input path/size/SHA-256/container properties, timestamp summary, counts, settings, detector (provenance, ONNX digest, providers, threads, precision, input size), overlay properties, warnings, environment, code revision with dirty paths, argv, output digests |
| `overlay.mp4` | with `--render`: processed frames with detections scoring ≥ 0.3 drawn (display only), mp4v at nominal fps / `every`; true times are in `frames.jsonl` |

If a run fails (timestamp, frame size change, detector error), the pipeline still writes
`run.json` with `status = "failed"` and the error naming the frame. It also writes the
records produced so far, then raises `PipelineError` (or its subclass `TimestampError`).

## Real sample (AC3)

| | |
|---|---|
| File | Wikimedia Commons [Landing aircraft - Haneda - 2021 3 31.webm](https://commons.wikimedia.org/wiki/File:Landing_aircraft_-_Haneda_-_2021_3_31.webm) |
| Author, licence | Nesnad, CC BY-SA 4.0 (checked through the Commons API) |
| Pins | 3,381,887 bytes; Commons SHA-1 `c201872a…`; SHA-256 `2288c9af…` (`tests/fixtures/video/real_samples.json`) |
| Measured | WebM, 1920 x 1080, nominal 29.97 fps, 392 frames, container times 0–13,046 ms in 33 or 34 ms steps |
| Content | ground-level view of an airliner landing; the camera pans with it (not fixed) |

`aeris.video.samples fetch` re-checks the licence and both hashes before storing the file
in `data/raw/video/` (Git-ignored). The copy used here was downloaded on 2026-09-27 and
checked against the Commons SHA-1. A later fetch hit HTTP 429, and the tool then reused
the verified copy by its pinned SHA-256.

**Full-clip run** (2026-09-27, ONNX detector, 1 thread, `--render`, shared 4-core Xeon VM;
evidence in `orchestration/runs/AERIS-008/attempt-01/evidence/haneda-onnx-full/`, not
in Git):

- **Counts:** 392 frames decoded and processed, none skipped; `decode_pts` timestamps
  with deltas 33–34 ms; no warnings.
- **Detections:** 4,111 detection records at or above the detector's 0.01 floor. Every
  frame has at least one aircraft detection scoring ≥ 0.3 (maximum 0.93). This is not a
  labelled evaluation.
- **Overlay:** 1920 x 1080, 392 frames.

Stage times in ms per frame, 391 frames after warm-up:

| Stage | P50 | P95 | P99 | Max |
|---|---|---|---|---|
| decode | 2.1 | 3.4 | 5.0 | 6.1 |
| preprocess | 1.8 | 2.4 | 3.1 | 5.5 |
| inference | 229.4 | 257.5 | 275.5 | 287.0 |
| postprocess | 3.5 | 4.7 | 5.7 | 8.9 |
| render | 14.8 | 19.9 | 22.5 | 26.8 |
| total | 253.2 | 280.0 | 303.4 | 319.4 |

That is about 3.9 processed frames per second, far below real time at 29.97 fps. These
are records from a shared VM, not a performance claim (AERIS-016).

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
- **Fake capture seam:**
  - a duplicate timestamp stops the run and `run.json` records the failure and partial
    output;
  - a frame size change fails;
  - an early end is warned about, and `--max-frames` is not treated as one;
  - the `inferred_fps` run;
  - exact stage timings under a scripted clock.
- **Real containers with the fake detector:**
  - AVI timestamps at 0, 100, … 800 ms, and MP4 at 80 ms steps;
  - `FakeDetector` boxes, scores and gaps against its formula;
  - the frame checksum;
  - overlay size and frame count;
  - output digests;
  - stride and skipped counts;
  - detector failure recorded;
  - missing, garbage and empty inputs;
  - bad arguments;
  - overlay writer failure;
  - the CLI.
- **Sample fetcher:** fetch and reuse, and refusals when the licence is not allowed, the
  licence changed, or the SHA-1 or bytes differ; the pinned manifest.

**REAL tests** (`AERIS_REQUIRE_REAL_VIDEO=1` makes skips fail):

- **Container timestamps:** the pinned clip decodes to 392 frames at 1920 x 1080.
  Timestamps are whole milliseconds with only 33 and 34 ms steps, and end at 13,046 ms.
  Inferring them from 29.97 fps would give 33.37 ms steps and 13,046.37 ms.
- **Truncated clip:** a copy cut to a third still runs and warns about the early end.
- **ONNX stages:** 10 frames (every 40th) with the real detector give byte-identical
  records to `det.detect`. All boxes are inside the frame, the overlay is 1920 x 1080
  with 10 frames, and all six stage summaries are present.

## Limitations

- OpenCV reports FFmpeg's best-effort presentation timestamp in milliseconds. It cannot
  tell a genuine 0 ms timestamp on the first frame from a missing one; only the second
  frame's delta exposes a stream without timestamps. PyAV would give exact PTS but is
  not used (ADR 008).
- Variable-frame-rate *files* are not generated in tests (OpenCV cannot write them).
  Uneven timing is covered by the policy tests, the fake capture and the real clip's
  33/34 ms steps.
- The overlay is constant-rate for viewing. Only `frames.jsonl` carries true times.
- The real sample pans. It is licensed for reuse under CC BY-SA 4.0 with attribution,
  and is fetched, not redistributed.
- The pipeline processes frames sequentially on one thread. There is no queueing or
  frame dropping under load, so "skipped" means stride only.
- Timings are wall-clock on a shared VM, not a benchmark.
