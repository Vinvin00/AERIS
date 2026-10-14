# Integrated Python MVP (AERIS-012)

`aeris INPUT.mp4 --output DIR` (also `python -m aeris`, entry point `aeris` after `pip install -e .`) runs
recorded video -> detections -> tracks -> forecasts and writes a new directory. It composes the accepted
modules without changing them: the AERIS-008 video pipeline, the AERIS-006 YOLOX baseline, the AERIS-009
tracker, the AERIS-010 forecasters, the AERIS-011 camera geometry and the AERIS-005 evaluators.
**It makes no accuracy claim.** The detector operating score, tracker and forecaster settings are untuned
defaults. The recorded-video run below is an artifact smoke check, not a perception benchmark.

## Outputs

| File | Contents |
|---|---|
| `overlay.mp4` | second-pass render of track boxes, IDs, lifecycle (grey tentative, green confirmed, orange lost/predicted) and the 1 s constant-velocity forecast; dimensions and frame count are checked |
| `tracks.csv` | one row per frame and track: state, velocity, size, box (blank if outside the frame), age, time since seen, image heading, speed, angular width/height/rate, `availability_reason` |
| `predictions.jsonl` | contract `PredictionRecord`s for both models at 0.1/0.25/0.5/1/2 s for every non-tentative track at each frame |
| `metrics.json` | counts (always) and truth-dependent metrics (only with labels) |
| `benchmark.json` | per-stage p50/p95/total (decode, preprocess, inference, postprocess, track, forecast), render total, warmup excluded and sample counts, wall clock, hardware |
| `run_manifest.json` | code revision and dirty flag, source content fingerprint, command, input SHA-256, detector/model/config hashes, provider and precision, tracker config hash, calibration, dataset, timestamp policy and source, seed (none, with reason), environment, dependency-lock digests, artifact hashes |

The directory must not exist: runs are immutable. Files are staged in a sibling `.NAME.partial-*`
directory and renamed on success, so a failure leaves nothing behind.

## Availability rules (AC2)

- **No labels:** `metrics.json` has `detection`, `tracking` and `forecast` with `available: false`, a reason
  and null AP/HOTA/IDF1/models; the manifest has `accuracy: null` with a reason. Counts, timings and tracks
  stay available.
- **Labels (`--annotations DIR`, a canonical dataset):** the AERIS-005 evaluators run (COCO AP and width bins,
  TrackEval HOTA/IDF1, forecast errors and interval coverage). Frames must match the run's sequence ID,
  frame indices and sizes or the run fails.
- **No calibration (`--calibration FILE`):** angular columns are blank with `calibration unavailable (pixel-only
  mode)`; image heading and speed still work. Heading is blank with a reason below 1 px/s.
- **No detections:** every artifact is still written (header-only CSV, empty JSONL, zero counts).

## Options

`--detections FILE` replays detections from JSON (`{"frames": {"<index>": [{"bbox": [x0,y0,x1,y1], "class_id": 0,
"score": 0.9}]}}`) instead of running the model: deterministic and weight-free, used by the tests and CI.
`--min-score` (default 0.3, untuned, not validation-selected) filters detections before tracking;
`--stride`, `--max-frames`, `--warmup-frames`, `--timestamp-policy`, `--no-render`, `--sequence-id`,
`--model-config`, `--input-note` (free-text provenance recorded in the manifest).

## Evidence (AC1, AC3)

`tests/integration/test_pipeline.py` (23 tests): the full artifact set and manifest hashes, byte-identical
deterministic replay runs, stride/max-frames/no-render, score filtering, `python -m aeris`, null truth metrics,
the labeled path through the evaluators, pixel-only versus calibrated angular outputs, no detections, empty,
garbage, missing and unreadable inputs, existing output directories (including empty ones), bad options and
calibrations, detector, render and overlay-size/count failures with no partial output, label alignment,
nonfinite score rejection, and the choice of raw detection
centers for the forecasters. A local-only test runs the real baseline on the photo-derived clip.

Local recorded-video smoke run (2026-10-13, `benchmarks/runs/aeris-012-tangier-real-2026-10-13/`, Git-ignored):
60 frames at 10 fps and 640x360, YOLOX-Tiny on ONNX Runtime CPU. The run produced all six artifacts. It recorded
59 detector outputs, 38 above the configured tracking score, one track, 58 track rows and 550 prediction records.
The 58 post-warmup samples had inference p50 0.011894 s and total p50 0.013027 s on this Mac. These are local
smoke measurements, not throughput or accuracy claims. There are no annotations; truth-dependent results are null
with reasons. The camera pans, while the MVP assumes a fixed camera, so the boxes and forecasts demonstrate artifact
generation only. The run manifest records the exact input and artifact hashes.

### Recorded clip provenance and reproduction

The source is [Plane landing at Tangier airport over Sidi Kacem beach](https://commons.wikimedia.org/wiki/File:Plane_landing_at_Tangier_airport_over_Sidi_Kacem_beach.webm),
filmed by Ideophagous on 2024-11-16 and published under CC0 1.0. The original WebM is 66.866 s,
1920x1080, 32,107,848 bytes; SHA-256 `4b6df4c458cc542e296dc469523cbc8513042a3b11244fc2c2cb99cc9ecabc6f`.
The acquisition budget for this task was one rights-cleared source under 35 MB and one six-second excerpt. Store
the downloaded source in ignored `data/raw/sample_videos/tangier-landing-source.webm`, then make the excerpt with:

```sh
ffmpeg -ss 48 -i data/raw/sample_videos/tangier-landing-source.webm -t 6 \
  -vf fps=10,scale=640:360 -an -c:v mpeg4 -q:v 4 \
  data/raw/sample_videos/tangier-landing-6s.mp4
```

The derived MP4 is 321,156 bytes, SHA-256 `0a7a60f544b302981df3c14254944be8637a7e4584eeb34105f4ea9bc11a22e3`.
Run `PYTHONPATH=python ORT_DISABLE_TELEMETRY=1 .venv/bin/python -m aeris
data/raw/sample_videos/tangier-landing-6s.mp4 --output benchmarks/runs/aeris-012-tangier-real-2026-10-13
--sequence-id tangier-landing-smoke --warmup-frames 2` on a checkout with the documented local model.
Neither video is committed to Git. The earlier photo-derived smoke run remains documented in its local manifest;
it is a synthetic-motion test and supplies no real video evidence.

## Limits

Single sequence per run; offline files only (no real-time source, queueing or dropped frames); tracker passes its
exact assigned detection center to both forecasters and passes no new measurement during a coast; angular rate uses consecutive observed
centers; no apparent-size-change labels in the CSV; no camera-motion handling; the forecasters need the track to be
confirmed first; code revision is `null` outside a Git checkout. Tested on macOS arm64, Python 3.14 only; CI runs the
replay tests and skips the real-model test.
