# Pretrained detector baseline (AERIS-006)

Status: **attempt 2, awaiting independent review.**

The review of attempt 1 escalated:
- AC2 and AC3 passed;
- AC1 failed because no real aircraft photo was available;
- six minor findings were raised.

Attempt 2 fixes the findings. It also adds real aircraft evidence, since the owner
allowed the Commons hosts on 2026-09-26: the pretrained model finds every required
aircraft photo, and each detection maps to an AERIS class 0 record on the aircraft. See
"Real aircraft results".

Nothing here is trained or tuned on AERIS data, and nothing here measures detector
quality on aircraft. The config (`configs/detector/baseline.json`) is the single source of
settings; `python/aeris/detection/baseline.py` refuses settings it does not implement.

## Choice and terms

| Item | Value |
|---|---|
| Model | YOLOX-S (depth 0.33, width 0.50, SiLU), 80 COCO classes, 640 x 640 input |
| Code | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (2025-06-08, package version 0.3.0), Apache-2.0 |
| Weights | `yolox_s.pth` from release `0.1.1rc0`, 72,089,125 bytes, SHA-256 `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` |
| Weight terms | Release assets of the Apache-2.0 repository. The repository states no separate weight terms. Upstream trained on COCO train2017. COCO images carry per-image Flickr licences and the annotations CC BY 4.0 |
| Upstream accuracy | 40.5 AP@[.50:.95] on COCO val2017 (README table; not measured by AERIS) |
| AERIS export | `models/onnx/yolox_s_aeris_opset17.onnx`, 35,897,112 bytes, SHA-256 `e68784f4593e8fbcae404d97da7c99f06e3970c7c56b17c0f041bfdde6558aef`, opset 17. It was exported by PyTorch 2.14.0 with the TorchScript-based exporter (`dynamo=False`), and a second export was byte-identical |
| Cross-check export | Upstream `yolox_s.onnx` from the same release: 35,858,002 bytes, SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, opset 11, exported by PyTorch 1.7. It is only a cross-check |

The owner chose YOLOX on 2026-09-25 over Ultralytics YOLO (AGPL-3.0 or enterprise
licence) and the torchvision detectors (weights host blocked here). YOLOX-S is the
smallest standard model evaluated at 640 px, which keeps small aircraft larger in the
model input than the 416 px Nano and Tiny variants. Nano and Tiny are the fallback if CPU
latency turns out too high. Latency has not been measured.

## Input and output

- **Input.** An `H x W x 3` uint8 **BGR** frame, as OpenCV decodes it. Scale
  `r = min(640 / H, 640 / W)`. Resize with `cv2.INTER_LINEAR` to
  `floor(W r) x floor(H r)`, place it top-left in a 640 x 640 canvas filled with 114, and
  pass it as float32 NCHW, 0–255, without mean/std normalisation (upstream non-legacy
  `ValTransform`). The tests check byte equality with a verbatim copy of upstream
  `preproc`.
- **Output.** `[1, 8400, 85]`: anchors for strides 8, 16 and 32 (80², 40², 20² cells,
  row-major). Channels 0–1 are grid offsets, 2–3 log sizes, 4 objectness probability and
  5–84 class probabilities. The model is exported with `decode_in_inference = False`, as
  upstream does, and decoded in NumPy:
  `cx = (dx + x) s`, `cy = (dy + y) s`, `w = exp(lw) s`, `h = exp(lh) s`.

## Postprocessing and class mapping

The steps follow upstream `yolox.utils.postprocess`, with the COCO evaluation defaults
from `yolox/exp/yolox_base.py`:

1. Score = objectness × the largest class probability. Class = argmax.
2. Keep score ≥ 0.01.
3. Greedy per-class NMS: suppress IoU > 0.65, order by score, ties by anchor index. The
   tests compare it with a brute-force reference.
4. Map classes with `aeris-aircraft-v1`: COCO index 4 (`airplane`, category id 5) →
   AERIS class 0 `aircraft`. Every other class is dropped, including `bird` and `kite`.
   Because the class is the argmax, an aircraft whose top class is `bird` is dropped.
   This is upstream behaviour, and is recorded as a limitation.
5. Invert the letterbox into original-frame pixels, clip to `[0, W] x [0, H]`, and drop
   boxes left empty (for example, boxes wholly in the padding).

There is no per-frame cap; evaluators apply their own top-k budget. Neither threshold is
validation-selected. AERIS-007 owns choosing the operating threshold (002 N3/F3).

**Inverse.** `cv2.resize` maps pixel edges linearly, which a test confirms on a gradient
image. So `x = x_input × W / floor(W r)` and `y = y_input × H / floor(H r)` is the exact
inverse into the contract's half-open, original-frame coordinates. Upstream divides both
axes by `r`. When `W r` or `H r` is not an integer, that is off by less than one resized
pixel. For a 1000 x 333 frame, the full content height maps to 332.81 px instead of 333.

Detections become contract-v1 `DetectionRecord`s (`to_records`). `model_provenance` names
the config, the YOLOX commit, the weights' SHA-256, the ONNX file's SHA-256 and the
runtime.

## Runtime and dependency locks

- ONNX Runtime 1.30.0 (MIT), CPU execution provider only. `OnnxModel` refuses any other
  provider list and checks the declared input and output shapes. Threads: 1 intra-op,
  1 inter-op.
- `requirements/detector-linux-x86_64-py311.txt`: hash-pinned onnxruntime 1.30.0, onnx
  1.23.0 and their dependencies. The `detector` and `dev` extras carry the same pins, so
  CI runs the ONNX Runtime tests with a generated constant graph. CI has no weights, so
  the REAL tests skip there.
- `requirements/detector-torch-linux-x86_64-py311.txt` locks the PyTorch reference path,
  hash-pinned:
  - torch 2.14.0 and torchvision 0.29.0 from PyPI;
  - their CUDA 13 libraries;
  - loguru, psutil and tabulate (YOLOX's model imports).

  It is installed with `--no-deps`, so triton is left out, and `--no-compile`.
  `download.pytorch.org` (CPU-only wheels) is denied here. On 2026-09-26 the owner chose
  PyPI's wheel without triton and raised the budget to 5 GB. The CUDA libraries are
  unused on CPU, but torch does not import without them.
- YOLOX is built from the pinned commit with
  `pip install --no-deps --no-build-isolation "yolox @ git+…@6ddff482…"`, because its
  `setup.py` needs torch to build. `aeris.detection.export` then checks the commit pip
  recorded and every installed YOLOX Python file against pip's RECORD hashes.

## Download budget

The budget is declared in `configs/detector/baseline.json`, and every item was recorded
before it was fetched. The owner's limit rose from 4 GB (2026-09-25) to 5 GB (2026-09-26).
It applies separately to total downloads and to total installed or stored bytes. The
current upper bounds are 2,988,175,864 bytes downloaded and 4,969,707,445 bytes installed.
That leaves 30 MB of installed headroom.

| Item | Downloaded | Installed / stored | Fetched |
|---|---|---|---|
| `yolox_s.pth` | 72,089,125 | 72,089,125 | yes |
| upstream `yolox_s.onnx` | 35,858,002 | 35,858,002 | yes |
| YOLOX source (blob-filtered clone) | about 4.7 MB | 9,955,988 | yes, outside the repository |
| onnxruntime + onnx and their dependencies | at most 42.4 MB | 98,259,473 | yes |
| torch + torchvision + 29 dependencies (no triton) | 2,776,101,368 | 4,599,523,860 | yes |
| YOLOX package (local build) | at most 5 MB | 6,123,885 | yes |
| bytecode Python writes on import (torch path) | 0 | at most 60,000,000 (48,717,539 measured) | generated |
| AERIS ONNX export | 0 | 35,897,112 | generated |
| sample images (at most 13, each at most 4 MB) | at most 52 MB | at most 52 MB | 10 files, 9,152,580 bytes |

The torch sizes were measured before fetching: download sizes from PyPI, and installed
sizes from each wheel's zip central directory (estimate 4,599,513,601). The installed
size was re-measured afterwards from the dist RECORD files. Its extra 10,259 bytes are
pip's metadata files. `--no-compile` only defers bytecode, which Python writes when a
module is first imported. Attempt 1's bound left it out, so the attempt-1 review found
the true upper bound was 5,010,424,984 bytes, above the limit. Attempt 2 counts bytecode
with a 60 MB allowance and cuts the image allowance to thumbnail-sized files.

Artifacts live in Git-ignored `models/checkpoints/`, `models/onnx/` and
`data/raw/detector_samples/`. To retrieve them, see `models/README.md`.

## Real sample images

`configs/detector/sample_images.json` lists the sample. `python -m aeris.detection.samples
fetch` downloads it to `data/raw/detector_samples/` and writes a verified `manifest.json`
there. A Commons file is accepted only if the Commons API reports public domain, CC0,
CC BY or CC BY-SA (not NC or ND). The fetcher asks for Commons' own 2048 px rendition,
which is enough, because the detector letterboxes to 640 px:
- if Commons serves the original (no wider than 2048 px), its bytes must match the SHA-1
  Commons records;
- otherwise the rendition's SHA-256 is recorded, together with the original's SHA-1 and
  the rendition URL.

Files are capped at 4 MB. The record keeps author, licence, source page and hashes.
Pinned URLs must match their SHA-256. Once fetched, the aircraft photos join
`parity_sample.real` in `configs/detector/baseline.json`, pinned by SHA-256.

Fetched on 2026-09-26 (10 of 13 entries). The licences and authors are as the Commons
API reported them:

| Id | Licence | Author | Stored as | Size |
|---|---|---|---|---|
| yolox-dog | Apache-2.0 repository (photographer not named) | YOLOX repository | pinned URL | 768 x 576 |
| commons-takeoff-blue-sky | public domain | Rosendahl | original, SHA-1 checked | 1280 x 915 |
| commons-airplanes-jets | public domain | Jon Sullivan | original | 1280 x 961 |
| commons-airplane-in-flight | CC BY 3.0 | Fitrah 9131 | original | 3060 x 4080 |
| commons-blue-sky-with-plane | CC BY 2.0 | Bradley Gordon | Commons rendition | 3840 x 2560 |
| commons-a380-blue-sky | CC BY 2.0 | Flickr user Axwel | original | 761 x 428 |
| commons-747-undercarriage | public domain | Arpingstone | original | 1500 x 1125 |
| commons-f15e-sky | public domain | U.S. Air Force photo / Staff Sgt. Aaron Allmon | original | 3000 x 1996 |
| commons-747-prototype | public domain | unknown author | Commons rendition | 3840 x 2321 |
| commons-birds-sky | CC BY-SA 3.0 | Augustus Binu / dreamsparrow.net | original | 3604 x 2158 |

Notes:
- Wikimedia snapped the requested 2048 px rendition width to its standard 3840 px size.
  Both renditions are about 1.0–1.6 MB, and a second fetch returned identical bytes.
- Three candidates (777 contrails, C-141 contrail, DCA final approach) were refused
  because upload.wikimedia.org rate-limited the original downloads (HTTP 429). They were
  not retried further.
- The fetcher paces requests and honours `Retry-After`. Wikimedia rate-limits by client
  IP, and this environment's egress IP is shared.
- The stored images total 9,152,580 bytes.

## Real aircraft results

**Hand-drawn regions first.** The aircraft regions in
`tests/fixtures/detection/aircraft_regions.json` were drawn from gridded views of the
photos and committed (`88cb3ca`) before any detector ran on them. So was the rule that
decides which photos must produce a detection: the aircraft is whole, and its region is
at least 32 px wide in the 640 px model input, COCO's small-object boundary. Such a
photo must yield an AERIS class 0 record, from COCO `airplane`, whose box has IoU ≥ 0.5
with the region (`RealAircraftTests`).

AERIS export on ONNX Runtime, 2026-09-26 (evidence: `attempt-02/evidence/*.overlay.jpg`
and `*.detections.jsonl`):

| Photo | Required | AERIS aircraft records | Best score | IoU with region |
|---|---|---|---|---|
| commons-takeoff-blue-sky | yes | 1 | 0.958 | 0.970 |
| commons-airplanes-jets | yes | 1 | 0.904 | 0.909 |
| commons-a380-blue-sky (non-integer resize) | yes | 1 | 0.904 | 0.939 |
| commons-f15e-sky (desert ground behind) | yes | 1 | 0.912 | 0.879 |
| commons-747-prototype | yes | 2 | 0.883 | 0.973 |
| commons-747-undercarriage (cut off by the frame) | no | 3 | 0.919 | 0.875 |
| commons-blue-sky-with-plane (about 30 px in the input) | no | 0 | – | – |
| commons-airplane-in-flight (about 13 px in the input) | no | 0 | – | – |
| commons-birds-sky (no aircraft; 207 bird detections) | no | 0 | – | – |
| yolox-dog (no aircraft) | no | 0 | – | – |

All five required photos pass. The two small aircraft are missed at the 640 px input.
That is the expected small-target weakness this project has to address later (for
example AERIS-017); it is not a defect of this task. The bird photo produces no false
aircraft here. That is one photo, not a false-positive rate.

## PyTorch reference, export and parity

`python/aeris/detection/export.py` holds the PyTorch side.

- `TorchModel` builds the model as upstream `exps/default/yolox_s.py` does: BatchNorm eps
  1e-3, with the exp values checked against the config. It loads the SHA-256-checked weights
  with `torch.load(weights_only=True)` and returns raw head outputs
  (`decode_in_inference = False`). It refuses a YOLOX install whose commit or files differ
  from the pin.
- `export_onnx` works as upstream `tools/export_onnx.py` does: SiLU is replaced by
  `x * sigmoid(x)`, there is no decode or simplifier, the batch is 1, the axes are static,
  and the opset is 17. The export is then checked with `onnx.checker`.
- `compare` checks one frame in two stages:
  - raw outputs: the largest absolute differences in the regression and probability
    channels;
  - final detections: both raw outputs go through the same NumPy postprocessing. The
    detections are matched by (anchor, class) in clipped original-frame pixels. Unmatched
    detections pass only at the declared score or NMS boundary (`boundary_abs` 0.001).
- `python -m aeris.detection.export parity --out DIR [--onnx NEW_PATH]` does four things:
  - exports the model to a temporary file, or to `NEW_PATH` if given; it refuses to
    overwrite an existing file, including the recorded artifact;
  - runs the fixed sample through PyTorch, the AERIS export and the upstream export;
  - reports whether the export's SHA-256 matches the recorded one;
  - writes `parity.json`.

**Fixed parity sample.** 13 frames:
- the 10 real photos above, pinned by SHA-256 in `parity_sample.real`, 8 of them with
  aircraft;
- one real-derived frame: `yolox-dog` rows 144–431 resized to 1000 x 333, a non-integer
  resize;
- two synthetic frames: uniform value 114, and uniform noise with seed 6.

Every frame's input hash is recorded in the report.

**Measured on 2026-09-26** (single thread; torch 2.14.0 CPU vs ONNX Runtime 1.30.0 CPU).
The maxima over the 13 frames:

| Measure | Max | Declared tolerance |
|---|---|---|
| raw regression | 1.75e-4 | 1e-3 |
| raw probability | 1.18e-5 | 1e-4 |
| final box | 5.58e-4 px | 0.1 px |
| final score | 1.41e-6 | 1e-4 |
| detections | 277 / 277 matched, 0 unmatched | unmatched only at boundaries |

All 13 frames pass, including every frame with aircraft. The upstream opset-11 export
also passes. The re-export's SHA-256 matches the recorded export. The per-frame results
are in `attempt-02/evidence/parity.json`.

**Upstream agreement.** Two REAL tests compare the NumPy postprocessing with YOLOX itself
on the real photo:
- the NumPy decode matches upstream `decode_outputs`;
- the threshold, argmax and per-class NMS give the same detections, classes and scores as
  upstream `yolox.utils.postprocess`, which uses torchvision `batched_nms`.

## Evidence so far

- `tests/unit/test_detection.py` (54 tests):
  - config, including the mapped-class names and the record class id;
  - byte equality of preprocessing with upstream;
  - the edge-convention inverse;
  - decoding;
  - NMS against a brute-force reference;
  - argmax class mapping and clipping;
  - records;
  - the ONNX Runtime wrapper;
  - the comparison rules; each loosening the attempt-1 review showed would survive now
    fails a test (boundary width, cross-class exemption, objectness, final score, and
    ONNX actually compared);
  - REAL tests: upstream ONNX, the PyTorch reference, class names against the pinned
    upstream file, upstream decode/postprocess agreement, export parity (nonzero
    differences, recorded hash), and `RealAircraftTests`.

  `RealAircraftTests` requires, for each aircraft photo, an AERIS class 0 record from COCO
  `airplane` whose box overlaps a hand-checked region by IoU of at least 0.5. The regions
  are in `tests/fixtures/detection/aircraft_regions.json`.

  The REAL tests skip where their inputs are absent, as in CI. With
  `AERIS_REQUIRE_REAL_DETECTOR=1` every such skip fails instead. The gate itself does not
  set this variable; changing the gate is outside this task's scope.
- `tests/unit/test_detection_samples.py` (15 tests): offline fetch checks with a fake HTTP
  layer. They cover the rendition path, tracked original URLs and `Retry-After`
  handling.
- REAL runs, through PyTorch and both ONNX exports:
  - on `dog.jpg`, the pretrained model finds the dog (0.91), bicycle (0.95) and truck
    (0.61) at their places in the 768 x 576 photo, and nothing maps to aircraft;
  - an overlay shows the boxes on the objects.

  This shows that the runtime paths and the inverse work.
- REAL aircraft runs: see "Real aircraft results".
- The attempt evidence folders (`orchestration/runs/AERIS-006/attempt-0N/evidence/`,
  Git-ignored) hold `parity.json` and the `dog.jpg` detections, candidates and overlay.

## Open work

1. A gate change (a separate task): the `detection` check should set
   `AERIS_REQUIRE_REAL_DETECTOR=1`, so REAL skips cannot pass silently.
2. Small aircraft (below about 32 px in the 640 px input) are missed. Tiling or a higher
   input resolution is future work, not a defect of this task.
3. The legacy TorchScript exporter is deprecated in PyTorch 2.9 and later. Moving to the
   `torch.export` exporter would need onnxscript and a new parity run.
