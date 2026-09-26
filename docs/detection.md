# Pretrained detector baseline (AERIS-006)

Status: **attempt 1, not accepted yet.** Implemented and tested:
- preprocessing, postprocessing, class mapping, provenance and the download budget;
- the ONNX Runtime path;
- the PyTorch reference;
- the AERIS ONNX export;
- the PyTorch/ONNX parity check, which passes on the fixed sample.

No real aircraft photo has been processed. This environment's network policy denies the
Commons image hosts, so the real sample is one non-aircraft photo plus derived and
synthetic frames. See "Open work".

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
current upper bounds are 3,040,175,864 bytes downloaded and 4,961,707,445 bytes installed.

| Item | Downloaded | Installed / stored | Fetched |
|---|---|---|---|
| `yolox_s.pth` | 72,089,125 | 72,089,125 | yes |
| upstream `yolox_s.onnx` | 35,858,002 | 35,858,002 | yes |
| YOLOX source (blob-filtered clone) | about 4.7 MB | 9,955,988 | yes, outside the repository |
| onnxruntime + onnx and their dependencies | at most 42.4 MB | 98,259,473 | yes |
| torch + torchvision + 29 dependencies (no triton) | 2,776,101,368 | 4,599,523,860 | yes |
| YOLOX package (local build) | at most 5 MB | 6,123,885 | yes |
| AERIS ONNX export | 0 | 35,897,112 | generated |
| sample images (at most 13, each at most 8 MB) | at most 104 MB | at most 104 MB | only `dog.jpg` (163,759) |

The torch sizes were measured before fetching: download sizes from PyPI, and installed
sizes from each wheel's zip central directory (estimate 4,599,513,601). The installed
size was re-measured afterwards from the dist RECORD files. Its extra 10,259 bytes are
pip's metadata files.

Artifacts live in Git-ignored `models/checkpoints/`, `models/onnx/` and
`data/raw/detector_samples/`. To retrieve them, see `models/README.md`.

## Real sample images

`configs/detector/sample_images.json` lists the sample. `python -m aeris.detection.samples
fetch` downloads it to `data/raw/detector_samples/` and writes a verified `manifest.json`
there. Commons files are accepted only if the Commons API reports public domain, CC0,
CC BY or CC BY-SA (not NC or ND), within 8 MB, with bytes matching Commons' SHA-1. The
record keeps author, licence, source page and hashes. Pinned URLs must match their SHA-256.

- `yolox-dog`: the YOLOX demo photo at the pinned commit (dog, bicycle, truck). It has
  been fetched and verified. The repository does not name the photographer.
- Twelve Commons candidates (aircraft in flight, contrails, final approach, and birds as a
  hard negative). They were shortlisted from web search results only, so their licence
  claims are unverified. They cannot be fetched from this environment (policy 403).

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
- `python -m aeris.detection.export parity --out DIR` exports the model, runs the fixed
  sample through PyTorch, the AERIS export and the upstream export, and writes
  `parity.json`.

**Fixed parity sample.** One real photo, one real-derived frame and two synthetic frames:

| Frame | Kind | Content |
|---|---|---|
| `yolox-dog` | real | 768 x 576 photo |
| `yolox-dog-wide` | real-derived | its rows 144–431 resized to 1000 x 333, a non-integer resize |
| `gray-720p` | synthetic | uniform value 114 |
| `noise-720p` | synthetic | uniform noise, seed 6 |

Every frame's input hash is recorded in the report.

**Measured on 2026-09-26** (single thread; torch 2.14.0 CPU vs ONNX Runtime 1.30.0 CPU):

| Frame | Raw regression max | Raw probability max | Final box max | Matched |
|---|---|---|---|---|
| yolox-dog | 1.31e-4 | 6.91e-6 | 1.25e-4 px | 22 / 22 |
| yolox-dog-wide | 1.36e-4 | 8.52e-6 | 2.28e-4 px | 13 / 13 |
| gray-720p | 1.24e-4 | 8.55e-6 | – (no detections) | 0 / 0 |
| noise-720p | 1.62e-4 | 8.29e-6 | – (no detections) | 0 / 0 |
| declared tolerance | 1e-3 | 1e-4 | 0.1 px | unmatched only at boundaries |

All frames pass, with no unmatched detections. The upstream opset-11 export gives
bit-identical ONNX Runtime outputs to the AERIS export on these frames. Two frames have
no detections, so they check raw outputs only.

**Upstream agreement.** Two REAL tests compare the NumPy postprocessing with YOLOX itself
on the real photo:
- the NumPy decode matches upstream `decode_outputs`;
- the threshold, argmax and per-class NMS give the same detections, classes and scores as
  upstream `yolox.utils.postprocess`, which uses torchvision `batched_nms`.

## Evidence so far

- `tests/unit/test_detection.py` (44 tests):
  - config;
  - byte equality of preprocessing with upstream;
  - the edge-convention inverse;
  - decoding;
  - NMS against a brute-force reference;
  - argmax class mapping and clipping;
  - records;
  - the ONNX Runtime wrapper;
  - the comparison rules;
  - REAL tests: upstream ONNX, the PyTorch reference, upstream agreement, and export
    parity with the recorded export hash.

  The REAL tests skip in CI, which has neither torch nor the weights.
- `tests/unit/test_detection_samples.py` (8 tests): offline fetch checks with a fake HTTP
  layer.
- REAL runs, through PyTorch and both ONNX exports:
  - on `dog.jpg`, the pretrained model finds the dog (0.91), bicycle (0.95) and truck
    (0.61) at their places in the 768 x 576 photo, and nothing maps to aircraft;
  - an overlay shows the boxes on the objects.

  This shows that the runtime paths and the inverse work. It does not show aircraft
  detection.
- The attempt's evidence folder (`orchestration/runs/AERIS-006/attempt-01/evidence/`,
  Git-ignored) holds `parity.json` and the `dog.jpg` detections, candidates and overlay.

## Open work

1. **Aircraft photos (AC1).** The owner allows `commons.wikimedia.org` and
   `upload.wikimedia.org`. Then fetch the Commons candidates, view them, add the fetched
   aircraft photos to the parity sample by SHA-256, and record aircraft detections and
   overlays. The class mapping to AERIS class 0 has so far been exercised only on
   hand-built outputs and in the negative, on the dog photo.
2. The legacy TorchScript exporter is deprecated in PyTorch 2.9 and later. Moving to the
   `torch.export` exporter would need onnxscript and a new parity run.
