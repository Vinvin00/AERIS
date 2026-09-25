# Pretrained detector baseline (AERIS-006)

Status: **attempt 1 in progress, not accepted.** The ONNX Runtime path, preprocessing,
postprocessing, class mapping, provenance and download budget are implemented and tested.
The PyTorch reference, the AERIS ONNX export and the PyTorch/ONNX parity check (AC3) are
not done yet: this environment's network policy denies `download.pytorch.org`, where the
owner-chosen CPU PyTorch wheels live. The Commons image hosts (for aircraft photos) are
denied too. See "Open work".

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
| Cross-check export | Upstream `yolox_s.onnx` from the same release: 35,858,002 bytes, SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, opset 11, exported by PyTorch 1.7. This is only a cross-check; the deliverable is AERIS's own export |

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
- Not locked yet: torch and torchvision CPU wheels, and YOLOX as a VCS pin
  (`yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX@6ddff482…`). YOLOX's
  `setup.py` needs torch installed to build. Its model imports also need loguru, psutil
  and tabulate.

## Download budget

Declared in `configs/detector/baseline.json` before anything was fetched. The upper bound
is 756 MB, against the owner's 4 GB limit.

| Item | Bytes | Fetched |
|---|---|---|
| `yolox_s.pth` | 72,089,125 | yes |
| upstream `yolox_s.onnx` | 35,858,002 | yes |
| YOLOX source (blob-filtered clone) | about 4.7 MB | yes, outside the repository |
| onnxruntime + onnx wheels | 32,447,369 | yes |
| their dependencies | at most 10 MB | yes |
| torch + torchvision CPU | at most 300 MB | no: host denied |
| sample images (at most 12 at 25 MB) | at most 300 MB | only `dog.jpg` (163,759) |

Artifacts live in Git-ignored `models/checkpoints/`, `models/onnx/` and
`data/raw/detector_samples/`. To retrieve them, see `models/README.md`.

## Real sample images

`configs/detector/sample_images.json` lists the sample. `python -m aeris.detection.samples
fetch` downloads it to `data/raw/detector_samples/` and writes a verified `manifest.json`
there. Commons files are accepted only if the Commons API reports public domain, CC0,
CC BY or CC BY-SA (not NC or ND), within 25 MB, with bytes matching Commons' SHA-1. The
record keeps author, licence, source page and hashes. Pinned URLs must match their SHA-256.

- `yolox-dog`: the YOLOX demo photo at the pinned commit (dog, bicycle, truck). It has
  been fetched and verified. The repository does not name the photographer.
- Twelve Commons candidates (aircraft in flight, contrails, final approach, and birds as a
  hard negative). They were shortlisted from web search results only, so their licence
  claims are unverified. They cannot be fetched from this environment (policy 403).

## Evidence so far

- `tests/unit/test_detection.py` (32 tests): config, preprocessing byte equality with
  upstream, the edge-convention inverse, decoding, NMS against brute force, argmax class
  mapping, clipping, records, the ONNX Runtime wrapper, and REAL tests.
- `tests/unit/test_detection_samples.py` (8 tests): offline fetch checks with a fake HTTP
  layer.
- REAL runs with the upstream ONNX export:
  - on `dog.jpg`, the pretrained model finds the dog (0.91), bicycle (0.95) and truck
    (0.61) at their places in the 768 x 576 photo, and nothing maps to aircraft;
  - on a uniform gray frame, all 85 output channels come back with the declared shape.

  These show that the runtime path and the inverse work. They do not show aircraft
  detection.

## Open work (AC1 and AC3)

1. The owner allows `download.pytorch.org` (and, for aircraft photos,
   `commons.wikimedia.org` and `upload.wikimedia.org`) in the environment's network
   settings. The alternative is the 4 GB-budget CUDA wheel from PyPI, which the owner did
   not choose.
2. Install CPU torch/torchvision and YOLOX at the pinned commit, and lock them.
3. PyTorch reference: the upstream model built from `exps/default/yolox_s.py`
   (BatchNorm eps 1e-3), with `decode_in_inference = False`.
4. Export at opset 17 with SiLU replaced by `x * sigmoid(x)`, as upstream
   `tools/export_onnx.py` does, then check the export with ONNX Runtime.
5. Parity on the fixed sample. The tolerances were declared in the config before
   measurement: raw regression 1e-3, probabilities 1e-4, final boxes 0.1 px, scores 1e-4,
   and unmatched detections only at the threshold boundaries.
6. Fetch the Commons candidates, view them, fix the parity sample by SHA-256, and record
   aircraft detections and overlays.
