# ADR 006 — Pretrained detector baseline and CPU ONNX path

Status: proposed for AERIS-006 attempt 2 review. Dates: 2026-09-25 and 2026-09-26.

Decision:

- **Model.** YOLOX-S at 640 x 640, from the official repository at commit `6ddff482`
  (Apache-2.0), with release `0.1.1rc0` weights `yolox_s.pth`
  (SHA-256 `f55ded71…913a30`). It is used as a pretrained COCO detector without training.
  The owner chose YOLOX on 2026-09-25 for its Apache-2.0 code and weight terms, its COCO
  `airplane` class and its upstream ONNX export.
- **Class mapping.** `aeris-aircraft-v1`: COCO index 4 (`airplane`) → AERIS 0. Classes
  come from the upstream argmax, and every other class is dropped.
- **Pre/postprocessing.** Upstream non-legacy preprocessing: BGR, 0–255, bottom/right
  pad of 114, `INTER_LINEAR`. Upstream COCO-evaluation postprocessing: score ≥ 0.01,
  per-class NMS at IoU > 0.65. Both run in NumPy on raw head outputs
  (`decode_in_inference = False`), so the PyTorch reference and the ONNX export share
  one decode/NMS path and are compared on raw outputs.
- **Inverse.** Boxes map back by the per-axis ratio actually applied
  (`W / floor(W r)`, `H / floor(H r)`), not by `1 / r` as upstream does. This is the exact
  inverse of OpenCV's edge-linear resize into contract half-open coordinates. Boxes are
  clipped to the frame; boxes left empty are dropped.
- **Runtime.** ONNX Runtime 1.30.0, CPU execution provider only, with the provider list
  checked at load. ONNX opset 17 for the AERIS export. Upstream's opset-11 `yolox_s.onnx`
  is kept as a cross-check only.
- **Parity tolerances.** Declared before measurement in `configs/detector/baseline.json`.
- **PyTorch reference.** PyTorch comes from PyPI (torch 2.14.0, torchvision 0.29.0), hash
  locked and installed without triton. The owner decided this on 2026-09-26, because
  `download.pytorch.org` is denied here, and raised the budget to 5 GB.
  - YOLOX is built from the pinned commit. Its commit and installed-file hashes are
    checked at run time.
  - The AERIS export uses the TorchScript-based exporter, as upstream does, and is
    byte-reproducible in the same environment.
  - It passes parity on the fixed 13-frame sample (10 photos, 8 of them with aircraft,
    one resized crop of a photo and two synthetic frames). The largest differences were 1.75e-4 in raw regression,
    1.18e-5 in probabilities and 5.58e-4 px in final boxes, and all 277 detections
    matched.

Alternatives rejected:

- Ultralytics YOLO: AGPL-3.0 unless an enterprise licence is bought.
- torchvision detectors: weights are served from a host this environment blocks, and
  their terms follow COCO's.
- YOLOX-Nano/Tiny as the default: their 416 px input shrinks small aircraft further. They
  stay the fallback if CPU latency, still unmeasured, is too high.
- Dividing by `r` when inverting (upstream): off by up to one resized pixel on
  non-integer resizes.
- Using the airplane probability whatever the argmax: that departs from the upstream
  postprocessing whose COCO numbers are published. AERIS-007 may revisit it with measured
  data.

Consequences:

- The PyPI torch wheel carries about 2.3 GB of CUDA libraries that CPU work does not use.
  A later environment with `download.pytorch.org` can switch to the CPU wheel. It must
  then re-run parity and record the new export hash.
- Real aircraft photos come from Wikimedia Commons, licence-checked through its API
  (public domain, CC BY, CC BY-SA). On every required aircraft photo the pretrained
  airplane class maps to AERIS 0 records on the aircraft. Small aircraft (below about
  32 px in the 640 px input) are missed; later tasks address that.
- The TorchScript exporter is deprecated upstream in PyTorch. Moving to the
  `torch.export` exporter is a later, separate change.
- Upstream reports COCO numbers only. AERIS-007 measures aircraft performance and selects
  the operating threshold.
