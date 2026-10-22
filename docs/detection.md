# Baseline detector (AERIS-006)

`python/aeris/detection/baseline.py` and `configs/detector/baseline.json` define a
pretrained YOLOX-Tiny COCO baseline. **No training, tuning or quality benchmark has
been done.** Three bounded FGVC images are a smoke check only.

## Provenance and terms

| Item | Value |
|---|---|
| Weights | `yolox_tiny.onnx`, 20,219,662 bytes, SHA-256 `427cc366…70b0f7` (full hash in config) |
| Source | `github.com/Megvii-BaseDetection/YOLOX` release `0.1.1rc0`, retrieved 2026-10-02 |
| Code license | Apache-2.0 (repository license metadata) |
| Weight license | Not separately stated; assumed to follow the repository. **Unverified** — confirm before any release |
| Training data | MS COCO 2017 train; overlap risk with COCO-derived evaluation data |
| Export | PyTorch 1.7, opset 11, done upstream (not by AERIS) |

Weights stay out of Git; `models/README.md` has retrieval steps. The loader rejects a
file whose SHA-256 differs from the config.

## Preprocessing, decoding and mapping

Input is BGR float32 0–255 (no mean/std), aspect-preserving `INTER_LINEAR` resize to
fit 416x416 and placed top-left on 114-valued padding. The graph's raw output
`(1, 3549, 85)` is decoded per stride 8/16/32 (`(xy+grid)*stride`, `exp(wh)*stride`).
Score is objectness x class score. Postprocessing keeps only COCO `airplane`
(index 4) → `aeris-aircraft-v1` class 0, applies score floor 0.1, class-aware greedy
NMS at IoU 0.45, at most 100 detections, divides by the letterbox ratio to return to
original-frame pixels, clips to the frame, and drops boxes that clip to zero area.
All other COCO classes are dropped. Helicopters, drones and birds have no mapped class.
The 0.1 floor is for AP; the operating threshold must be selected on validation data.

## Runtime agreement (AC3)

ONNX Runtime (`CPUExecutionProvider` only, provider list checked) and OpenCV DNN CPU
run the same graph on three fixed images. Declared tolerances: raw output 1e-3,
box 0.5 px, score 1e-3. These were set after a first probe showed ~7e-5 raw
differences (about 14x looser) and before the recorded report. Recorded result in
`tests/fixtures/detection/runtime_agreement.json`: raw max |Δ| 7.4e-5, box max Δ
2.4e-4 px, score max Δ 1.8e-7, identical detection counts, all within tolerance.
Model-forward p50 on this Mac: ONNX Runtime about 9.6 ms, OpenCV DNN about 20.8 ms
(one image, 20 runs, one warmup; not a benchmark).

**Limits:** this compares two runtimes on one exported graph. It does not prove that
a PyTorch checkpoint exports faithfully to ONNX, since PyTorch was not installed.
OpenCV 5 printed a warning that its new graph engine ignores `setPreferableTarget`.

## Smoke results (3 images, not accuracy)

`1143132` and `1627560`: one airplane detection each, class 0; `1143132` has IoU > 0.85
with the FGVC label box. `1885942` (cabin interior): no airplane detection. FGVC's
20 px bottom banner is not cropped by this detector (the dataset records it as a model-
input crop for later evaluation work).

## Reproduce

```sh
.venv/bin/python -m pip install onnxruntime==1.30.0 onnx==1.23.1
PYTHONPATH=python .venv/bin/python -m aeris.detection.baseline IMG1.jpg IMG2.jpg --output report.json
.venv/bin/python tools/check_tests.py tests/unit/test_detection.py
```

Real-model tests skip when the weights or FGVC sample are absent (for example in CI).

## Tracker-guided crops (AERIS-034, opt-in)

`python/aeris/detection/guided.py` runs the same model on up to four model-input-sized windows cut at
native resolution around confirmed or lost tracks, centred on each track's last state propagated to the
current time. Edge fragments are dropped, crop boxes are merged by NMS, and boxes overlapping a
frame-pass box (IoU ≥ 0.5) are dropped; the remainder reach the tracker as extend-only measurements.
Enable it with `--guided-crops` in the CLI or web server; it is off by default. On the QUT development
split it did not pass the PLAN-011 rule (lower crop-inclusive detection recall, more off-target
observations); see [ADR 034](decisions/034-tracker-guided-crop.md). Not a quality claim.
