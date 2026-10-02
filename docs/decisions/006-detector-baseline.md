# ADR 006 — YOLOX-Tiny COCO baseline run through ONNX Runtime CPU

Status: proposed for AERIS-006 review. Date: 2026-10-02.

Baseline: the official YOLOX-Tiny COCO ONNX release (416x416, opset 11, 20.2 MB).
Code is Apache-2.0 per repository metadata. The release does not state weight terms
separately; we assume the repository license, which is an unverified legal position,
and the weights were pretrained on COCO. Ultralytics models were not used because of
their AGPL/enterprise terms. No project license has been chosen.

The exported graph returns raw head output, so grid/stride decoding, class-aware NMS,
letterbox inversion and class mapping are implemented in `python/aeris/detection/
baseline.py` and shared by all runtimes. Only COCO `airplane` maps to
`aeris-aircraft-v1` class 0.

Alternatives not taken: RT-DETR (larger, transformer) and torchvision SSDlite (weight
terms and PyTorch footprint). PyTorch was not installed, so the "Python reference" is
OpenCV DNN executing the same ONNX graph beside ONNX Runtime. That checks runtime and
postprocessing parity, not PyTorch-to-ONNX export fidelity, which was done upstream.

Consequences: no detection quality claim exists. Three FGVC images are a smoke check.
COCO pretraining may overlap any COCO-derived evaluation data. The fixed score floor
is not an operating threshold; that must be selected on validation data.
