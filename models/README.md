# Model storage

checkpoints/, onnx/, and tensorrt/ hold external/generated model artifacts and are
ignored by Git. Record versions, terms, weight hashes and retrieval instructions in
configs/detector/. No model has been trained.

## AERIS-006 pretrained baseline (configs/detector/baseline.json)

| File | Source | Bytes | SHA-256 |
|---|---|---|---|
| `checkpoints/yolox_s.pth` | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth | 72,089,125 | `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` |
| `onnx/yolox_s_upstream_0.1.1rc0.onnx` (cross-check only) | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx | 35,858,002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

Retrieve and check them:

    curl -fL -o models/checkpoints/yolox_s.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
    curl -fL -o models/onnx/yolox_s_upstream_0.1.1rc0.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
    sha256sum models/checkpoints/yolox_s.pth models/onnx/yolox_s_upstream_0.1.1rc0.onnx

Terms: Apache-2.0, published as release assets of the Apache-2.0 YOLOX repository; the
upstream weights were trained on COCO train2017.

Generated: `onnx/yolox_s_aeris_opset17.onnx` (35,897,112 bytes, SHA-256
`e68784f4593e8fbcae404d97da7c99f06e3970c7c56b17c0f041bfdde6558aef`). It is exported
from the weights above by `python -m aeris.detection.export parity --out DIR` with torch
2.14.0 (requirements/detector-torch-linux-x86_64-py311.txt). A re-export in the same
environment is byte-identical; other torch versions may produce different bytes.
