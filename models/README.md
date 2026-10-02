# Model storage

checkpoints/, onnx/, and tensorrt/ hold external/generated model artifacts and are
ignored by Git. Record versions, terms, weight hashes and retrieval instructions in
configs/detector/.

Baseline weights (AERIS-006), not in Git:

```sh
mkdir -p models/onnx
curl -L -o models/onnx/yolox_tiny.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
shasum -a 256 models/onnx/yolox_tiny.onnx   # must equal configs/detector/baseline.json
```

Download is 20,219,662 bytes. The loader refuses a file whose SHA-256 differs.
No model has been trained or fine-tuned.
