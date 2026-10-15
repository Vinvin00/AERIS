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

Larger detector weights (optional, used by the `balanced`, `accurate` and
`accurate-cascade` profiles in `aeris.detection.profiles`), not in Git:

```sh
curl -L -o models/onnx/yolox_m.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
shasum -a 256 models/onnx/yolox_m.onnx   # must equal configs/detector/yolox-m.json
```

Download is 101,259,744 bytes; input 640 px, same raw-head layout as yolox-tiny. The
weight licence position is the same unverified assumption as for yolox-tiny, and the
two-runtime agreement report has not been run for this model. Profiles are chosen with
`--profile` on the CLI and the selector in the web console; no profile has been validated
against labelled data. The CLI default stays `fast` (the AERIS-006 baseline).
