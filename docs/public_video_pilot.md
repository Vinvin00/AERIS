# QUT S5 real-video pilot (AERIS-031)

This is one descriptive replay of a published, stationary ground-camera aircraft
sequence. It is real video, but it is not a representative aircraft benchmark or
a live physical-bearing validation. The result is pinned in
`benchmarks/qut_video_pilot.json`.

## Reproduce

The publisher [describes the dataset and BSD-3-Clause terms](https://open.qcr.ai/dataset/aircraft-collision-course/).
The source release is pinned to Hugging Face revision
`0bb9ece0c70d56a56f0334dfdd867b0d7fce15f9`. From the repository root:

```sh
mkdir -p data/raw/qut
curl -fL 'https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/resolve/0bb9ece0c70d56a56f0334dfdd867b0d7fce15f9/Videos/S5.mp4' -o data/raw/qut/S5.mp4
curl -fL 'https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/resolve/0bb9ece0c70d56a56f0334dfdd867b0d7fce15f9/Target%20Labels/target-aircarft-labels/S5_labels.csv' -o data/raw/qut/S5_labels.csv
shasum -a 256 data/raw/qut/S5.mp4 data/raw/qut/S5_labels.csv
.venv/bin/python scripts/qut_video_pilot.py
```

The expected SHA-256 values are recorded in the manifest. The script rejects a
different hash or a total source size above 2 MB. The source files stay under
Git-ignored `data/raw/qut/`; only the small result manifest is committed. The
existing local YOLOX-Tiny ONNX model and Python environment are required; see
`docs/detection.md` for its provenance and weight-terms limitation.

## Alignment and measurement

The decoded S5 video has 1,677 frames at 1,280 × 720, with source PTS from
0.00 to 67.04 s at 25 fps. The publisher's
[target-info.csv](https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/blob/main/target-info.csv)
lists S5 global frames 57,750–59,426, which map to video
indices 0–1,676. The label CSV begins at global frame 58,297, giving first
video index 547, and ends at index 1,676. Its 1,130 rows contain approximate
target centroids (`u`, `v`) in original-frame pixels and a published `range`
column. The range unit is unverified and the column is not used. The loader
rejects a frame-count or coordinate mismatch instead of guessing an offset.

The runtime reads video only. It runs the accepted detector, tracker and two
causal forecasters before the evaluator opens labels. The evaluator pairs a
target label to an output only when exactly one center is within a fixed 50 px
gate. Forecasts are compared to future target labels using source timestamps;
interpolation is allowed only between adjacent labeled frames. These errors
are **conditional on unique association**, so their denominators and unmatched
frames matter. The gate and score settings are untuned and were not selected
from this sequence.

The manifest records actual detection, tracking, forecast and match counts, and
pixel errors by forecast model and horizon. It reports AP, HOTA/IDF1, ENU
bearing, predicted range, live accuracy and confidence intervals as unavailable
with reasons. Approximate single-target centroids do not provide exhaustive
boxes, complete negative frames, reliable all-aircraft identities, camera
intrinsics or surveyed pose. Real-video detection and forecast error here do not
establish accuracy on another camera or scene.
