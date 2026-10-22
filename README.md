# AERIS

Monocular aircraft detection, tracking and image-space trajectory estimation.

<a href="https://github.com/Vinvin00/AERIS/releases/download/demo-video-v1/aeris-demo.mp4"><img src="https://github.com/Vinvin00/AERIS/releases/download/demo-video-v1/aeris-cover.png" alt="AERIS demo: four jets in formation with detection brackets" width="100%"></a>

*Demo video (21 s, click the image): AERIS output on public footage at 30 fps, showing boxes, tracks
and its 1 s forecasts. Forecast errors in it are scored against later AERIS detections, not labelled
ground truth. Footage: Lucastphotography (CC BY 4.0), Jo Brodie (CC0), Designism (CC0), via Wikimedia
Commons. Rebuild with [`video/`](video/README.md).*

**Status: research components, no real-video evidence yet.** Implemented and accepted
through the project's review process: canonical data records and dataset QA, analytic
ground-truth fixtures, metric adapters (COCO AP, TrackEval HOTA/IDF1), a pretrained YOLOX-Tiny
ONNX detector baseline, timestamp-aware video decoding, Kalman tracking, causal image-space
forecasts, calibrated camera-relative geometry, and a baseline manifest with a threshold
protocol. An integrated recorded-video CLI (`python -m aeris`, see [the MVP guide](docs/mvp.md))
is written and tested but **not accepted**: its real-clip criterion is open. No owned
real aircraft video exists, so no detection accuracy, tracking performance or forecast
error has been measured. Start with [the operating guide](docs/START_HERE.md).

The [ONNX smoke record](benchmarks/onnx_smoke.json) records real CPU inference on
a six-frame clip derived from one aircraft photograph. This demonstrates model
execution and overlay generation; it is not a real-video benchmark. Reproduce it
with [these instructions](docs/publication.md).

## AI contributions

Development used AI agents (Claude and Codex) for substantial implementation,
documentation and review, coordinated through the task registry in
`orchestration/tasks.json`. Commits keep their `Co-authored-by` trailers; see
[the contribution record](docs/publication.md). The repository owner is
accountable for the project; AI output is not evidence of human validation.

## Repository safety

[The publication audit](docs/publication_safety.md) covers history, metadata and tracked
files. Rerun it with `python3 tools/audit_publication.py`.

- [Feasibility review and corrections](docs/FEASIBILITY.md)
- [Revised roadmap](docs/ROADMAP.md)
- [Shared data, geometry and evaluation contracts](docs/CONTRACTS.md)
- [Agent evaluation and feedback loop](docs/EVALUATION_LOOP.md)
- [Task backlog and dependencies](orchestration/tasks.json)
- [Coordinator prompt](orchestration/prompts/coordinator.md)
- [Current project state](docs/PROJECT_STATE.md)

```sh
python3 tools/check_tests.py
python3 tools/orchestrate.py lint
python3 tools/orchestrate.py status
```

Requires Python >=3.11; C++ foundation requires CMake >=3.20 and C++20. Use `rtk`
as a prefix if available. See the guide for full environment/build/review commands.

```text
python/aeris/          Python reference package; task-owned modules added incrementally
cpp/                  C++20 foundation; complete runtime is a later task
configs/              Data/model/tracker/camera/evaluation configurations
tests/                Workflow, component and synthetic behavioral tests; local model smoke tests
tools/                Offline orchestration and nonempty-test gates
orchestration/        Task DAG, prompts, templates; ignored local run evidence
docs/                 Reviewed plan, contracts, decisions and operating instructions
data/ models/         External data/weights, excluded from Git
benchmarks/           Versioned benchmark manifests; large runs excluded
experiments/          Reproducible experiment specifications and reports
```

The MVP assumes recorded RGB footage and a fixed camera. Pixel motion is observable;
metric range is conditional research. TensorRT/Jetson is an optional NVIDIA extension.
See the feasibility review for limitations and primary sources.

**Licensing is unresolved.** No project distribution license has been selected, so the code
is all rights reserved by default. YOLOX-Tiny weights (code Apache-2.0, weight terms unverified)
and the FGVC sample carry their own terms and are not redistributed; no third-party footage
or weights are bundled. Record dataset, model and dependency terms before adoption.
