# AERIS

Monocular aircraft detection, tracking and image-space trajectory estimation.

**Status: engineering foundation and agent orchestration kit.** The perception
pipeline is planned, not implemented. No aircraft accuracy or runtime results are
claimed. Start with [the operating guide](docs/START_HERE.md).

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
python3 tools/orchestrate.py prepare AERIS-001
```

Requires Python >=3.11; C++ foundation requires CMake >=3.20 and C++20. Use `rtk`
as a prefix if available. See the guide for full environment/build/review commands.

```text
python/aeris/          Python reference package; task-owned modules added incrementally
cpp/                  C++20 foundation; complete runtime is a later task
configs/              Data/model/tracker/camera/evaluation configurations
tests/                Workflow tests now; behavioral/integration fixtures as developed
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

No project distribution license has been selected. Dataset, model and dependency
terms must be recorded before adoption; no third-party footage or weights are bundled.
