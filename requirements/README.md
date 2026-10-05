# Dependency locks

ML/CV tasks must select compatible Python versions and add reproducible
platform-specific dependency locks here. pyproject.toml ranges are not a claim of a
frozen ML training environment.

- `dev.txt`: tooling versions validated with the scaffold (not an ML lock).
- `data-macos-arm64-py314.txt`: hash-pinned numpy and headless OpenCV used by the
  AERIS-003 loader, QA and contact sheet on the local Python 3.14.6 `.venv`
  (installed footprint about 153 MB). Linux CI on 3.11/3.12 resolves the ranges in
  pyproject.toml. numpy 2.5.3 itself requires Python >= 3.12.
- `evaluation-macos-arm64-py314.txt`: exact standard evaluator and required numeric
  versions exercised by AERIS-005. The local task installed these with `--no-deps`
  on the existing data environment; TrackEval's plotting/dataset extras were not
  needed by the HOTA and Identity API calls. This is a version pin, not a hash lock.
- `detection-macos-arm64-py314.txt`: ONNX Runtime 1.30.0 (CPU) and graph-inspection
  `onnx` used by the AERIS-006 baseline. Version pin only; Linux and Python 3.11/3.12 untested.
- AERIS-009 tracker needs only `scipy==1.17.1` (extra `tracking`), already pinned by
  `evaluation-macos-arm64-py314.txt`. Linux and Python 3.11/3.12 untested.
