# Dependency locks

ML/CV tasks must select compatible Python versions and add reproducible
platform-specific dependency locks here. pyproject.toml ranges are not a claim of a
frozen ML training environment.

- `dev.txt`: tooling versions validated with the scaffold (not an ML lock).
- `data-macos-arm64-py314.txt`: hash-pinned numpy and headless OpenCV used by the
  AERIS-003 loader, QA and contact sheet on the local Python 3.14.6 `.venv`
  (installed footprint about 153 MB). Linux CI on 3.11/3.12 resolves the ranges in
  pyproject.toml. numpy 2.5.3 itself requires Python >= 3.12.
