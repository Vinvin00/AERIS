# Dependency locks

ML/CV tasks must select compatible Python versions and add reproducible
platform-specific dependency locks here. pyproject.toml ranges are not a claim of a
frozen ML training environment.

- `dev.txt`: tooling versions validated with the scaffold (not an ML lock).
- `data-macos-arm64-py314.txt`: hash-pinned numpy and headless OpenCV used by the
  AERIS-003 loader, QA and contact sheet on the local Python 3.14.6 `.venv`
  (installed footprint about 153 MB). numpy 2.5.3 itself requires Python >= 3.12.
- `data-linux-x86_64-py311.txt`: the same tooling for Linux x86_64 (glibc >= 2.28) on
  CPython 3.11: numpy 2.4.6 and opencv-python-headless 5.0.0.93. Verified by a
  hash-checked install into a fresh venv followed by the full test suite.

- `eval-linux-x86_64-py311.txt`: hash-pinned pycocotools 2.0.11 and scipy 1.17.1 for the
  AERIS-005 evaluators, plus the install command for official TrackEval at commit
  `12c8791b`. That is a VCS pin, whose integrity check is the commit SHA; it is verified at
  run time. Verified by a fresh-venv install of both Linux locks and the full test suite.

CI on Python 3.11/3.12 still resolves the ranges in pyproject.toml rather than a lock.
The `dev` extra includes the evaluators so CI installs them.
