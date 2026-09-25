# Foundation validation

Validated locally on 2026-09-24, Apple arm64 macOS, Python 3.14.6,
Apple Clang 21.0.0, CMake 4.4.2. Development tools are in the local `.venv`;
validated Python tooling versions are recorded in `requirements/dev.txt`.

| Check | Result |
|---|---|
| `python3 tools/check_tests.py` | 26 behavioral tests passed |
| `.venv/bin/python -m pytest -q` | 26 tests passed |
| `.venv/bin/ruff check .` | Passed |
| `python3 tools/orchestrate.py lint` | 20 tasks and 25 check definitions; valid acyclic dependencies |
| `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` | Configured |
| `cmake --build build --parallel 2` | Built |
| `ctest --test-dir build --output-on-failure` | 1 C++ timestamp contract test passed |

The workflow tests exercise successful acceptance and dependency unlocking as well
as failed commands, missing evidence, out-of-scope changes, empty test files,
self-review, stale source/report/reviews, altered logs, low rubric scores,
blocking findings, retry limits, context preservation and competing writers.
These are tests of the workflow mechanism, not an independent assessment of a real
agent's engineering output. Review identities remain a coordinator responsibility.

GitHub Actions configuration is supplied but has not run remotely. Its Python
3.11/3.12 matrix has not been executed locally. The package installed successfully
in the local Python 3.14 environment; that does not establish ML dependency support.

No real aircraft data/model, AP, tracking accuracy, forecast accuracy, or inference
latency was evaluated. No development task was accepted on the strength of scaffold
tests. Git was initialized locally; no remote, commit or publication was created.
