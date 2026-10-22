# PLAN-012 — Lint-only fixes in the orchestrator files (TOOL-003)

Date: 2026-10-22. Status: r2, approved for preparation. Review r1
(`orchestration/runs/coordination/PLAN-012-review-r1.md`) returned APPROVE WITH CHANGES; r2 applies
[R1]..[R4].

## Problem

CI (`.github/workflows/ci.yml`, `ruff check .`) has failed on `deploy-hosted` since at least
2026-10-22 12:02 UTC on three ruff findings. One was in `python/aeris/web/server.py` and is fixed
directly (`68d285d`). The other two are in files the repository treats as verifier code, which
AGENTS.md says are changed only through a reviewed task:

- `tools/orchestrate.py:282` — `norm = lambda text: ...` (E731, assigned lambda);
- `tests/test_orchestration.py:769` — `import contextlib, io` (E401, multiple imports on one line).

## Decision

Add TOOL-003, a behaviour-preserving lint fix: replace the lambda with an equivalent nested `def`
and split the import into two lines. No other line of either file changes. No gate, contract,
check list, threshold or test expectation changes. Acceptance needs `ruff check .` clean, the full
orchestration test file and the unit check passing, and a reviewer confirming the diff is exactly
those two edits. A diff limited to these two hunks cannot weaken a gate.

## Evidence and review requirements

- [R1] `ruff check .` is not a registered gate check and none is added (that would be a gate change).
  The implementer stores the ruff version, command and output in
  `orchestration/runs/TOOL-003/attempt-NN/ruff.txt`; the reviewer re-runs it.
- [R2] The edited `tools/orchestrate.py` runs its own verification. The reviewer confirms with
  `git diff --unified=0` that each file has exactly one hunk, and re-runs `lint` and
  `tools/check_tests.py` once with the pre-change `orchestrate.py` (from `git show`, in a scratch
  copy) to confirm the same outcome.
- [R3] Baseline at `68d285d`: `tests/test_orchestration.py` runs 78 tests, all passing; the
  change must leave 78 passing.
- [R4] Branches: `68d285d` (server.py) and TOOL-003 land on `tracking-improvements` (worktree
  `AERIS-tracking`), which is then merged into `deploy-hosted`, the branch CI reports on.
