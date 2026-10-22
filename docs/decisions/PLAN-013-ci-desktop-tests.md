# PLAN-013 — CI installs what the required Qt tests need (TOOL-004)

Date: 2026-10-22. Status: r2, approved for preparation. Review r1
(`orchestration/runs/coordination/PLAN-013-review-r1.md`) returned APPROVE WITH CHANGES; r2 applies
[S1]..[S4].

## Problem

After TOOL-003 made `ruff check .` clean, CI run 39853854581 on `deploy-hosted` (`d8d7008`)
reached the unittest step for the first time in that branch's recent history and failed on two
tests in `tests/ui/test_ui.py` (`RequiredRunnerUITests.test_unlabeled_result_never_claims_accuracy`,
`test_dark_landing_and_compact_workspace_reach_controls`) with `ModuleNotFoundError: No module named
'PySide6'`. Python 3.12 failed the same way (`errors=2, skipped=9`; corrected after TOOL-004 review F3). The class docstring calls these
"checks discovered by the repository's required unittest gate"; locally they pass because the
development venv has the `desktop` extra (`PySide6-Essentials==6.12.0`). The workflow installs only
`.[dev,evaluation]`. The tests already run Qt headless (`QT_QPA_PLATFORM=offscreen`).

## Decision

Add TOOL-004: the `python` job in `.github/workflows/ci.yml` installs the `desktop` extra and the
Ubuntu system libraries the Qt offscreen platform loads: after `sudo apt-get update`, at least
`libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`. Any further package must be justified in
`ci.txt` by the CI log line showing the missing library [S1]. No test is skipped, removed or changed; no other job or step changes;
`orchestration/gates.json` is untouched. Making the tests skip without PySide6 was rejected: it
would quietly weaken checks the repository marks as required.

## Evidence

`.github/` is on the orchestrator's `REBASE_DENYLIST` (`tools/orchestrate.py:24`, paths never
overlaid into retry baselines) [S4], and a workflow change can only be proven on GitHub. Before acceptance, the implementer pushes `tracking-improvements` to `origin` (a branch
of its own; CI runs on every push) and records the run ID, commit and per-job conclusions in the
attempt folder (`ci.txt`). Acceptance needs both Python jobs and the cpp job green on that commit, and `ci.txt` quotes, per
Python job, the two Qt test lines ending in `... ok` and the unittest summary line with `skipped=`
no higher than the baseline 9 of run 39853854581 (391 tests, `errors=2, skipped=9`) [S2].

[S3] Pushing a new remote branch needs explicit approval from the owner; the implementer asks before
pushing. The commit verified by CI must be the same SHA later fast-forwarded to `deploy-hosted`;
if it is not, the `deploy-hosted` run is the acceptance evidence instead.
