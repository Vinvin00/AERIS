# PLAN-002 - AERIS-010 write scope after MAINT-001

Status: mechanical planning change, needs the AERIS-010 reviewer's confirmation.
Date: 2026-10-06.

AERIS-010 attempt 01 was prepared on 2026-10-05 and closed with `abandon` at 2026-10-06
10:37 by a separate owner-directed session (reason recorded in its `feedback.md`: work
redirected to CI, documentation, provenance and smoke evidence; no forecast changes existed).
That session's maintenance task MAINT-001 was accepted and committed as `cb53b07`, touching
`.github/workflows/ci.yml`, `README.md`, `docs/publication.md`, `scripts/smoke_onnx.py`,
`benchmarks/onnx_smoke.json` and `orchestration/tasks.json`.

The orchestrator keeps attempt 01's file snapshot as the ownership baseline for every retry,
so those already-accepted files would appear in AERIS-010's change list and fail the scope
check. As for AERIS-009 (PLAN-001), AERIS-010's `write_globs` now also list exactly those
files. Authority is widened only for the duration of AERIS-010; the reviewer is asked to
confirm that the only differences in those files since the baseline are the MAINT-001 changes
and this plan's `tasks.json` edit, and that AERIS-010 itself did not touch them.
Attempt 02 carries the forecast implementation, which was written after the closure.
