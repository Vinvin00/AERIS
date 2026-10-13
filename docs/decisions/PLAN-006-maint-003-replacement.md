# PLAN-006 - MAINT-003 replaces MAINT-002

Status: mechanical, coordinator decision requested by the owner. Date: 2026-10-13.

MAINT-002 attempt 01 was prepared on 2026-10-07 against an uncommitted `orchestration/tasks.json`
(the MAINT-002 entry itself was added by another session and committed later as `b210258`).
`prepare MAINT-002 --retry` therefore cannot find a base commit: no commit in the last 30 has a
`tasks.json` matching attempt 01's recorded hash, and `--base-commit f0bf670` was refused with
"does not match the recorded baseline". Attempt 01 was paused with no source changes.

Rather than edit run state or weaken the retry check, MAINT-003 is a copy of MAINT-002 (same
objective, write scope, acceptance criteria, checks and budget) with a fresh attempt budget, and
MAINT-002 is set to `deferred` so it cannot be prepared. No scope or criterion changed.

Reviewer: confirm with `git diff HEAD -- orchestration/tasks.json` that the only changes are the
MAINT-003 entry and MAINT-002 `status: deferred`.
