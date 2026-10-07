# PLAN-005 - AERIS-007 retry scope after TOOL-001

Status: mechanical, needs the AERIS-007 reviewer's confirmation. Date: 2026-10-07.

TOOL-001 (accepted, `a29e9c2`) lets a retry baseline absorb clean, out-of-scope files changed by
external commits. A dry run of the rule for AERIS-007 (`prepare --retry --base-commit ab24d0e`, the
HEAD when attempt 01 was prepared) rebased nothing, for two reasons that follow from its design:

1. `f0bf670` (PLAN-003) amended AERIS-007's own entry in `orchestration/tasks.json`, which is in the
   task's `write_globs`; the commit touching a task file is "tainted", and the task's own contract
   change is never overlaid.
2. The accepted TOOL-001 changes (`tools/orchestrate.py`, `tests/test_orchestration.py`) are on the
   protected-path denylist, so they are never overlaid either; the commits that document them
   (`docs/EVALUATION_LOOP.md`, the TOOL-001 ADR, PLAN-004) arrive together with `tasks.json`.

AERIS-007's `write_globs` therefore also list those accepted files (this plan, PLAN-004,
`docs/decisions/TOOL-001-*.md`, `docs/EVALUATION_LOOP.md`, `tools/orchestrate.py`,
`tests/test_orchestration.py`), in addition to the PLAN-003 planning files. AERIS-007 does not touch
them: the reviewer should confirm with `git diff ab24d0e..HEAD` that their only changes are the
accepted TOOL-001 commit `a29e9c2`, PLAN-004/005, the MAINT-002 task entry and the PLAN-003 changes.

This is the fourth scope widening and shows the remaining gap in TOOL-001: accepted changes to protected
paths or to a task's own contract cannot be overlaid. A follow-up tooling task should add an explicit,
audited coordinator override for exactly those cases (recorded in PROJECT_STATE). The stray untracked
file `:memory:.ses` left by another session was moved out of the repository (to the scratchpad) because
it would otherwise fail the scope check; nothing in it was read beyond its 51-byte size and type.
