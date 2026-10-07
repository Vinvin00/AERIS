# PLAN-004 - A reviewed tooling task to end the retry write-scope workaround

Status: reviewed, approved with changes (applied). Date: 2026-10-07. Owner decision: on 2026-10-07 the
project owner chose option 2 (fix the orchestrator first) when AERIS-007's retry collided with
already-committed changes for the third time, and confirmed the other session was finished.

## Problem

`prepare --retry` keeps the *first* attempt's file snapshot as the ownership baseline, so that
cumulative work across attempts stays visible. But any change committed by other, already-accepted
work between attempt 01 and the retry (planning changes, maintenance tasks) then looks like the
retrying task's own change and fails the write-scope check. PLAN-001 (AERIS-009), PLAN-002
(AERIS-010) and PLAN-003 (AERIS-007) each worked around it by widening the retrying task's
`write_globs` to include those files, including `orchestration/tasks.json`. That is a repeated,
growing weakening of the scope guard.

## Proposal

New task `TOOL-001` changes `tools/orchestrate.py`:

- The first `prepare` of a task records `base_commit` (`git rev-parse HEAD`) and whether the tree was dirty.
- On retry the baseline is the first attempt's baseline overlaid with the current content of files that
  satisfy all of: outside the task's `write_globs`; not on a protected-path denylist
  (`orchestration/gates.json`, `tools/`, `tests/test_orchestration.py`, `orchestration/prompts/`,
  `.github/`, `AGENTS.md`, `docs/CONTRACTS.md`), so a weakened verifier cannot be committed and
  retried past; tracked, present in HEAD, with working-tree bytes equal to `git show HEAD:path`
  (compared by the tool, not trusted from `git status`); and changed solely by commits in
  `base_commit..HEAD` none of which touches a `write_globs` file (such a commit is "tainted" and all
  files it touches stay at the original baseline, even if later reverted).
- `orchestration/tasks.json` is eligible only if the retrying task's own entry is unchanged since
  `base_commit`. At most 50 files may be overlaid. Deleted files are dropped from the baseline, new
  files added.
- Fail closed on any Git error, a `base_commit` that is not an ancestor of HEAD (amend, rebase or
  reset), any merge commit in the range, or a missing `base_commit`. Only a directory with no `.git`
  keeps the old behaviour.
- Legacy attempts without `base_commit` (AERIS-007's attempt 01) need an explicit
  `--base-commit`, which must be an ancestor of HEAD and whose tracked files must match the recorded
  base hashes; it is recorded and printed.
- Each overlaid file is recorded in `base.json` as `{path, sha256, commits}` and printed by
  `prepare` and listed in `implementer.md` and `reviewer.md`. In-scope changes (committed or not),
  gates and the no-Git behaviour are unchanged; `base[files]` stays the overlaid map and
  `source-before` is unchanged. Existing tests pass unmodified; new tests use disposable Git
  repositories and the abuse tests must be shown to fail against the pre-change tool.

## Risks and why they are acceptable

The rule cannot tell an implementer's own out-of-scope commit from another session's; an implementer
who commits such an edit mid-task and retries gets it overlaid. The defences are the clean-tree and
no-tainted-commit conditions, the denylist (so verifiers and contracts cannot be moved this way), the
cap, and the printed audit list that the independent reviewer must check; the list is advisory and
relies on that reviewer. Dirty state at the first baseline stays in the delta. Reflog or hook
tampering is out of scope. TOOL-001 is itself verified by the current (old) tooling, so it needs a
fresh non-author reviewer who tries the abuses; the self-referential caveat is accepted.

## Consequences and prerequisites

- AERIS-007's retry can use its original scope once accepted; the reviewed edit then removes
  `orchestration/tasks.json`, `docs/ROADMAP.md`, `docs/dataset.md`, `docs/PROJECT_STATE.md` and
  `PLAN-003-*` from its `write_globs`. Its attempt 01 is legacy: the retry passes `--base-commit
  ab24d0e` (the HEAD when it was prepared), checked against the recorded hashes.
- The working tree must be clean for rebased files: the uncommitted MAINT-002 task entry in
  `orchestration/tasks.json` has to be committed or removed by its owner before a retry depends on
  that file.
- PLAN-001..003's widening of the closed AERIS-009 and AERIS-010 stays as the historical record;
  those tasks are unaffected.
