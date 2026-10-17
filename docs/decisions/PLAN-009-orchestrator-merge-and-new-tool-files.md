# PLAN-009 - A reviewed tooling task so retries survive merge commits and new tool files

Status: proposed, awaiting independent review. Date: 2026-10-16. Owner decision: on 2026-10-16 the
project owner chose to fix the orchestrator (option A) when AERIS-012's third attempt could not be
prepared, and chose the narrow "new `tools/` files only" denylist rule.

## Problem

AERIS-012 has two closed attempts and one left. Two TOOL-001 rules now block it:

1. `prepare --retry` calls `rebase_candidates`, which refuses any merge commit in
   `base_commit..HEAD` (TOOL-001 AC1). `7eb4161` ("Merge branch 'main' into claude/review-fixes") sits in
   that range, so the call raises `Merge or root commit 7eb4161ecd in the retry range`. This is
   reproducible by calling `rebase_candidates` with attempt 01's `base.json` (read-only).
2. `tools/` is on `REBASE_DENYLIST`. `tools/audit_publication.py` (MAINT-003, commit `7eb6a50`) was added by
   a trusted out-of-scope commit but can never be overlaid, so `verify` reports it as an out-of-scope
   change (attempt 02's escalation). The denylist exists so a weakened verifier cannot be committed
   and retried past; it was not meant to block unrelated new tools.

Any task retried after a branch merge or a new tool hits the same walls.

## Proposal

New task `TOOL-002` changes `tools/orchestrate.py`, its tests and `docs/EVALUATION_LOOP.md`:

- A two-parent merge commit in the range no longer fails closed. Its changed files are those that differ
  from its first parent. The commits it brings in from the second parent are already listed by
  `rev-list base..HEAD` and are classified one by one as before. A merge that touches a `write_globs`
  file taints every file it touches, exactly like an ordinary mixed commit. Root commits and merges
  with more than two parents still fail closed, as does any Git error.
- `tools/` stays denied, with one exception: a file under `tools/` is eligible only if it did not exist at
  `base_commit`. Existing gate code (`tools/orchestrate.py`, `tools/check_tests.py`) is never eligible, even
  if it was deleted and re-created. All other TOOL-001 conditions still apply: outside `write_globs`,
  present in HEAD with working-tree bytes equal to HEAD, touched only by untainted commits, the 50-file
  cap, and the audit record in `base.json`, `prepare` output, `implementer.md` and `reviewer.md`.
- Every other denylist entry, the gates and the no-Git behaviour are unchanged. Existing tests pass
  unmodified except the one asserting that a merge commit is refused, which is replaced by tests of the new rule.

## Revision after independent review (2026-10-16)

A fresh reviewer approved with changes. Applied, and owner-confirmed for the `tools/` rule:

- **Shadowing (high).** `lint` and `check_tests.py` run as scripts, so `tools/` is on `sys.path` and a new
  `tools/json.py` would replace the stdlib module inside the gates. The automatic "absent at base" exception is
  therefore replaced by an opt-in: `prepare --retry --allow-new-tool <path>` names each path. Only a direct
  `tools/<name>.py` regular file is eligible, absent from the base tree and the original inventory (casefold + NFC,
  for case-insensitive APFS), with a stem that is not a stdlib or installed module and not
  `sitecustomize`/`usercustomize`/`conftest`/`__init__`/`__main__`. The audit record gains size and first lines.
- **Merge diff (high).** `diff-tree` without `-m` returns nothing for a merge, which would silently defeat tainting.
  The merge file set is fixed to `diff-tree -m --first-parent`, and the merge hash is added to `commits`. Merging a
  feature into main taints the feature's files conservatively (accepted false negative, documented).
- **Spec hygiene.** TOOL-002 now states that it amends TOOL-001 AC1/AC2, enumerates the Git failures and merge cases,
  and pins the AERIS-012 replay (read-only `rebase_candidates`, stored output) instead of "the real repository".
- `REBASE_NEVER` is kept as a defensive guard but is redundant with the base checks; a mutation test must show
  that each new guard is caught.

## Risks and why they are acceptable

A merge can carry a conflict resolution that is not in any single commit. The byte-equal-to-HEAD check
and the audit list still apply, and a merge touching task-scope files taints its whole file set. A new
`tools/` file is part of the baseline, not a gate, because it cannot alter the existing verifier files;
but a check script could import it, so the reviewer must read each overlaid `tools/` path in the audit
list. As with TOOL-001, the rule cannot tell an implementer's own out-of-scope commit from another
session's, and the independent reviewer is the defence. TOOL-002 is verified by the old tooling and
needs a fresh non-author reviewer who tries the abuses (merge carrying a task-scope edit, merge with
three parents, a re-created `tools/orchestrate.py`, a new `tools/` file that shadows an import).

## Consequences

- AERIS-012's attempt 03 can be prepared with `--retry` once TOOL-002 is accepted. Its `cli.py`,
  `pyproject.toml` and `docs/mvp.md` were touched by other tasks since the baseline; those commits touch
  AERIS-012 `write_globs` files, so the preserved work must be reconciled by the implementer, not rebased.
- TOOL-001's ADR and AC1 are amended by TOOL-002's ADR; its historical record stays as written.

## Addendum 2026-10-17: packet size (owner decision)

TOOL-002 attempt 01 was verified, then revised by its independent reviewer (AC3 mutation coverage; wording errors
about `git diff-tree --first-parent`). The retry packet came to about 170 KB against the 160 KB cap in `prepare`,
mainly because every `docs/decisions/*.md` (35 files, about 77 KB) is added to every packet. The owner chose to fix
the cause inside TOOL-002 rather than widen its scope or drop its `read` list (editing the task's own entry would
make `orchestration/tasks.json` an out-of-scope change on retry).

Rule to implement in `prepare` (and the reviewer packet if it shares the code): include a decision record in the
packet only if it is named in the task's `read` list, matches the task's own `write_globs` (e.g. its own ADR),
belongs to a direct dependency (filename starts with the dependency id), or is `0001-scope.md`. All others are
listed by name in a short "Decision records not included" section, never silently dropped. The 160 KB check still
fails closed. This does not change what a task may do, only what the packet pre-loads; the implementer already
reads the repository. Reviewer must confirm the omitted-ADR list is printed and that all existing packet tests
pass except any that asserted blanket inclusion, which are updated and named in the report.
