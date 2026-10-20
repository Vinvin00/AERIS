# TOOL-002 - Retry baselines across merge commits and opted-in new tool files

Status: implemented in attempt 01, awaiting independent review. Date: 2026-10-16. Design: PLAN-009 including its
"Revision after independent review". This record amends TOOL-001 AC1 (merge refusal) and AC2 (`tools/` denylist);
TOOL-001's own ADR stays as written.

## Old and new rules side by side

| Topic | TOOL-001 (old) | TOOL-002 (new) |
|---|---|---|
| Commit in `base_commit..HEAD` with 2 parents | `prepare --retry` refuses ("Merge or root commit") | Allowed. Its files are those differing from its first parent (`git diff-tree <merge>^1 <merge>`); the merge hash is added to `commits` of each such file; if any of them matches `write_globs`, every file of the merge is tainted |
| Root commit, merge with 3+ parents | Refused | Still refused (fail closed) |
| Git error, unresolvable or non-ancestor base | Refused | Still refused |
| Anything under `tools/` | Always denied | Denied unless the owner passes `--allow-new-tool tools/<name>.py` for that exact path (repeatable, retry only) and it passes the checks below |
| Other denylist entries (`orchestration/gates.json`, `tests/test_orchestration.py`, `orchestration/prompts/`, `.github/`, `AGENTS.md`, `docs/CONTRACTS.md`) | Denied | Unchanged, even if absent at base and even with the flag |
| 50-file cap, byte-equal-to-HEAD, outside `write_globs`, untainted commits, audit record | As documented | Unchanged |

Checks on an `--allow-new-tool` path (any failure raises, nothing is prepared): direct child `tools/<name>.py` with a valid Python identifier as stem;
casefolded name not `orchestrate.py`/`check_tests.py`; stem not a stdlib module (`sys.stdlib_module_names`), not
importable as a top-level module from `sys.path` (with `tools/` itself excluded), not `sitecustomize`,
`usercustomize`, `conftest`, `__init__`, `__main__`; absent (casefold + NFC) from the original inventory and the
`base_commit` tree; no same-stem file, package directory or extension module in the base or HEAD `tools/` tree.
Then, per file: regular-file mode in HEAD (symlinks and gitlinks are skipped; a typechanged symlink replaced by a
regular file with the same bytes is also skipped; a symlink in the working tree fails the inventory), clean against HEAD, touched only by untainted commits. `base.json` records
`allowed_new_tools` and, for each overlaid `tools/` path, `size` and `first_lines` (first three lines, each capped at 200 characters); these are
printed by `prepare` (stderr), and shown in `implementer.md` and `reviewer.md`.

Why opt-in: `lint` and `check_tests.py` run as scripts, so `tools/` is on `sys.path`; a new `tools/json.py` would
replace the stdlib module inside the gates. The owner names each file, and the reviewer reads it.

## Implementation note: the merge diff command

PLAN-009's revision named `git diff-tree -m --first-parent`. On this machine (git 2.54.0 Homebrew and Apple Git
2.50.1) that does not fail: `git diff-tree --first-parent` on a merge exits 0 and prints nothing, and
`-m --first-parent` silently ignores the flag and lists the diffs against both parents. The first would defeat
tainting without any error; the second attributes first-parent files to the merge. The tool therefore uses the
explicit substitute `git diff-tree <merge>^1 <merge>` (verified equal to the first-parent file set, with
`--no-renames`). Plain `diff-tree` without either form also prints nothing for a merge.

## Packet size (PLAN-009 addendum 2026-10-17)

`prepare` used to add every `docs/decisions/*.md` (35 files, about 77 KB) to every implementer packet, which pushed
TOOL-002's retry packet to about 170 KB against the 160 KB cap. A decision record is now pre-loaded only if it is
named in the task's `read` list, matches the task's own `write_globs`, belongs to a direct dependency (its filename
starts with the dependency id) or is `0001-scope.md`. All others are listed by name under "Decision records not
included". The 160 KB check still fails closed. `prepare` now builds and size-checks the packet before creating the
attempt folder, so a refusal leaves nothing behind, and a non-retry `prepare` returns a path only if
`implementer.md` exists. The reviewer packet does not include decision records and is unchanged. This changes what
is pre-loaded, not what a task may do.

## Residual risks

- A merge may carry a conflict resolution that is in no single commit. The byte-equal-to-HEAD check, the audit list
  and tainting limit this; the reviewer must read the audit list.
- Feature merged into main taints the feature's files if any of the merge's first-parent diff is task scope (false
  negative for rebasing, i.e. conservative: more work stays in the delta). Main merged into the feature branch is
  rebased normally. Both directions are tested.
- The rule cannot tell an implementer's own out-of-scope commit from another session's, as in TOOL-001.
- An allowed `tools/` helper can still be imported by a check script or contain harmful code; only the name,
  location and base checks are automatic. The audit shows size and first three lines, not a review of the file.
- `importlib` discovery depends on the interpreter and environment running `prepare`: a module installed later or
  elsewhere can still be shadowed. The name checks reduce, not remove, this.
- `REBASE_NEVER` (gate file names) overlaps the base checks because both gate files exist at base. It is still
  protected: a case-variant of a gate name that is absent at base is refused only by this guard (tested).
- NFD-committed non-ASCII names (macOS `core.precomposeunicode`) are neither overlaid nor refused when allowed; the
  NFC/NFD comparison is tested against base entries, not end to end against an NFD commit. A new warning on stderr
  reports any allowed path that matched no overlay candidate.
- `first_lines` holds at most three lines of at most 200 characters (longer lines end in `...[truncated]`), so
  the audit shows the start of a file, never all of it.
- TOOL-002 is verified by the old tooling. A fresh non-author reviewer should try the abuses: merge carrying a
  task-scope edit, octopus merge, re-created gate file, stdlib-named, package-directory, `.pth`, case-variant and
  symlink/gitlink `tools/` entries.

## AERIS-012 reproducer

Before: `Workflow.rebase_candidates` on `orchestration/runs/AERIS-012/attempt-01/base.json` raised "Merge or root
commit 7eb4161ecd in the retry range". After, the command to prepare attempt 3 once TOOL-002 is accepted is:

    python3 tools/orchestrate.py prepare AERIS-012 --retry --allow-new-tool tools/audit_publication.py

The read-only replay (no `prepare`) and its output are stored in
`orchestration/runs/TOOL-002/attempt-01/aeris012-replay.txt`. AERIS-012's `cli.py`, `pyproject.toml` and
`docs/mvp.md` were touched by other tasks' commits that match its `write_globs`, so that preserved work stays in the
delta and the implementer must reconcile it.


## AC4 replay evidence (read-only, `prepare` not called)

Command: `.venv/bin/python orchestration/runs/TOOL-002/attempt-01/aeris012-replay.py` (calls `Workflow.rebase_candidates` with AERIS-012 attempt-01 `base.json`, base `ffda3604300284bbfb5987ae7915d01966f43d41`, allow-list `tools/audit_publication.py`), at HEAD `7871db12730c8bd71a50f04cceb23ab5f7e9e124` (working tree with the uncommitted TOOL-002 changes; raw output in `orchestration/runs/TOOL-002/attempt-01/aeris012-replay-2.txt`, Git-ignored). Result: **no refusal; 21 audit entries**, all `overlay`:

- `.gitignore`
- `README.md`
- `benchmarks/qut_video_pilot.json`
- `docs/PROJECT_STATE.md`
- `docs/camera_model.md`
- `docs/decisions/029-world-referenced-bearings.md`
- `docs/decisions/030-live-bearings.md`
- `docs/decisions/031-public-aircraft-video-pilot.md`
- `docs/decisions/PLAN-006-maint-003-replacement.md`
- `docs/decisions/PLAN-007-live-bearing-dependency.md`
- `docs/decisions/PLAN-008-public-aircraft-video-pilot.md`
- `docs/decisions/PLAN-009-orchestrator-merge-and-new-tool-files.md`
- `docs/decisions/PLAN-010-replay-acceptance-023-028.md`
- `docs/decisions/UI-002-dark-console.md`
- `docs/public_video_pilot.md`
- `docs/publication_safety.md`
- `python/aeris/estimation/world_bearing.py`
- `scripts/qut_video_pilot.py`
- `tests/unit/test_qut_video_pilot.py`
- `tests/unit/test_world_bearing.py`
- `tools/audit_publication.py`

`tools/audit_publication.py`: size 9018, sha256 `9c1556e3419ebb6539e266d257857e03455b66adaa62ae188578c51f18776736`, added by `7eb6a50340`; first lines recorded in the audit entry. The merge `7eb4161ecd` that blocked the old tool no longer refuses. An earlier run at HEAD `228bab3bed` gave 19 entries; the two extra entries are from commits made since.

## Test evidence (attempt 02)

`.venv/bin/python -m unittest tests.test_orchestration`: 78 tests pass. Against the pre-change tool
(`git show HEAD:tools/orchestrate.py`) 31 distinct new tests fail or error (56 failing results counting sub-tests:
6 failures, 50 errors); most are `TypeError` on the new `allow_new_tool` argument or the old merge refusal.

Each guard was disabled on its own and the full suite re-run; every one below was caught: NFC normalisation alone
(casefold kept), original-inventory term, base-tree term, HEAD-tree term, `stem.isidentifier`, the `sys.path`
exclusion of `tools/` (whole filter, the repo-`tools/` term and the checkout-`tools/` term separately), HEAD
regular-file mode (typechange test), unmatched-allow warning, `first_lines` cap, and the packet rules (omitted list,
blanket inclusion, `0001-scope.md`, `write_globs`, dependency prefix, 160 KB cap, folder created before the size
check, missing `implementer.md`). The earlier `len(name) > 3` rule was removed: `.py` has an empty stem, which
`isidentifier` already refuses (tested).
