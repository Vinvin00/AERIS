# TOOL-001 - Retry baselines over committed external changes

Status: proposed for TOOL-001 review. Date: 2026-10-07. Plan: PLAN-004.

`prepare --retry` kept the first attempt's file snapshot so cumulative work stayed visible, which
made commits by other accepted work look like the task's own out-of-scope changes. The retry baseline
now overlays the files described in `docs/EVALUATION_LOOP.md` (outside `write_globs`, unprotected,
clean in Git, changed only by commits that touch no `write_globs` file, at most 50, recorded and
audited). Design choices: `base_commit` is recorded at the first `prepare` rather than inferred from
timestamps; every Git failure fails closed; file bytes are compared directly instead of trusting
`git status` (assume-unchanged and skip-worktree cannot hide a dirty file); a commit that touches any
task file taints all the files it touches even if later reverted; verifier and contract paths are
never overlaid; `prepare` creates the attempt directory only after every check passed.

Verification: the new Git tests fail against the pre-change tool (19 of 21; the two that pass assert
behaviour that must not change) and every rule is mutation-tested. One mutation survives by design:
removing the in-scope-glob exclusion in `rebase_candidates`, which is redundant because a commit that
touches an in-scope path is already tainted.

Limits: TOOL-001 is verified by the old tooling. The overlay cannot distinguish an implementer's
out-of-scope commit from another session's; the audit list is the control. Reflog, hooks and history
rewriting that keeps `base_commit` an ancestor are out of scope. Merge commits in the range always stop
a retry.
