# Agent output evaluation and feedback loop

## State machine

`planned -> ready -> prepared -> implemented -> verified -> reviewed -> accepted`

Rejected work creates feedback and closes the attempt. `prepare --retry` creates a
new packet with prior feedback and the original ownership baseline. Three attempts
maximum; after that the coordinator must revise the plan or obtain a missing decision.
The CLI reports dependency readiness; the evidence directory records finer-grained
attempt state. A ready task may already have an open attempt; `prepare` resumes it.

### Retry baselines over committed external changes

The first `prepare` records `base_commit` (HEAD) and whether the tree was dirty. On `--retry` the
baseline is the first attempt's baseline overlaid with files that are (1) outside the task's
`write_globs`, (2) not protected (`orchestration/gates.json`, `tools/` unless opted in below, `tests/test_orchestration.py`,
`orchestration/prompts/`, `.github/`, `AGENTS.md`, `docs/CONTRACTS.md`), (3) present in HEAD with
working-tree bytes equal to HEAD, and (4) changed only by commits in `base_commit..HEAD` that touch no
`write_globs` file. `orchestration/tasks.json` also needs the task's own entry unchanged. At most 50 files
are overlaid; each is recorded in `base.json` (`rebased`: path, SHA-256, commits, action), printed to
stderr by `prepare`, and listed in `implementer.md` and `reviewer.md`. Anything else, including all
uncommitted and in-scope work, stays the task's delta. Every Git failure, a non-ancestor `base_commit`
(rewritten history), a root commit or a merge with more than two parents in the range or a missing
`base_commit` stops the retry (fail closed); only a directory without `.git` keeps the old behaviour. An attempt that predates this rule
needs `prepare --retry --base-commit <commit it was prepared on>`, which must be an ancestor of HEAD and
reproduce the recorded file hashes. A refused `prepare` leaves no half-made attempt.

Merges (TOOL-002): a two-parent merge in the range is allowed. It contributes the files that differ from its
first parent (`git diff-tree <merge>^1 <merge>`) and its hash is listed in `commits` for each; if any of them
matches `write_globs`, all of them are tainted (merging a feature into main therefore taints the feature's files
conservatively). Root commits, merges with 3+ parents and Git errors still fail closed.

New tools (TOOL-002): `tools/` stays denied unless `prepare --retry --allow-new-tool tools/<name>.py` names the path
(repeatable; rejected on a first attempt). The path must be a direct child `tools/<name>.py`, regular file in HEAD,
absent (casefold + NFC) from the original inventory and the `base_commit` tree with no same-stem file or package
directory, not a stdlib or installed top-level module, not `sitecustomize`/`usercustomize`/`conftest`/`__init__`/
`__main__`, not `orchestrate.py`/`check_tests.py` in any case; `base.json` records `allowed_new_tools` plus size and
first three lines per overlaid `tools/` path, shown in `prepare`, `implementer.md` and `reviewer.md`. The reviewer
must read each. AERIS-012: `python3 tools/orchestrate.py prepare AERIS-012 --retry --allow-new-tool tools/audit_publication.py`.
An allowed path that matches no overlay candidate prints a warning on stderr; `first_lines` lines are capped at
200 characters. See `docs/decisions/TOOL-002-retry-merges-and-tool-files.md`.

Packet contents: besides the contract files, the implementer packet pre-loads a `docs/decisions/*.md` record only
if the task's `read` list names it, the task's own `write_globs` match it, its filename starts with a direct
dependency's id, or it is `0001-scope.md`. All other records are listed by name under "Decision records not
included". The packet must still be at most 160 KB; `prepare` checks that before creating the attempt folder, so a
refusal leaves nothing behind, and a non-retry `prepare` returns a path only if `implementer.md` exists.

Residual risk: the rule cannot tell an implementer's own out-of-scope commit from another session's, so
the audit list is advisory and the independent reviewer must check it.

## Mandatory gates

1. Dependencies accepted; task is not deferred; no other active writer owns checkout.
2. Exact file-change list matches the report and all paths fit task ownership.
3. Required artifacts exist, each criterion cites an artifact and explanation,
   and the report contains a usable handoff with no unresolved implementation issues.
4. Repository-owned checks execute with argv lists, timeouts, captured output, and
   real exit codes. Required test files must exist and discover at least one test.
5. A fresh independent reviewer evaluates all criteria and scores all dimensions.
6. `accept` repeats evidence validation: source/report/check-log hashes and review
   bindings must still match. Only then is the task recorded in state.json.

The tool validates structure and evidence freshness, not the truth of every written
explanation. Checks, scores, report IDs and local receipts are not cryptographic
attestations against a malicious agent. The coordinator controls tool execution and
reviewer assignment. Repository source changes to the gate tool are outside normal
implementation scopes. Agents with filesystem access can still tamper; this workflow
is intended for cooperative engineering, not sandboxing untrusted executors.

Large data/models and generated outputs are not included in the source fingerprint.
Their integrity must be covered by manifests and task checks; code hashing alone
does not verify an external benchmark result. The baseline files are snapshotted for
review. Git branches/worktrees remain preferable for patch history and integration.

## Rubric (each dimension independently >=3)

| Score | Meaning |
|---|---|
| 0 | Missing work or unusable evidence |
| 1 | Major correctness or completeness gaps |
| 2 | Plausible but incomplete; material untested assumptions |
| 3 | Meets the task with reproducible evidence and stated limitations |
| 4 | Meets task and checks relevant adverse cases with unusually clear evidence |

Dimensions: correctness, tests, reproducibility, contracts, scope. Do not use a
weighted average that hides a failing dimension. Any unresolved critical/major
finding blocks acceptance regardless of numeric score. Minor findings can remain
only with explicit reviewer rationale and follow-up recorded in the review.

Critical: false results, leakage invalidating claims, incompatible data corruption.
Major: missing criterion, reproducible incorrect behavior, omitted required checks,
unsupported physical inference, material stale context. Minor: a bounded issue that
does not break task acceptance, such as an optional usability improvement.

## Feedback quality

Each finding should state file/location, triggering input, expected/actual behavior,
impact, and the smallest required correction. A retry gets all earlier findings and
the exact logs. Never ask the next agent to “make it better” without a falsifiable
criterion. Scope or contract changes require a reviewed planning change before a new
packet; they are not an escape hatch for failed implementation.

## Gate tiers

- Per task: offline workflow tests, task-specific behavioral tests, scope and review.
- Per integration: full unit/integration suite, C++ checks when involved, shared fixture parity.
- Per model change: controlled development benchmark versus locked baseline.
- Per release: real-video evidence, holdout evaluation, reproducibility review and
  actual hardware performance. A skipped unavailable hardware gate is not a pass;
  optional work may be excluded from the release with explicit scope.

The initial CI checks only the delivered foundation. Later tasks add real metric,
runtime, and regression checks; do not confuse a green scaffold with a finished system.
