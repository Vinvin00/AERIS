# PLAN-003 - Split AERIS-007 into an offline protocol task and a deferred real-baseline task

Status: reviewed, approved with changes (applied). Date: 2026-10-07. Owner decision: on 2026-10-07 the
project owner chose option 1 ("split AERIS-007 via planning change") after the coordinator
reported that the task cannot be met as written. The owner can overrule this by reverting the commit.

## Why

AERIS-007 asks for "a real labeled baseline": AP, size recall, negatives and supports, a
frozen development split and thresholds from measured repeatability, plus a separate holdout.
The only labeled real data is three FGVC images whose frames are `partial`; the AERIS-005
evaluator excludes partial frames by design, so AP is unavailable on them, and one of the
three is a cabin interior. No complete-label real set exists (see `docs/dataset.md`; the owner
decisions on footage, rights and annotation budget are open). The criterion depends on missing
external data, so this is a task-contract planning failure, not an implementation failure. No
threshold, test or code was changed.

## What changes

1. `AERIS-007` keeps the offline half: reproducibility manifest (model/config/environment/sample
   digests, stating it is not a real baseline), the metric report with unavailable-with-reason
   behaviour, a regression-config schema and the threshold-selection protocol. Its objective
   and acceptance text are rewritten (visibly weaker: no real measurement). Its required
   `configs/evaluation/regression.json` becomes `regression_schema.json`; no threshold value is
   set without a measured real development set.
2. New deferred `AERIS-022` carries the original real-measurement requirements (real digests
   and provenance, measured AP/size recall/negatives/supports, thresholds frozen from measured
   repeatability, holdout separate and unopened). It cannot be dispatched until the owner
   supplies complete-label real data; activation needs a separately reviewed change.
3. `AERIS-015` (regression gates) and `AERIS-017` (tiny-target experiment) now depend on
   `AERIS-022`, because both compare against an approved real baseline; `AERIS-016` (CPU
   performance, which depends on 015) and `AERIS-018` are blocked transitively. 016's own
   criteria are latency and parity based and need no quality baseline, but it is built on the
   regression-gate task, so it waits too; relaxing that dependency would be a separate decision. `AERIS-012` still depends on `AERIS-007` only, so the integrated MVP can proceed
   and must report truth-dependent metrics as unavailable. The ROADMAP G2 note, `docs/dataset.md`
   and `docs/PROJECT_STATE.md` point to 022.

## Coverage preserved

Original AERIS-007 clause | Now carried by
--- | ---
Real data/model/config/environment/artifact digests make the baseline reproducible | 007 AC1 (smoke sample, labelled not-real) and 022 AC1 (real)
AP, size recall, negatives and supports reported, no copied plan numbers | 007 AC2 (schema and unavailable behaviour) and 022 AC2 (measured)
Thresholds frozen before candidate comparisons; holdout separate | 007 AC3 (protocol, no values) and 022 AC3 (frozen values)

## Consequences

- AERIS-007 can be accepted with no real measurement; no real detection quality is claimed.
- G2's "real baseline" exit evidence stays unmet until AERIS-022; G5-G7 items that need an
  approved baseline (015, 017) and everything behind them (016, 018) stay blocked until then.
- AERIS-007's AC2 "unavailable with reasons" behaviour is already provided by the AERIS-005
  evaluator; 007 satisfies it by wiring and demonstration, not by new metric code.
- The regression gate task cannot invent thresholds: it depends on the real baseline.

## Mechanical consequence: AERIS-007 write scope

Retries keep attempt 01's file snapshot, and AERIS-007's attempt 01 was prepared before this
change, so the planning files below would appear in its change list and fail the scope check
(as for PLAN-001 and PLAN-002). AERIS-007's `write_globs` therefore also list this plan,
`orchestration/tasks.json`, `docs/ROADMAP.md`, `docs/dataset.md` and `docs/PROJECT_STATE.md`,
for the duration of AERIS-007 only. The AERIS-007 reviewer should confirm that those files contain
only the changes described here; the AERIS-007 report records the planning commit hash so
the reviewer can diff against it. This repeats a workaround; a tooling change (per-attempt
baselines or excluding reviewed planning commits) would remove the need and would itself be a
separate reviewed task.

## Review (independent session, 2026-10-07)

Verdict: approve with changes. Verified: FGVC frames are `partial` and excluded from AP/FP by
the AERIS-005 evaluator, one sample image is a cabin interior, no complete-label real set exists,
and the footage/annotation owner decisions are open, so the original criteria depended on missing
data; no code, test, gate or threshold changed; AERIS-022 is well formed and `prepare` refuses
deferred tasks; AERIS-012 depending on 007 only is right because its criteria already require
null truth-dependent metrics with reasons. Applied: disclosure that 016 is blocked transitively
(here, in the ROADMAP note and in `docs/PROJECT_STATE.md`), this review section, the AC2 wiring
note, the commit-hash instruction, and rewrapped long lines in `docs/dataset.md` and
`docs/PROJECT_STATE.md`.
