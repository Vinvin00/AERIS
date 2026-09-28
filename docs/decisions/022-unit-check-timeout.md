# ADR 022 — Unit-check timeout margin

Status: proposed for AERIS-022 attempt 2 review. Date: 2026-09-27/28.

Context: `orchestration/gates.json`'s `unit` check (`tools/check_tests.py`, the full
`tests/unit/` suite) has run against a 120 s timeout since G1. The suite has grown with
every task since (327 tests by AERIS-009, 353 by AERIS-010), and this was already a
disclosed, deferred governance follow-up ("008/009 unit timeout") after AERIS-009's
`verify` timed out once (exit 124) before passing cleanly on an immediate retry.
AERIS-010 escalated it from an occasional near-miss to a consistent failure: `verify`
timed out on the `unit` check three consecutive times while every other check (including
its own task-specific `trajectory` check) passed cleanly, blocking acceptance for reasons
unrelated to the implementation's correctness.

Decision: raise `unit`'s `timeout_s` from 120 to **900** in `orchestration/gates.json`,
a data-only change (no change to `tools/orchestrate.py`, the gate tool itself, which
EVALUATION_LOOP.md says is "outside normal implementation scopes" for a normal task).
No other check's `argv`, `timeout_s`, or the check set changes.

Measured evidence (`.venv/bin/python tools/check_tests.py`, unconstrained, unmodified
suite and command throughout):

| Run | Tests | Internal time | Wall time | Conditions |
|---|---|---|---|---|
| AERIS-009 era (its own attempt-2 report) | 327 | 118.8 s | 111–119 s across runs | not recorded |
| Attempt 1, run 1 | 353 | 131.238 s | 134.044 s | quiet |
| Attempt 1, run 2 | 353 | 121.071 s | 123.989 s | quiet |
| Attempt 1's independent reviewer, run 1 | 353 | 528.697 s | 563.658 s | contended: load average up to 3.98 on 4 vCPUs, immediately after that review's own container started (environment manager, telemetry collector and interactive harness competing for CPU during boot) |
| Attempt 1's independent reviewer, run 2 (minutes later) | 353 | 117.751 s | 120.107 s | quiet: load average back down to 0.33–1.5 |

**Revision history.** Attempt 1 proposed 240 s (roughly 1.8–2x the two quiet same-session
measurements available at the time) and was sent back on independent review: the
reviewer re-ran the identical, unmodified gate command twice and reproduced a
contended run at 528.7 s internal / 563.7 s wall on this same infrastructure — over
2.2x the proposed 240 s budget, from ambient VM contention alone, with no change to the
code or suite between that run and a quiet 117.8 s/120.1 s run minutes later. That
falsified attempt 1's central claim (that 240 s covered realistic shared-VM variance):
under conditions the reviewer directly reproduced, `verify`'s unit check would still
time out at 240 s, the exact failure this task exists to prevent. This revision widens
the evidence base to include that contended sample rather than only quiet-VM runs, and
re-derives the margin against it.

900 s is derived directly from the worst *reproduced* condition (563.658 s wall, the
reviewer's contended run): roughly 1.6x that figure, rather than 1.8–2x over a quiet
run as attempt 1 did — the same order-of-magnitude margin philosophy as before, now
anchored to the more representative (worse) data point instead of the one that turned
out not to bound real contention. This is still bounded, not "as large as possible":
a timeout far beyond any observed condition would mask a genuine hang for an
unreasonable time, and 900 s is the largest budget in the gate set (`cpp-build`'s is
the next-largest at 300 s) only because the unit check is, honestly, the most
contention-exposed one measured so far, not because a round, generous number was
chosen without evidence.

**Residual risk, stated plainly, not solved away.** VM contention is not bounded in
principle: a future run could in theory exceed even 900 s if ambient load is worse than
anything measured here. This ADR does not claim 900 s is a guarantee, only that it
covers every condition actually reproduced during this task's two attempts, with margin.
If `verify`'s unit check times out again at 900 s, that is new evidence the residual
risk has materialized, not a sign this ADR was wrong on the evidence it had — the
correct response then is to gather that new data point and either raise the timeout
again or move to splitting the suite (below), not to reflexively raise the number
further without measurement.

Alternatives rejected:

- **Splitting `tests/unit/` into parallel or per-module gate checks.** This would make
  each check's own budget far less exposed to whole-suite contention, but it changes
  `tools/orchestrate.py`'s and `tools/check_tests.py`'s own behavior (how the suite is
  invoked and aggregated) and every task's `checks` list, a materially larger change
  than this task's narrow, data-only scope. This is the escalation path if 900 s later
  proves insufficient (see "Residual risk" above), not a fix attempted now.
- **Leaving the timeout at 120 s (or attempt 1's 240 s) and treating each failure as a
  flake to retry.** Rejected: AERIS-010 already showed three consecutive timeouts at
  120 s, and attempt 1's own reviewer reproduced a failure of the 240 s budget within
  one review session — not occasional variance either time; retrying indefinitely
  against a budget the suite structurally exceeds is not a fix, and AGENTS.md requires
  escalating a repeated failure rather than looping on it.
- **A timeout with no evidence behind it at all (e.g. an arbitrary 1 hour "to be safe").**
  Rejected: 900 s is chosen specifically because it is the smallest round-ish margin
  that covers the worst condition actually measured; inflating it further without a
  worse measurement would repeat attempt 1's mistake in the opposite direction —
  picking a number the evidence doesn't actually require.

Consequences:

- Every task whose `checks` list includes `unit` (all of G2 onward) now runs that check
  under the new 900 s budget; no task's own write scope needed to change for this.
- A `verify` run whose only failure is a genuinely hung unit test will now take up to
  900 s to report that, instead of 120 s. This is the accepted cost of not reproducing
  AERIS-010's false-failure mode on ordinary contention.
- This does not address the separately flagged governance gap that `AERIS_REQUIRE_REAL_*`
  environment variables are not set by any gate (006/007/008/009/010): `tools/orchestrate.py`'s
  check runner has no per-check `env` field today, and adding one is a `tools/orchestrate.py`
  source change, which EVALUATION_LOOP.md places outside a normal task's scope. That
  follow-up stays open for a future, larger governance task that is willing to touch the
  gate tool itself, with its own review.
- If a future `verify` run times out at 900 s under conditions similar to those already
  reproduced (not worse), that would mean this ADR's evidence was misread, not that the
  risk was unknown; if it happens under materially worse contention than measured here,
  splitting the suite (the first rejected alternative above) is the appropriate next
  step, not a further blind timeout increase.
