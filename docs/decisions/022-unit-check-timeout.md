# ADR 022 — Unit-check timeout margin

Status: proposed for AERIS-022 attempt 1 review. Date: 2026-09-27.

Context: `orchestration/gates.json`'s `unit` check (`tools/check_tests.py`, the full
`tests/unit/` suite) has run against a 120 s timeout since G1. The suite has grown with
every task since (327 tests by AERIS-009, 353 by AERIS-010), and this was already a
disclosed, deferred governance follow-up ("008/009 unit timeout") after AERIS-009's
`verify` timed out once (exit 124) before passing cleanly on an immediate retry.
AERIS-010 escalated it from an occasional near-miss to a consistent failure: `verify`
timed out on the `unit` check three consecutive times while every other check (including
its own task-specific `trajectory` check) passed cleanly, blocking acceptance for reasons
unrelated to the implementation's correctness.

Decision: raise `unit`'s `timeout_s` from 120 to **240** in `orchestration/gates.json`,
a data-only change (no change to `tools/orchestrate.py`, the gate tool itself, which
EVALUATION_LOOP.md says is "outside normal implementation scopes" for a normal task).
No other check's `argv`, `timeout_s`, or the check set changes.

Measured evidence (this build, `.venv/bin/python tools/check_tests.py`, unconstrained):

| Run | Tests | Internal time | Wall time |
|---|---|---|---|
| AERIS-009 era (reported in its own attempt-2 report) | 327 | 118.8 s | 111–119 s across runs |
| This session, run 1 | 353 | 131.238 s | 134.044 s |
| This session, run 2 | 353 | 121.071 s | 123.989 s |

240 s is roughly 1.8–2x the slower of these two measurements, giving headroom for
further, honest test-suite growth and for the same shared-VM wall-clock variance that
caused AERIS-009's own single near-miss, without being so large that a genuinely hung
process would go undetected for an unreasonable time. It matches the same order of
magnitude as the existing `cpp-build` check's already-larger 300 s budget, rather than
introducing a new scale of timeout into the gate set.

Alternatives rejected:

- **Splitting `tests/unit/` into parallel or per-module gate checks.** This would keep
  each check comfortably under 120 s, but it changes `tools/orchestrate.py`'s and
  `tools/check_tests.py`'s own behavior (how the suite is invoked and aggregated) and
  every task's `checks` list, a materially larger change than this task's narrow,
  data-only scope. Worth considering in a future governance task once the suite grows
  enough to make a single 240 s run itself impractical.
- **Leaving the timeout at 120 s and treating each failure as a flake to retry.**
  Rejected: AERIS-010 already showed three consecutive timeouts, not an occasional
  variance; retrying indefinitely against a budget the suite structurally exceeds is not
  a fix, and AGENTS.md requires escalating a repeated failure rather than looping on it.
- **Raising the timeout much higher (e.g. 600 s), to avoid revisiting this soon.**
  Rejected as unjustified by any measurement: the current suite needs roughly 2x margin,
  not 5x; a timeout far beyond the evidence would mask a genuine hang for too long.

Consequences:

- Every task whose `checks` list includes `unit` (all of G2 onward) now runs that check
  under the new 240 s budget; no task's own write scope needed to change for this.
- This does not address the separately flagged governance gap that `AERIS_REQUIRE_REAL_*`
  environment variables are not set by any gate (006/007/008/009/010): `tools/orchestrate.py`'s
  check runner has no per-check `env` field today, and adding one is a `tools/orchestrate.py`
  source change, which EVALUATION_LOOP.md places outside a normal task's scope. That
  follow-up stays open for a future, larger governance task that is willing to touch the
  gate tool itself, with its own review.
- If the suite grows enough that 240 s stops being a comfortable margin, splitting the
  suite (the rejected alternative above) becomes the more appropriate fix, not a further
  timeout increase without new measurement.
