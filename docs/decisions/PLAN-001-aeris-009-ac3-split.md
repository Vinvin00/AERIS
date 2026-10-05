# PLAN-001 — Split AERIS-009 AC3 into a synthetic part and a deferred real-video task

Status: reviewed, approved with changes (applied). Date: 2026-10-05. Owner decision: on
2026-10-05 the project owner chose option 2 (split the criterion through a reviewed
planning change) in the coordinating session, with the words "option 2, and start
AERIS-011". The owner can overrule this by reverting the commit.

## Why

AERIS-009 AC3 originally read: "Tracking metrics run on identity-labeled real video;
absent labels block real-performance claims." No identity-labeled real video exists
(footage source and rights, a capture plan and the annotation budget are open owner
decisions; see `docs/dataset.md`). An independent reviewer found AC1 and AC2 met and AC3
unmeetable by any code change, and escalated rather than accepting synthetic evidence.
This is a planning failure of the task contract (a criterion that depends on missing
external data), not an implementation failure. No threshold or test was changed.

## What changes

1. `AERIS-009` AC3 becomes: the detections-tracker-TrackEval path runs on
   identity-labeled synthetic scenes and the AERIS-004 analytic fixture, and absent identity
   labels return unavailable tracking metrics that block real-performance claims; real-video
   metrics are carried by AERIS-021. This is exactly what the existing tests and fixtures
   evidence. It is a weaker criterion than before, and that is deliberate and visible here.
2. New `AERIS-021` (deferred): the original real-video requirement, with its own
   criteria (real HOTA/IDF1 with support, provenance and rights, development/held-out
   discipline). It cannot be dispatched until the owner supplies footage and an annotation
   plan, and activation needs a separately reviewed change.
3. `AERIS-018` (release) now depends on `AERIS-021`, so a release cannot be assembled
   while real tracking evidence is missing ("Missing required evidence is a blocked
   release, never a fabricated pass" - CONTRACTS.md). The ROADMAP G3 membership notes 021;
   G3's exit evidence ("MOT report on real identities") is not met until 021 is accepted.

## Coverage preserved

Original AC3 clause | Now carried by
--- | ---
Tracking metrics run on identity-labeled real video | AERIS-021 AC1 (deferred)
Absent labels block real-performance claims | AERIS-009 AC3 (tested: unlabeled identities give `available: false`, null HOTA/IDF1)

## Consequences

- AERIS-009 may be accepted without any real-video tracking number; the project makes no
  real tracking claim and `docs/tracking.md` says so.
- The forecasting tasks (AERIS-010 onward) are unblocked from AERIS-009's perspective but
  inherit the same absence of real labels.
- A reviewer must still confirm AERIS-009's evidence against the new AC3 wording.

## Review (independent session, 2026-10-05)

Verdict: approve with changes. The change is a legitimate contract correction rather than
an escape hatch: the criterion depended on data that does not exist and no code, test,
threshold or gate was altered. Applied from the review: AERIS-021 now also depends on
AERIS-003 and AERIS-005, gains evaluation/manifest write scope and a required manifest
artifact; the ROADMAP G3 exit evidence and G7 note, `docs/dataset.md` and
`docs/PROJECT_STATE.md` point to 021; this plan is committed separately from the AERIS-009
code. Carried to 021's development split: non-consecutive tentative hits and the loose
default acceleration sigma. Not checked: `tools/orchestrate.py` has no gate-scoped
"request through a gate" logic that treats deferred tasks specially; only `status`,
`prepare` (refuses deferred tasks) and `lint` handle them.
