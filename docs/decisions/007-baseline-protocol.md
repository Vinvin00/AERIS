# ADR 007 — Smoke manifest, report contract and a schema-first threshold protocol

Status: proposed for AERIS-007 review. Date: 2026-10-07. See PLAN-003 for why the original
real-baseline criteria were split out to the deferred AERIS-022.

With no complete-label real data, thresholds cannot be derived from measured repeatability, so
none are created. Instead: (1) a digest-complete manifest that is explicitly not a baseline and
whose tests recompute every digest, including the evaluator's, so silent drift is caught;
(2) the existing AERIS-005 report contract demonstrated on synthetic complete labels and on
real partial labels (unavailable, with a reason); (3) a regression-config JSON schema with
cross-field rules (no values in a draft, complete repeatability evidence and a pre-comparison
freeze record when frozen, tolerance not below measured noise, holdout never opened) so the later
real baseline cannot be filled in loosely. Enforcement code for the rules belongs to
AERIS-015; the tests carry only a reference checker. Alternative not taken: a placeholder
`regression.json` with provisional numbers, which would invite treating illustrative values as
gates.
