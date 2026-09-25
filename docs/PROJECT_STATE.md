# Durable project state

As of 2026-09-24: no perception pipeline, downloaded dataset, trained model,
real benchmark, or published release exists. AERIS-001 and AERIS-002 are accepted
(see below). G0 is the engineering scaffold, not a detection milestone.

Decisions: fixed-camera RGB recorded-video MVP; original-frame pixel coordinates;
CPU first; ground truth and regression designed early; Python reference before C++
port; separate optional range and NVIDIA work; no software license selected yet.

Open decisions: hardware, hours/week, data source and rights, model/code/weight terms,
calibration availability, deployment FPS/resolution, training budget, project license.
Agents may proceed with offline contracts and fixtures without resolving all of these.
Dataset/model tasks must record their actual choices and limitations.

Operational state is in `orchestration/runs/state.json`, created on first acceptance.
Use `status` for readiness. Preserve the runs directory between sessions; it contains
packets, reports, checks, reviews, and feedback and is intentionally excluded from Git.
Copy durable task decisions into `docs/decisions/` and keep evidence in a backed-up
artifact store before moving machines. Do not commit private source data or secrets.

## Accepted (2026-09-24)

- AERIS-001 canonical records (`python/aeris/data/contracts.py`, ADR 001).
- AERIS-002 dataset audit (`docs/dataset.md`, `configs/data/source_manifest.json`,
  ADR 002). FGVC is a still-image smoke check only; the owned video pilot is the only
  source that can back tracking/forecast claims, and it does not exist yet.

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only.

## Reviewer follow-ups (minor, not yet done)

| From | Follow-up | Owner task |
|---|---|---|
| 001 | Records are frozen dataclasses but `hash()` raises (dict in `availability_reason`); set `__hash__ = None` or store reasons as a tuple | AERIS-003 |
| 001 | `validate_sequence` accepts mixed `decode_pts`/`inferred_fps` sources in one sequence; decide reject or report | AERIS-008 |
| 002 N1 | Manifest validator checks key presence but not values: empty strings, empty `artifacts`/`sources`, non-bool flags, unknown status/evaluation names, `1e400` | AERIS-003 (when porting to `python/aeris/data/validate.py`) |
| 002 N2 | 9 of 35 validator-rule mutants survive the tests (fallback, role/status, width bins, duplicate IDs, `holdout_rule`) | AERIS-003 |
| 002 N3 | FGVC recall-only smoke check needs a fixed detection budget (top-k or pre-fixed score threshold) or it rewards box spam | AERIS-005/006 |
| 002 N4 | Wording: the report says 18 object patterns (17 in fact); ADR 002 says session splitting is "the only rule" preventing leakage (overstated) | next ADR 002 touch |

## Owner decisions blocking real video evidence (not G1)

Footage source and rights, capture sessions (at least 6), annotation tool, and approval
of the 24 h annotation/review budget. See `docs/dataset.md`.

Next action: `prepare AERIS-003` (dataset loader and QA, last G1 task). AERIS-004 is
also ready but belongs to G2.
