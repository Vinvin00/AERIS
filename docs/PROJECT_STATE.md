# Durable project state

As of 2026-09-30: no perception pipeline, trained model, real video benchmark,
or published release exists. AERIS-001 through AERIS-004 are accepted. G1 is
complete as an engineering/data-QA milestone, not a detection milestone. Three
original FGVC image/label pairs were inspected; the full archive was not acquired.

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

## Accepted

- AERIS-001 canonical records (`python/aeris/data/contracts.py`, ADR 001).
- AERIS-002 dataset audit (`docs/dataset.md`, `configs/data/source_manifest.json`,
  ADR 002). FGVC is a still-image smoke check only; the owned video pilot is the only
  source that can back tracking/forecast claims, and it does not exist yet.
- AERIS-003 dataset loader and QA (`python/aeris/data/validate.py`,
  `python/aeris/data/fgvc.py`, `python/aeris/data/visualize.py`, ADR 003), accepted
  2026-09-30 after passing governance, 75 unit, and 21 dataset checks plus a fresh
  independent review. The three-image original FGVC QA sample includes one cabin
  interior requiring curation before exterior-aircraft evaluation. See
  `docs/data_qa.md` and `orchestration/runs/AERIS-003/attempt-03/`.
- AERIS-004 analytic ground-truth fixtures (`python/aeris/simulation/fixtures.py`,
  `tests/fixtures/simulation/`, ADR 004), accepted 2026-09-30 after 78 full-suite
  and 3 fixture tests plus independent review. Runtime inputs and evaluator truth
  are separate. These synthetic scenes test arithmetic and interfaces only.

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only.

## Reviewer follow-ups (minor, not yet done)

| From | Follow-up | Owner task |
|---|---|---|
| 001 | Mapping-bearing records are now explicitly unhashable; closed in AERIS-003 | Closed |
| 001 | `validate_sequence` accepts mixed `decode_pts`/`inferred_fps` sources in one sequence; decide reject or report | AERIS-008 |
| 002 N1 | Manifest validator checks key presence but not values: empty strings, empty `artifacts`/`sources`, non-bool flags, unknown status/evaluation names, `1e400` | AERIS-003 (when porting to `python/aeris/data/validate.py`) |
| 002 N2 | 9 of 35 validator-rule mutants survive the tests (fallback, role/status, width bins, duplicate IDs, `holdout_rule`) | AERIS-003 |
| 002 N3 | FGVC recall-only smoke check needs a fixed detection budget (top-k or pre-fixed score threshold) or it rewards box spam | AERIS-005/006 |
| 002 N4 | Wording: the report says 18 object patterns (17 in fact); ADR 002 says session splitting is "the only rule" preventing leakage (overstated) | next ADR 002 touch |

## Owner decisions blocking real video evidence (not G1)

Footage source and rights, capture sessions (at least 6), annotation tool, and approval
of the 24 h annotation/review budget. See `docs/dataset.md`.

Next action: `prepare AERIS-005` (metric adapters and evaluation protocol).
