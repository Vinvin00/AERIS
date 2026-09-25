# Durable project state

As of 2026-09-25: no perception pipeline, downloaded dataset, trained model,
real benchmark, or published release exists. G1 (AERIS-001 to 003) and the G2 tasks
AERIS-004 and 005 are accepted (see below). They are data contracts, audit and QA
tooling, analytic synthetic fixtures and pinned evaluators, not perception evidence. G0 is the engineering
scaffold, not a detection milestone.

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

## Run-archive recovery (2026-09-25)

The repository reached GitHub as one commit (`0d28c9a`) without `orchestration/runs/`.
The 2026-09-24 acceptances of AERIS-001/002 and the unfinished AERIS-003 attempt could
therefore not be carried over (`recovery.md`). A cloud coordinator session reconstructed
each task from source:

1. Baseline: the tree minus the files the task created.
2. Report: reconstructed by the coordinator from the source.
3. `verify`, then a fresh independent subagent review.
4. `accept`.

Each evidence folder has a `RECONSTRUCTION.md`. The reconstructed archive was given to
the owner as `aeris-runs-2026-09-25.tar.gz`. Newer copies followed:
`aeris-runs-2026-09-25-g2a.tar.gz` adds AERIS-004 and `aeris-runs-2026-09-25-g2b.tar.gz`
adds AERIS-005. Extract the newest copy at the repository root so
`orchestration/runs/state.json` exists before the next `prepare`; without it, `status`
shows AERIS-001 as ready again. Keep the original macOS archive if it still exists: it
holds the original reviews and the bounded FGVC real-sample evidence cited in
`docs/data_qa.md`, which this session could not see.

This cloud environment's network policy blocks web fetches from the dataset and paper
hosts (`robots.ox.ac.uk`, `arxiv.org`, `github.com`), so measured external facts
(FGVC checksums, Last-Modified times) were not re-measured during reconstruction.

## Accepted (2026-09-25, reconstructed)

Scores are correctness/tests/reproducibility/contracts/scope out of 4.

- AERIS-001 canonical records (`python/aeris/data/contracts.py`, ADR 001).
  Review: accept, 4/3/3/4/4.
- AERIS-002 dataset audit (`docs/dataset.md`, `configs/data/source_manifest.json`,
  ADR 002). Review: accept, 3/3/3/4/4. FGVC is a still-image smoke check only; the
  owned video pilot is the only source that can back tracking/forecast claims, and it
  does not exist yet.
- AERIS-003 dataset loader and QA (`python/aeris/data/{validate,fgvc,visualize}.py`,
  `docs/data_qa.md`, ADR 003). Review: accept on attempt 2 of 3, 4/3/4/3/4.
  Attempt 1 was sent back for one major finding, since fixed: counts of unannotated
  attributes were reported as zeros instead of null. The completion changes added a
  Linux x86_64 CPython 3.11 hash lock and made contract JSON parser failures
  reportable. They also made records explicitly unhashable.

## Accepted (2026-09-25)

- AERIS-004 analytic ground-truth fixtures (`python/aeris/simulation/fixtures.py`,
  `tests/fixtures/simulation/`, ADR 004). Review: accept on attempt 1, 4/3/4/3/4.
  Image-space and calibrated projection families are split into `inputs/` and `truth/`.
  CI's Python 3.12 job caught a byte-determinism bug before the verdict: built-in
  `sum()` over floats changed in 3.12. It was fixed within the attempt. Generation is
  byte-identical on CPython 3.10 to 3.13; frames were checked on 3.11 to 3.13.
- AERIS-005 metric adapters and evaluation protocol (`python/aeris/evaluation/`,
  `docs/evaluation.md`, ADR 005). Review: accept on attempt 2 of 3, 4/4/4/4/4. The
  adapters are:
  - pinned pycocotools 2.0.11 COCOeval on complete frames, with ignore and partial frames
    handled as crowd regions;
  - official TrackEval commit `12c8791b` (VCS pin, scoped `np.float`/`np.int` shim;
    parity with unpatched NumPy 1.23 was identical);
  - an oracle-identity forecast protocol with a model-independent eligible set.

  Attempt 1 was sent back for one major finding, since fixed: tracking distractor
  removal deleted correct boxes that overlapped an ignored object. It now uses
  TrackEval's joint MOTChallenge assignment. Sequence-bootstrap intervals need at least
  10 sequences.

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only. One reviewer session reviewed 001 and 002. A fresh
session reviewed both 003 attempts; the coordinator co-authored 003. Fresh sessions
reviewed 004 and both 005 attempts, which the coordinator wrote.

## Reviewer follow-ups (minor, not yet done)

| From | Follow-up | Owner task |
|---|---|---|
| 001 F1 | 17 of 34 contract mutants survive. The untested checks are ID, version and timestamp checks on detection, prediction, truth and annotation records; there are no `annotation_id`/`track_id` identifier tests. The reconstructed 001 report (F2) wrongly claimed them | next contracts task (unassigned) |
| 001 F4 | `validate_sequence` accepts mixed `decode_pts`/`inferred_fps` sources in one sequence; decide reject or report | AERIS-008 |
| 002 N1/F1 | Manifest validator value checks: 27 of 40 invalid values pass (no sources, `False` class ID, `1e400`, unknown evaluation names). Not ported to `python/aeris/data` by AERIS-003 | unassigned |
| 002 N2/F2 | 11 of 38 validator-rule mutants survive | unassigned |
| 002 N3/F3 | FGVC recall-only smoke check needs a fixed detection budget. AERIS-005's evaluator now supports a pre-fixed inclusive score threshold and a per-frame top-k budget; the actual values still have to be fixed before any FGVC run | AERIS-006/007 |
| 002 N4/F4 | Wording: ADR 002 overstates session splitting as "the only rule" preventing leakage; the original report said 18 object patterns (17) | next ADR 002 touch |
| 003 A2 F1 | A `true` flag on an attribute listed in `unannotated_attributes` is accepted silently. With `ignore` listed, the statistics reason then wrongly says the width bins count every object. Reject or report such flags | next task owning `python/aeris/data/*` (unassigned; outside AERIS-005's scope) |
| 003 A2 F2 | Two mutants survive. A sheet that never prints NEGATIVE passes `test_sheet_captions_and_attribute_tags` (the attempt-2 report overstated that test). No test lists `ignore` as unannotated | next task owning `python/aeris/data/*` (unassigned) |
| 004 F2 | Some generated fields are protected only by the byte snapshot, not by semantic assertions: vertical angular extent, range, range rate, detection class ID, false-positive score, noise model and manifest record counts. The reviewer found 7 such mutants. The coordinator's 25-mutant run is now saved in the evidence folder | next task touching `aeris/simulation` |
| 004 F3 | Truth angular extent is measured between marker rays. CONTRACTS.md defines the bbox extent between edge rays (≤ 4.6e-6 relative difference here). Document it in ADR 004 or add edge-ray truth | AERIS-011 |
| 004 F4 | Synthetic detector scores identify row kinds (duplicate exactly 0.55, false positive 0.30), so a threshold between 0.55 and 0.70 silently drops both. AERIS-005's tests probe thresholds 0.25/0.5/0.55/0.9 around them; vary the scores before tracker threshold tests | AERIS-009 or next `aeris/simulation` task |

Closed by AERIS-003:

- Records holding `availability_reason` are explicitly unhashable (001 follow-up).
- `contracts.loads` reports RecursionError, oversized-integer and undecodable input as
  `ContractError`, so the loader reports such lines instead of crashing (001 F5).
- All six attempt-1 review findings: statistics nulls, stronger tests, unverified claims
  qualified, CLI errors, per-line decoding, and FGVC split subsets.

## Owner decisions blocking real video evidence (not G1)

Footage source and rights, capture sessions (at least 6), annotation tool, and approval
of the 24 h annotation/review budget. See `docs/dataset.md`.

Integration: branch `claude/inspiring-keller-xf368q`, draft PR Vinvin00/AERIS#1. CI
(GitHub Actions, Python 3.11/3.12 and C++) passed on the accepted AERIS-005 source
(`67a71b2`). CI installs the pinned evaluators through the `dev` extra.

Next action: restore `orchestration/runs/` from the newest archive. AERIS-006
(pretrained detector and ONNX export spike, G2) is ready but needs owner decisions first:
which detector code and weights (and so which license terms) to adopt, and a download
and install budget for PyTorch, ONNX Runtime and weights. AERIS-007 (measured baseline)
then needs real labelled data, which is also blocked on owner decisions (see above).
