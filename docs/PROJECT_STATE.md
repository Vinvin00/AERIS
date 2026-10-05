# Durable project state

As of 2026-09-30: no perception pipeline, trained model, real video benchmark,
or published release exists. AERIS-001 through AERIS-006, AERIS-008, AERIS-009 and AERIS-011 are accepted. G1 is
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
- AERIS-005 metric adapters and evaluation protocol (`python/aeris/evaluation/metrics.py`,
  `docs/evaluation.md`, ADR 005), accepted 2026-10-01 on attempt 03 after 87 full-suite
  and 9 metrics tests plus fresh independent review. Pinned pycocotools 2.0.11 and
  TrackEval 1.3.0 run on synthetic fixtures only; no real AP/HOTA/IDF1 exists. Attempt 02
  was closed by a coordinator sequencing error (`verify` resets `review.json`), not a
  code defect. Minor follow-up: AERIS-002 N3 (fixed detection budget) is now handled by
  the per-frame budget and required score threshold.
- AERIS-006 pretrained detector baseline (`python/aeris/detection/baseline.py`,
  `configs/detector/baseline.json`, `docs/detection.md`, ADR 006), accepted 2026-10-02
  on attempt 01 after 102 full-suite tests and fresh independent review. YOLOX-Tiny
  COCO ONNX (Apache-2.0 code; weight terms unverified), ONNX Runtime CPU vs OpenCV DNN
  agree within declared tolerances on three FGVC images. Smoke check only: no quality
  claim, no PyTorch export-fidelity check, weights Git-ignored (see `models/README.md`).
- AERIS-008 timestamp-aware video pipeline (`python/aeris/video/pipeline.py`,
  `docs/video.md`, ADR 008), accepted 2026-10-02 on attempt 02 (attempt 01 revised for a
  skipped PTS count check and an unasserted total) after 121 full-suite tests and fresh
  review. ffprobe PTS with marked inferred fallback; mixed-source helper closes the
  AERIS-001 follow-up. No owned real video exists; the real sample is derived from one
  still photo. ffprobe is an unpinned external tool.
- AERIS-009 Kalman tracker (`python/aeris/tracking/tracker.py`, `configs/tracker/`,
  `docs/tracking.md`, ADR 009), accepted 2026-10-05 on attempt 02 (attempt 01 was escalated
  because AC3 needed identity-labeled real video; AC3 was amended by PLAN-001, and AERIS-021
  now carries the real-video requirement, deferred). 141 full-suite tests, fresh review.
  **No real tracking performance exists.** Defaults are untuned.
- AERIS-011 camera geometry (`python/aeris/estimation/camera.py`, `docs/camera_model.md`,
  ADR 011), accepted 2026-10-05 on attempt 01 after 160 full-suite tests and fresh review.
  Rays verified against an independent distortion model on analytic/synthetic cameras only;
  no real calibration exists. Angular claims are suppressed without calibration, after
  camera motion or on a zoom mismatch; no range, approach or yaw claims.

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only.

## Reviewer follow-ups (minor, not yet done)

| From | Follow-up | Owner task |
|---|---|---|
| 001 | Mapping-bearing records are now explicitly unhashable; closed in AERIS-003 | Closed |
| 001 | `validate_sequence` accepts mixed `decode_pts`/`inferred_fps` sources; AERIS-008 added a rejecting helper in the video pipeline, `validate_sequence` itself unchanged (contract change) | Helper done; contract change open |
| 002 N1 | Manifest validator checks key presence but not values: empty strings, empty `artifacts`/`sources`, non-bool flags, unknown status/evaluation names, `1e400` | AERIS-003 (when porting to `python/aeris/data/validate.py`) |
| 002 N2 | 9 of 35 validator-rule mutants survive the tests (fallback, role/status, width bins, duplicate IDs, `holdout_rule`) | AERIS-003 |
| 002 N3 | FGVC recall-only smoke check needs a fixed detection budget (top-k or pre-fixed score threshold) or it rewards box spam | AERIS-005/006 |
| 006 | 1627560 class/box not asserted in a test; airplane scored per anchor instead of YOLOX argmax-class selection (undocumented); confirm weight license before release | AERIS-007+/release |
| 008 | p95 not asserted in tests; read-back dimension comparison unexercised; max_frames equal to clip length cannot detect extra PTS | later video task |
| 009 | Tentative hits are non-consecutive; velocity test overrides default accel sigma 200 with 2.0; equal-score simultaneous births get IDs by input order | AERIS-021 development split |
| 011 | Fixed 2026-10-05 after acceptance (invertibility check, `ray_to_pixel` None outside frame, SE n-2 test); post-acceptance change not under a formal receipt or fresh review | Closed (see commit) |
| 002 N4 | Wording: the report says 18 object patterns (17 in fact); ADR 002 says session splitting is "the only rule" preventing leakage (overstated) | next ADR 002 touch |

## Owner decisions blocking real video evidence (not G1)

Footage source and rights, capture sessions (at least 6), annotation tool, and approval
of the 24 h annotation/review budget. See `docs/dataset.md`. These also block the deferred
AERIS-021 (real-video tracking evidence), which AERIS-018 now depends on; AERIS-009's AC3 was
split by `docs/decisions/PLAN-001-aeris-009-ac3-split.md`.

Next action: `status`, then `prepare` the next ready task.
