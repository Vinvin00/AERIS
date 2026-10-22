# Durable project state

As of 2026-09-30: no perception pipeline, trained model, real video benchmark,
or published release exists. AERIS-001 through AERIS-006, AERIS-007, AERIS-008, AERIS-009, AERIS-010, AERIS-011, TOOL-001 and TOOL-002 are accepted. G1 is
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
- AERIS-010 causal forecasts (`python/aeris/estimation/trajectory.py`,
  `docs/trajectory_estimation.md`, ADR 010), accepted 2026-10-06 on attempt 02 (attempt 01 was
  closed by a separate owner-directed session before any forecast code existed; PLAN-002 widened
  the write scope to the MAINT-001 files) after 179 full-suite tests and fresh review.
  Constant-velocity and separately specified acceleration filters, visible misses, errors and
  interval coverage on synthetic data only. **No real forecast accuracy or calibration exists**;
  the acceleration model is not generally better than constant velocity (see docs).
- TOOL-001 orchestrator retry baselines (`tools/orchestrate.py`, `tests/test_orchestration.py`,
  `docs/EVALUATION_LOOP.md`, ADR TOOL-001, PLAN-004), accepted 2026-10-07 on attempt 01 after 218
  tests and an adversarial independent review that found no bypass. A retry baseline now overlays
  clean, unprotected, out-of-scope files changed only by commits touching no write_globs file
  (audited in `base.json`, `prepare` output and the packets); Git errors fail closed. Limits: the
  denylist (verifier and contract paths) and a task's own `tasks.json` entry are never overlaid, so
  retries after accepted tooling or contract changes still need explicit handling; the audit list
  is advisory.
- AERIS-007 offline baseline protocol (`benchmarks/baseline_manifest.json`,
  `configs/evaluation/regression_schema.json`, `docs/baseline.md`, ADR 007, PLAN-003/005), accepted
  2026-10-07 on attempt 02 after 218 tests and fresh review that recomputed every digest. The
  manifest is explicitly `smoke_not_a_baseline`; on the three partially annotated FGVC images the
  metric report is unavailable (`no_complete_annotations`). **No measured baseline, split or
  threshold exists**; those are the deferred AERIS-022.

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only.

- AERIS-023 to AERIS-028 (exact tracker measurements; forecast timing and noise; forecasts scored
  against later detections; arrow heading gate; low-score track extension; scene-relative forecasts
  under camera motion, default `--camera-motion estimate`), accepted 2026-10-17 by historical replay
  under PLAN-010 (`docs/decisions/PLAN-010-replay-acceptance-023-028.md`) in an isolated worktree,
  each with a fresh independent reviewer. 024, 026 and 028 needed 2, 2 and 3 attempts; every
  review-driven test, ADR correction and metadata change was then landed on `main`. Evidence:
  `orchestration/runs/AERIS-02{3..8}/`. One-clip smoke numbers only; **no real tracking, direction
  or forecast accuracy exists**.

- TOOL-002 orchestrator retry across merges and opted-in tool files (`tools/orchestrate.py`,
  `tests/test_orchestration.py`, `docs/EVALUATION_LOOP.md`, ADR TOOL-002, PLAN-009), accepted
  2026-10-20 on attempt 02 (attempt 01 revised for untested guards and inaccurate evidence) after 78
  orchestration tests, 328 full-suite tests and a fresh independent review (30 mutations, 29 killed,
  the survivor equivalent). Two-parent merges no longer block a retry; a new `tools/<name>.py` is
  overlaid only with `--allow-new-tool <path>` after name, base and importability checks. `prepare` now
  omits unrelated decision records from the packet (listed by name) and leaves no half attempt folder
  after a size refusal. AERIS-012 attempt 03 command:
  `prepare AERIS-012 --retry --allow-new-tool tools/audit_publication.py`.

- AERIS-035 QUT multi-sequence evaluation harness (`scripts/qut_eval.py`, `configs/qut/`,
  `docs/qut_eval.md`, ADR 032, PLAN-011), accepted 2026-10-22 on attempt 02 (attempt 01 revised for
  an unbound `compare` input and an ambiguous-frame count) after 360 tests and a fresh independent
  review. It replaced AERIS-032, whose contract put configs in `configs/evaluation/` (PLAN-011
  Addendum A). All 15 pinned QUT sequences (29.35 MB, Git-ignored), pre-registered dev/holdout split
  and selection rule; S1/S2/S4 frame alignment is inferred (±1 frame). Development baseline, fast
  profile: target detection recall 0.123, track recall 0.115, 26 target fragments; detection is the
  main limit on QUT. Approximate centroid labels: **no AP, HOTA/IDF1 or accuracy claim.**

- AERIS-033 manoeuvre-robust tracker options (`size_sigma_frac_sqrt_s`, `gate_mode`; defaults
  reproduce AERIS-009 bit for bit), 11 pre-registered `configs/tracker/manoeuvre-*.json`
  candidates and `--tracker-config` in the CLI, live and web paths, ADR 033, accepted 2026-10-22 on
  attempt 01 after 368 tests and fresh independent review. The PLAN-011 rule selected
  `manoeuvre-a3000-s0-full` (QUT dev fragments 26 -> 24). It passed a six-sequence holdout but
  **fails the full seven-sequence holdout** run in AERIS-034 (guards, mostly S2), so it stays opt-in. F-35 smoke clip, unlabelled: 55 -> 6 tracks. Size-aware
  candidates (8-9 fragments) failed the fixed-50 px guards on very close aircraft; a size-relative
  guard needs a planning change. **Default stays `baseline.json` until the owner decides.** The
  holdout's selection record (SHA-256 prefix 4b90d7) is ADR 033's text before its "## Holdout"
  heading (no trailing newline), excluding the superseded-outcome note added at AERIS-034 acceptance.
- AERIS-034 tracker-guided full-resolution crops (`python/aeris/detection/guided.py`, extend-only
  tracker pass, `--guided-crops` in CLI and web, live default off), ADR 034, accepted 2026-10-22 on
  attempt 01 after 390 tests and fresh independent review. **Negative result:** the pre-registered
  candidate fails the PLAN-011 rule on the QUT development split (crop-inclusive detection recall
  0.1168 vs frame-pass 0.1231; off-target 17 -> 44), so no holdout run and crops stay opt-in. F-35
  (unlabelled): 153 -> 156 tracked frames. The harness now drops and counts out-of-image target labels.

- TOOL-003 orchestrator lint fixes (ruff E731 in `tools/orchestrate.py`, E401 in
  `tests/test_orchestration.py`; PLAN-012), accepted 2026-10-22 on attempt 01 after independent
  review that also re-ran the checks with the pre-change orchestrator. With the direct
  `server.py` fix (68d285d), `ruff check .` is clean, unblocking CI.

- TOOL-004 CI installs the `desktop` extra and Qt system libraries for the required
  `tests/ui/test_ui.py` Qt tests (PLAN-013), accepted 2026-10-22 on attempt 01 after independent
  review. Evidence: run 39854490606 on `3a07518`, all jobs green, 391 tests, `skipped=9` (same as
  before), both Qt tests ok; `deploy-hosted` was fast-forwarded to that exact commit. The apt list
  may need updating if the `ubuntu-latest` image changes.

## Reviewer follow-ups (minor, not yet done)

| From | Follow-up | Owner task |
|---|---|---|
| 001 | Mapping-bearing records are now explicitly unhashable; closed in AERIS-003 | Closed |
| 001 | `validate_sequence` accepts mixed `decode_pts`/`inferred_fps` sources; AERIS-008 added a rejecting helper in the video pipeline, `validate_sequence` itself unchanged (contract change) | Helper done; contract change open |
| 002 N1 | Manifest validator checks key presence but not values: empty strings, empty `artifacts`/`sources`, non-bool flags, unknown status/evaluation names, `1e400` | AERIS-003 (when porting to `python/aeris/data/validate.py`) |
| 002 N2 | 9 of 35 validator-rule mutants survive the tests (fallback, role/status, width bins, duplicate IDs, `holdout_rule`) | AERIS-003 |
| 034 | Reviewer minors: harness `frames_bgr` index/pan not asserted or tested (verified aligned on S3); live crop ms not recorded; CLI guided mode leaves `benchmark.json` track stage near zero (note only in run_manifest); crop-recall score floor untested; commit candidate lists before runs in future tasks | follow-up |
| 033 | Reviewer minors (1, 2, 4 and the boundary test fixed in AERIS-034): empty/non-dict config silently runs baseline; string config values raise TypeError; position-gate boundary test; server resolves relative --tracker-config once; QUT fast-zoom run and common-subset forecast comparison missing from ADR 033; wording in docs/tracking.md | AERIS-034 (tracker/server/tests in scope) or follow-up |
| 002 N3 | FGVC recall-only smoke check needs a fixed detection budget (top-k or pre-fixed score threshold) or it rewards box spam | AERIS-005/006 |
| 006 | 1627560 class/box not asserted in a test; airplane scored per anchor instead of YOLOX argmax-class selection (undocumented); confirm weight license before release | AERIS-007+/release |
| 008 | p95 not asserted in tests; read-back dimension comparison unexercised; max_frames equal to clip length cannot detect extra PTS | later video task |
| 009 | Tentative hits are non-consecutive; velocity test overrides default accel sigma 200 with 2.0; equal-score simultaneous births get IDs by input order | AERIS-021 development split |
| 011 | Fixed 2026-10-05 after acceptance (invertibility check, `ray_to_pixel` None outside frame, SE n-2 test); post-acceptance change not under a formal receipt or fresh review | Closed (see commit) |
| 010 | Half-update on PSD failure and the chi-square 0.5 assertion fixed 2026-10-07 after acceptance (atomic update, quantile test); post-acceptance change without a fresh independent review. Still open: `eligible_truth` does not reject duplicate identity timestamps (unreachable via `evaluate_forecasts`) | Mostly closed |
| TOOL-001 | Direct `subprocess.run` calls in `rebase_candidates`/`_verify_explicit_base`/`merge-base` skip the `GIT_*` env scrub that `_git` applies; `_commit_files` raises UnicodeDecodeError (still fail-closed) on non-UTF-8 paths; no explicit coordinator override for accepted changes to denylisted paths or a task's own contract (would remove the last widening need) | follow-up tooling task |
| 007 | Regression-schema reference checker lets NaN threshold/tolerance, duplicate seeds, a draft with split hashes or freeze fields, and a hash with a trailing newline pass; `baseline_manifest_sha256` is not checked against the manifest; R6 `min_support` untested; docs wording "null-effect seed list" unclear | AERIS-015 enforcement or a contract-change task |
| TOOL-002 | Dependency-prefix ADR inclusion rule matches few AERIS-NNN ADRs (named `NNN-*.md`); unmatched/duplicate `--allow-new-tool` paths are recorded in `base.json` and only warned on stderr; NFD-committed names are neither overlaid nor refused; mutation script is outside the repo | follow-up tooling task |
| 002 N4 | Wording: the report says 18 object patterns (17 in fact); ADR 002 says session splitting is "the only rule" preventing leakage (overstated) | next ADR 002 touch |

- AERIS-023..028 replay reviews (all minor): CliError guard and per-pair spy assertions (023);
  `__main__` blocks placed before newer test classes in `tests/ui/test_web.py`,
  `tests/unit/test_trajectory.py` and `tests/unit/test_tracking.py` (running a file directly skips
  them; discovery-based checks are unaffected); tall-box sigma and frame-drop timing cases (024);
  ±180° wrap, float-sum tolerance, raw-detection reference and tiny predicted displacement (025);
  jittered stationary target, runnable sweep command, `velocity()` guards (026); live low-score path,
  detection-metric input, score == threshold boundary, ADR 027 smoke provenance (027); live
  `to_view` rotation under camera roll (028).

## Owner decisions blocking real video evidence (not G1)

Footage source and rights, capture sessions (at least 6), annotation tool, and approval
of the 24 h annotation/review budget. See `docs/dataset.md`. These also block the deferred
AERIS-021 (real-video tracking evidence) and AERIS-022 (real measured baseline and frozen
gates). AERIS-015 and 017 depend on 022, which blocks 016 and 018 transitively. AERIS-009's AC3
and AERIS-007's real-measurement criteria were split out by
`docs/decisions/PLAN-001-aeris-009-ac3-split.md` and `PLAN-003-aeris-007-split.md`.

Next action: AERIS-012 (integrated Python MVP) is ready; AERIS-022 and AERIS-021 stay deferred.
