# Durable project state

As of 2026-09-27: no tracking, forecasting, trained model, target-domain benchmark
or published release exists. G1 (AERIS-001 to 003) and G2 (AERIS-004 to 007) are
accepted, and so is AERIS-008, the first G3 task (see below). They are data contracts,
audit and QA tooling, analytic synthetic fixtures, pinned evaluators, a pretrained COCO
detector with a CPU ONNX path, and a timestamp-aware video pipeline that runs it on
recorded video. The first measured baseline is on public AOT aircraft video, a domain
proxy (grayscale, airborne camera) rather than the fixed RGB ground camera the MVP
targets. G0 is the engineering scaffold, not a detection milestone.

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
`aeris-runs-2026-09-25-g2a.tar.gz` adds AERIS-004, `aeris-runs-2026-09-25-g2b.tar.gz`
adds AERIS-005, `aeris-runs-2026-09-27-g2c.tar.gz` adds AERIS-006 and
`aeris-runs-2026-09-27-g2d.tar.gz` adds AERIS-007 and `aeris-runs-2026-09-27-g3a.tar.gz`
adds AERIS-008. Extract the newest copy at the repository root so
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

## Accepted (2026-09-27)

- AERIS-006 pretrained detector and export spike (`python/aeris/detection/`,
  `configs/detector/`, `docs/detection.md`, ADR 006). Review: accept on attempt 2 of 3,
  3/3/3/3/4.
  - YOLOX-S (Apache-2.0, commit `6ddff482`, release `0.1.1rc0` weights) as a pretrained
    COCO detector, with COCO `airplane` mapped to AERIS class 0 by argmax.
  - Upstream preprocessing and NMS reproduced in NumPy, with an exact inverse into
    original-frame pixels. ONNX Runtime 1.30.0, CPU provider only.
  - The AERIS opset-17 export is byte-reproducible and passes parity against the
    PyTorch reference on 13 frames (10 licensed Commons/upstream photos, 8 with
    aircraft), under tolerances declared before measurement: 277/277 detections matched.
  - AC1: every photo marked required by a rule committed before detection
    (`tests/fixtures/detection/aircraft_regions.json`) gives an AERIS class 0 record
    on the aircraft (IoU 0.88 to 0.97). Two small aircraft (about 29 px and 13 px wide
    in the 640 px input) give no aircraft record; birds and dog photos give none.
  - Owner decisions (2026-09-25/26): YOLOX; Commons photos; PyPI torch without triton,
    because `download.pytorch.org` is denied; a 5 GB budget (2.99 GB downloaded and
    4.97 GB installed, measured).

  Attempt 1 was escalated for one major finding: no real aircraft photo was available,
  so AC1 could not be shown. Attempt 2 fetched licence-checked Commons photos once the
  owner allowed the hosts, and fixed the six minor findings. Weights, exports and
  photos are not in Git; their hashes and retrieval steps are.

- AERIS-007 measured baseline and development gate (`python/aeris/data/aot.py`,
  `python/aeris/evaluation/baseline.py`, `configs/evaluation/`,
  `benchmarks/baseline_manifest.json`, `docs/baseline.md`, ADR 007). Review: accept on
  attempt 3 of 3, 4/3/4/4/4.
  - **Data.** Airborne Object Tracking (AOT, CDLA-Permissive-1.0, public S3), chosen
    after the owner asked for labelled data found online. Nothing is committed; frames
    are pinned by S3 ETag and SHA-256.
  - **Benchmark `aeris-007-aot-v2`.**
    - Capture-day splits: development 31 days / 1,102 frames, validation 15 / 767,
      and a sealed holdout of 15 days / 1,023 frames that was not downloaded.
    - Nominal 6/16 px marker boxes (81 % of AOT aircraft labels) are ignore regions.
  - **Development results** at the validation-chosen threshold 0.16:
    - AP50 0.067, AP 0.028;
    - precision 0.376, recall 0.083 (44 of 531);
    - 0.067 false positives per negative frame (373 frames);
    - recall by width 0 / 0.007 / 0.069 / 0.451.
  - **Gate.** 1-thread and 4-thread runs were bit-identical, so
    `configs/evaluation/regression.json` uses the declared resolution floors (0.001 AP,
    one count).
  - **History.**
    - Attempt 1 was sent back for two major bugs: float-noise markers were scored as
      aircraft, and the gate rounding failed one-count-worse candidates. The fix became
      benchmark v2, disclosed as a revision made after the v1 results were seen. v1
      outputs are archived.
    - Attempt 2 was accepted by its reviewer but closed by the tool, because the
      review's finding details were objects rather than strings.
    - Attempt 3 fixed three wording and record minors.

- AERIS-008 timestamp-aware video pipeline (`python/aeris/video/`, `docs/video.md`,
  ADR 008). Review: accept on attempt 3 of 3, 3/3/4/3/4.
  - **Decoding.** OpenCV's FFmpeg backend (LGPL, already a dependency). PyAV was
    rejected because its wheel bundles GPL x264/x265 and no project licence is chosen.
  - **Timestamps.** `decode_pts` only from containers that store a time for every
    frame:
    - Matroska/WebM: every decoded frame is checked against block times AERIS reads
      itself, and laced, invisible or multi-video-track files are refused;
    - MP4/QuickTime: sample tables trusted through FFmpeg, not re-parsed.

    MPEG-TS, AVI, Ogg and raw streams need the explicit `inferred_fps` mode. Zero or
    negative deltas fail on every decoded frame, including stride-skipped ones.
  - **Timings.** Per-frame decode, preprocess, inference, postprocess, render, `other`
    and a whole-frame `total`. Throughput comes from the loop's wall time. Skipped
    frames are counted by reason.
  - **Real sample.** Commons "Landing aircraft - Haneda - 2021 3 31.webm" (CC BY-SA 4.0,
    3.4 MB, 392 frames, 1920 x 1080), pinned by SHA-1 and SHA-256 and not committed.
    Full clip with the ONNX detector on the shared VM: total P50 277.3 ms, 3.56
    processed frames/s. This is a record, not a performance claim.
  - **History.**
    - Attempt 1 was sent back for two major findings. A raw MJPEG stream got
      FFmpeg-invented times labelled `decode_pts`. And `total` left out the pixel
      hash, so throughput was overstated.
    - Attempt 2 was sent back for one major finding: MPEG-TS and laced Matroska still
      let FFmpeg fill in times that are not stored.
    - Attempt 3 refused those and added the frame-by-frame Matroska check.

- AERIS-021 overlay render geometry correction (`python/aeris/video/pipeline.py`,
  ADR 021), a follow-up task for Codex's supplementary AERIS-008 review finding
  **C008-1** (major; see below). Review: accept on attempt 1 of 3, 4/4/3/3/4.
  - `Overlay.__init__` refuses (`PipelineError`, no file written) an odd input width or
    height, rather than let OpenCV's `mp4v` writer silently crop it while `run.json`
    kept reporting the requested (wrong) size. Headless processing is unaffected.
  - Measured directly against this Linux build across five fourccs (`mp4v`, `MJPG`,
    `FFV1`, `HFYU`, `RGBA`) and three containers: every one rounds an odd dimension
    down, so this is general OpenCV `VideoWriter` behaviour here, not `mp4v`-specific.
  - The MP4 overlay regression test now decodes the written file and checks its real
    geometry against `run.json.overlay`, closing the general form of the defect, not
    only the odd/even case; a dedicated test covers odd width, odd height and both odd
    through the fake-capture seam (OpenCV's own writer cannot emit a genuinely
    odd-dimension file in this sandbox, which also has no `ffmpeg` CLI).
  - AERIS-008 itself was not reopened: its three attempts and acceptance are untouched.
  - The reviewer's own probes (6 hand-written mutants, an independent five-fourcc
    rounding re-measurement, and a `git diff` confirming AERIS-008's files were
    untouched) corroborate the fix; findings below.

- AERIS-009 Kalman tracking and association (`python/aeris/tracking/tracker.py`,
  ADR 009). Review: accept on attempt 2 of 3, 4/4/3/4/4.
  - **State and association.** `[cx,cy,vx,vy,w,h]` constant-velocity Kalman state;
    `[cx,cy,w,h]` measurement (size is matched into the gate, not left out — it is what
    keeps a crossing's two tracks apart when their predicted centres coincide). Gated
    Hungarian assignment runs in two passes: already-confirmed tracks match every
    detection first, then tentative tracks compete for what is left.
  - **Why the cascade.** Found empirically on the AERIS-004 crossing fixture's own
    duplicate-detection case: a single flat assignment pass lets a track just spawned
    from a duplicate (wide, barely-constrained covariance) win an established track's
    next detection on Mahalanobis cost alone, handing over its identity. The two-pass
    cascade (DeepSORT's own rationale) fixes this by construction; a within-frame
    IoU-NMS duplicate suppression step is kept too, as an independent defence, even
    though it alone did not suppress this fixture's own jittered duplicate (measured
    IoU 0.44, just under its 0.5 threshold).
  - **Lifecycle** (`tentative -> confirmed -> lost -> deleted`) is timed in elapsed
    seconds throughout, never frame count; process noise `Q(dt)` depends on elapsed
    time; covariance updates use Joseph's form, symmetrized and eigenvalue-clipped
    every step.
  - **Real evidence (AC3):** 23 pinned, contiguous real AOT frames (one flight, a
    scored Helicopter measuring 15.6-19.7 px wide throughout), reporting two separate
    real results on the same flight: an oracle ablation (ground truth fed directly to
    the tracker, clearly labelled as such) gives 0 identity switches, MOTA 0.957, IDF1
    0.978; the real end-to-end AERIS-006 ONNX detector finds nothing (0 detections),
    the honest result given AERIS-007's already-measured near-zero recall below 32 px.
  - **History.**
    - Attempt 1 was sent back for one major finding: the real target's measured size
      was wrongly given as "16-45 px" in the report, docs and ADR; the true figure in
      the pinned 23-frame window is 15.6-19.7 px (the 45 px came from the same flight,
      211 frames outside the pinned evidence). Two minor test-coverage gaps were also
      raised.
    - Attempt 2 corrected the wording everywhere and closed both minor gaps: a test
      that `predict()`/`update()` actually call the PSD-safety helper (not just the
      helper alone), and a test of the exact `lost_timeout_s` deletion boundary
      (strictly greater-than).
  - During this task, `verify`'s full unit-suite check (327 tests, 111-119 s) actually
    timed out once against its 120 s budget before passing on an immediate retry — the
    governance gap already flagged for 006/007/008 stopped being theoretical.

- AERIS-022 unit-check timeout margin (`orchestration/gates.json`, ADR 022), a
  governance follow-up dispatched mid-AERIS-010 specifically because AERIS-010's
  `verify` could not otherwise reach a passing `unit` check. Review: accept on attempt 2
  of 3, 3/3/3/4/4.
  - **Why.** AERIS-010's `verify` timed out on the `unit` check three consecutive times
    (exit 124) against the old 120 s budget while every other check, including its own
    task-specific one, passed cleanly — the "008/009 unit timeout" follow-up escalated
    from an occasional near-miss to a consistent, correctness-unrelated blocker.
  - **Fix.** Raises `orchestration/gates.json`'s `unit` check `timeout_s` to **900**
    (attempt 1 proposed 240; see History). Data-only change: no other check's
    `argv`/`timeout_s`/existence changed, and `tools/orchestrate.py` itself (the gate
    tool) was not touched, per EVALUATION_LOOP.md's guidance that gate-tool source
    changes are outside normal task scope.
  - **History.**
    - Attempt 1 proposed 240 s, derived from two quiet-VM measurements (131.238 s and
      121.071 s wall). Its independent reviewer sent it back on a major finding: they
      independently re-ran the identical, unmodified gate command and reproduced a
      **contended** run at 528.697 s internal / 563.658 s wall (load average up to 3.98
      on 4 vCPUs, right after their own review container started), over 2.2x the
      proposed 240 s, with a quiet run minutes later matching the original figures. This
      falsified attempt 1's claim that 240 s covered realistic shared-VM variance.
    - Attempt 2 re-derived the margin against that reproduced worst case instead of the
      quiet baseline (900 s ≈ 1.6x the 563.658 s figure) and added an explicit
      "Residual risk" section to ADR 022: 900 s covers every condition reproduced so
      far, not a guarantee against worse contention, with a named escalation path
      (gather the new data point; split the suite) if it recurs. Its reviewer
      independently re-checked the evidence-table figures against attempt 1's own
      review verbatim, confirmed the arithmetic (900/563.658 ≈ 1.597), and confirmed via
      `git diff`/`git show` that only that one value changed.
  - **Open, non-blocking.** The margin rests on one contended sample, not several; a
    genuine test hang now takes up to 15 minutes to surface via `verify` instead of 2
    (an accepted, disclosed tradeoff). Does not address the separate, larger
    `AERIS_REQUIRE_REAL_*`-not-set-by-gates follow-up (needs a `tools/orchestrate.py`
    source change, out of this task's scope).

- AERIS-010 causal image-space forecasts (`python/aeris/estimation/trajectory.py`,
  ADR 010). Review: accept on attempt 2 of 3, 4/4/4/4/4. Attempt 2's code is identical
  to attempt 1 — attempt 1 was closed only by an unrelated `unit`-check timeout (see
  AERIS-022 above), not a code revision.
  - **Two independently specified causal Kalman models**, both consuming truth
    `AnnotationRecord` observations directly under AERIS-005's already-accepted
    oracle-identity `evaluate_forecasts` (not modified here): `constant_velocity`
    (`[cx,vx,cy,vy]`, discrete white-noise-acceleration process noise — the same
    physical assumption as `aeris.tracking.tracker`'s own centre model, independently
    implemented) and `constant_acceleration` (`[cx,vx,ax,cy,vy,ay]`, continuous
    white-noise-jerk process noise integrated exactly through its own transition —
    reviewer-verified from first principles with `sympy` against the module's closed
    forms, confirming these are genuinely different noise models, not the same model
    integrated one order further).
  - **Causal warm-up, not a two-frame difference.** Velocity needs 2 observations,
    acceleration needs 3, before a forecast's centre/covariance are anything but null
    with a reason — the reviewer confirmed the off-by-one boundary by direct code trace
    and found no path that leaks a future observation into an earlier forecast.
  - **Analytic evidence (AC2).** AERIS-004's own `sim-cv-crossing` (read-only reuse)
    cross-checks the velocity model to within 0.027 px across 75 points; this task's own
    new `tests/fixtures/trajectory/constant_acceleration.jsonl` (dyadic frame times,
    exact in IEEE-754 double precision) cross-checks the acceleration model to within
    0.015 px after warm-up and shows the velocity model diverging by ~88x on the same
    genuinely accelerating target as the horizon grows. The reviewer independently
    recomputed all 40 fixture frames from the closed form (exact match) and re-derived
    both process-noise matrices symbolically (exact match), rather than trusting either
    claim or the test file's own quadrature check.
  - **Comparable evaluation (AC3).** Both models score `eligible_total=71` with an
    identical `eligible_set_digest` on the real evidence (23 pinned AOT frames, the same
    flight AERIS-009 validated its tracker against) — true by construction, since
    `evaluate_forecasts`' eligibility depends on truth alone. The acceleration model
    does not outperform the velocity model on this one short (2.2 s) real flight; this
    is reported plainly, not curated, and the reviewer reproduced the full per-horizon
    RMSE table exactly.
  - **Open, non-blocking (from review).** `docs/trajectory_estimation.md` and ADR 010
    still read "attempt 1" in their status lines (cosmetic; see follow-up table). The
    coordinator's own report for this task narrated attempt 1's closure as "three
    consecutive unit-check timeouts"; the reviewer traced attempt-1's actual
    `decision.json` and found the literally recorded `problems` entry was "changed_files
    must exactly match source changes" (a transient, since-corrected `orchestration/gates.json`
    working-tree edit made while triaging the timeout, before AERIS-022 was properly
    spun out) — the underlying timeout fact itself is independently corroborated by
    attempt-1's own `verification.json` (`exit_code 124`) and by AERIS-022's separately
    accepted review, and attempt 2's own commit touches only its 6 declared files, so
    this does not implicate attempt 2's implementation, only an imprecision in the
    closure narrative.

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only. One reviewer session reviewed 001 and 002. A fresh
session reviewed both 003 attempts; the coordinator co-authored 003. Fresh sessions
reviewed 004, both 005 attempts, both 006 attempts, the three 007 attempts, the three
008 attempts, AERIS-021, both 009 attempts, both 010 attempts (identical code; attempt 1
closed only for the AERIS-022 timeout reason) and both 022 attempts, which the
coordinator wrote. The first reviewer sessions for 006 attempt 2, 008 attempt 1 and 022 attempt 1
stopped early before writing anything (rate limit for the first two; the 022 attempt-1
session's own report arrived after its review was already independently completed and
written, so nothing was lost); fresh or continued sessions completed those reviews.
Another fresh session reviewed the AERIS-007 planning change. 022 attempt 1's reviewer
independently reproduced a real VM-contention timing sample that the coordinator's own
proposal had not accounted for, sending that attempt back; this is the clearest instance
so far of an independent review changing the outcome through its own reproduction, not
just checking the author's claims.

Supplementary review (2026-09-27): the owner authorised a parallel Codex session to
review read-only on PR #1. Codex's local AERIS-003 attempt-2 review accepts, with R1–R5
and N1/N2 resolved, 20 fresh dataset tests and adversarial probes. It is bound to
source fingerprint `f16ec29c…`, the original attempt-2 source, not the accepted
`265da4cb…`, so it corroborates the original attempt only. Its review file stays in
Codex's local checkout, and the canonical review is unchanged. Codex also notes that the
FGVC cabin image `image1885942` is unsuitable for exterior detection. Codex acknowledged
reviewing AERIS-008 attempt 1 (`1f73d59`) but had posted no findings when AERIS-008 was
accepted.

Codex's supplementary macOS review of the final AERIS-008 source (`b00c99c`, PR #1
comment `5859280428`, 2026-09-27) reran the video suite there (33/34 pass, one skip:
no local ONNX export) and independently re-verified the Matroska stored-time check
against `ffprobe` (max difference `1.82e-12` ms) and the fetched clip's hashes. It
raised one new finding, **C008-1** (major), on the `--render` overlay path, not on
detection or timestamp correctness. AERIS-008 stayed accepted throughout (three
attempts already used); the finding was fixed as the separate follow-up task AERIS-021
(above), closed in the follow-up table below.

## Reviewer follow-ups (not yet done)

| From | Follow-up | Owner task |
|---|---|---|
| 001 F1 | 17 of 34 contract mutants survive. The untested checks are ID, version and timestamp checks on detection, prediction, truth and annotation records; there are no `annotation_id`/`track_id` identifier tests. The reconstructed 001 report (F2) wrongly claimed them | next contracts task (unassigned) |
| 001 F4 | `validate_sequence` accepts mixed `decode_pts`/`inferred_fps` sources in one sequence; decide reject or report. AERIS-008 avoids it (one timestamp policy per run) but could not change `python/aeris/data/` | next task owning `python/aeris/data/*` (unassigned) |
| 002 N1/F1 | Manifest validator value checks: 27 of 40 invalid values pass (no sources, `False` class ID, `1e400`, unknown evaluation names). Not ported to `python/aeris/data` by AERIS-003 | unassigned |
| 002 N2/F2 | 11 of 38 validator-rule mutants survive | unassigned |
| 002 N3/F3 | FGVC recall-only smoke check needs a fixed detection budget. AERIS-005's evaluator now supports a pre-fixed inclusive score threshold and a per-frame top-k budget; the actual values still have to be fixed before any FGVC run | AERIS-006/007 |
| 002 N4/F4 | Wording: ADR 002 overstates session splitting as "the only rule" preventing leakage; the original report said 18 object patterns (17) | next ADR 002 touch |
| 003 A2 F1 | A `true` flag on an attribute listed in `unannotated_attributes` is accepted silently. With `ignore` listed, the statistics reason then wrongly says the width bins count every object. Reject or report such flags | next task owning `python/aeris/data/*` (unassigned; outside AERIS-005's scope) |
| 003 A2 F2 | Two mutants survive. A sheet that never prints NEGATIVE passes `test_sheet_captions_and_attribute_tags` (the attempt-2 report overstated that test). No test lists `ignore` as unannotated | next task owning `python/aeris/data/*` (unassigned) |
| 004 F2 | Some generated fields are protected only by the byte snapshot, not by semantic assertions: vertical angular extent, range, range rate, detection class ID, false-positive score, noise model and manifest record counts. The reviewer found 7 such mutants. The coordinator's 25-mutant run is now saved in the evidence folder | next task touching `aeris/simulation` |
| 004 F3 | Truth angular extent is measured between marker rays. CONTRACTS.md defines the bbox extent between edge rays (≤ 4.6e-6 relative difference here). Document it in ADR 004 or add edge-ray truth | AERIS-011 |
| 006 A1 F6 | `AERIS_REQUIRE_REAL_DETECTOR=1` is not set by the detection gate, so REAL tests could skip and still pass verify. Propose a gate change (separate contract-change task) | next governance task |
| 006 A2 F1 | The two small aircraft were not simply missed: they were localized and labelled `kite` (B-17 score 0.608, IoU 0.78; airliner 0.018, IoU 0.52), then dropped by argmax-then-map. `docs/detection.md`, ADR 006 and the report call it a resolution-only miss. Correct the wording; weigh the class rule with measured data. AERIS-007 measured it on AOT: other COCO classes cover only 15 of 487 missed development aircraft | a later detector task (rule); next `docs/detection.md` touch (wording) |
| 006 A2 F2 | Two surviving mutants: a parity box check over x coordinates only passes all 54 strict tests; removing the fetcher's request pacing passes the sample tests. Add a y-only box perturbation and a pacing test | next task touching `aeris/detection` |
| 006 A2 F3 | Fetch record: the local `manifest.json` came from an uncommitted intermediate fetcher (no `original_size_bytes`), so the submitted fetcher has run only against fakes; the three HTTP 429 refusals survive only in prose; five stored files are 3000–3840 px wide, not "2048 px renditions or smaller originals"; COCO's small-object boundary is by area (< 32² px²), not width. Re-fetch with the submitted fetcher when useful and correct the wording | next task touching `aeris/detection` |
| 006/008/009/010 docs | ADR 005/006/008/009/010 still read "proposed", `docs/detection.md` "attempt 2, awaiting independent review", `docs/video.md` "attempt 3, awaiting independent review", `docs/tracking.md` "attempt 1, awaiting independent review" (accepted on attempt 2) and `docs/trajectory_estimation.md`/ADR 010 "attempt 1" (accepted on attempt 2). Set them to accepted | next docs touch |
| 006 other | Small-target detection below about 32 px input width; migrate the export from the deprecated TorchScript exporter to `torch.export` (re-run parity, record the new hash) | AERIS-007 / later detector task |
| 007 A3 F10 | `docs/baseline.md` gate table: the basis for false positives per negative frame should read "one false positive on a negative frame", not "one frame" | next `docs/baseline.md` touch |
| 007 gate | The baseline check does not set `AERIS_REQUIRE_REAL_BASELINE=1`, so REAL tests could skip and still pass (same as 006 F6). Include it in the gate change | next governance task |
| 007 review format | Attempt 2 was closed because review findings had object `detail`s. `orchestration/prompts/reviewer.md` (and the generated reviewer.md) should say `detail` is a string | next governance task |
| 007 tests | The REAL tests compare the stored selection with its digests but do not re-derive it from the ground-truth CSVs (about 80 s). Reviewers did it manually | AERIS-015 (regression gates) |
| 007 precision gate | Precision's one-count tolerance lets two added false positives pass at current counts. False positives on frames with aircraft are caught only through precision. Consider a false-positives-per-complete-frame gate | AERIS-015 |
| 007 domain | AOT is a domain proxy. A fixed-camera RGB benchmark (owner pilot, or TartanAviation if reachable: its servers use ports 8080/8081) is needed for target claims | owner decision / AERIS-018 |
| 007 small targets | Recall is 0 below 16 px and 0.7 % at 16–32 px on AOT, and point-like targets are unscored. Crowd matching can charge loose boxes on markers | AERIS-017 |
| 008 A3-R1 | Surviving test mutants: the stored-time check moved after the stride skip, or run only when `every == 1`; tolerance loosened to 1.5 ms; the lacing refusal applied to every track (would refuse laced audio); the outer truncation handler made a refusal; stored times not made relative to the first. In the REAL ONNX test the P50 finiteness check now covers only `total` (a leaked loop variable). Add an `every=4` backward case, a 1 ms mismatch, laced audio, a truncated block header, a non-zero first block time; move the P50 check back into the stage loop | next task touching `aeris/video` |
| 008 A3-R2 | `matroska_timeline` ends an unknown-size cluster at Void (`0xEC`) or CRC-32 (`0xBF`), which may legally appear inside a cluster. Such a file is refused with a misleading "differs from the stored block time" error (fails safe; nothing mislabelled). Treat them as skippable cluster children; `docs/video.md` wording | next task touching `aeris/video` |
| 008 A3-R3 | A crafted MPEG-TS behind an 8- or 16-byte MP4 `skip` box is classed as MP4; FFmpeg opens it as TS and interpolated times are labelled `decode_pts`. No muxer writes such a file, but the `docs/video.md` limitation ("caught only by the open, decode or delta checks") is too optimistic. Walk the top-level ISO BMFF boxes and require `moov`, or reword | next task touching `aeris/video` |
| 008 MP4 | MP4 times are FFmpeg's reading of `stts`/`ctts`, edit lists and fragment defaults; no real MP4 file was checked, only OpenCV-written constant-rate ones | next task touching `aeris/video` |
| 009 gate | The `tracking` check does not set `AERIS_REQUIRE_REAL_TRACKING=1`, so its REAL tests could skip and still pass verify (same as 006/007/008); `verify` was run with it set explicitly | next governance task (roll into the same gate change as 006/007/008) |
| 010 gate | The `trajectory` check does not set `AERIS_REQUIRE_REAL_FORECAST=1`, so its REAL tests could skip and still pass verify (same as 006/007/008/009); `verify` was run with it set explicitly | next governance task (roll into the same gate change) |
| 022 residual | The 900 s `unit` timeout is derived from one reproduced contended sample (563.658 s wall), not several; ADR 022 already names the exact next step (gather the new data point; split the suite under materially worse contention) if a future `verify` times out on `unit` again | next occurrence's task, per ADR 022's own escalation rule |
| 021 R021-1 | The AERIS-021 report, ADR 021 and this file's AERIS-021 entry attribute the OpenCV `VideoWriter` odd-dimension rounding to "4:2:0 chroma subsampling". The reviewer's own re-measurement found `RGBA` (uncompressed, no chroma subsampling) rounds identically, contradicting that specific mechanism. Doesn't affect the fix (the even/odd check is mechanism-agnostic and empirically justified); soften the wording to "measured, mechanism not confirmed" | next `aeris/video` docs touch |
| 021 R021-2 | The AERIS-021 `report.json` summary says "36 video tests pass in strict mode"; `check-video.log` and the reviewer's own rerun both show 35. Harmless narrative off-by-one, no verification artifact affected | next report touch (informational; no action needed beyond noting it here) |
| 004 F4 | Synthetic detector scores identify row kinds (duplicate exactly 0.55, false positive 0.30), so a threshold between 0.55 and 0.70 silently drops both. AERIS-005's tests probe thresholds 0.25/0.5/0.55/0.9 around them; AERIS-009's own tracker tests use fixed 0.9/0.5 scores and did not address this; vary the scores before tracker threshold tests | next `aeris/simulation` task |

Closed by AERIS-022:

- **008/009 unit timeout**: the shared `unit` check's timeout was 120 s; AERIS-010 hit
  three consecutive timeouts against it, escalating this from an occasional near-miss
  (one timeout during AERIS-009, cleared on immediate retry) to a consistent,
  correctness-unrelated blocker. Raised to 900 s on attempt 2, after attempt 1's own
  reviewer reproduced a 563.658 s-wall contended run that would still have exceeded
  attempt 1's proposed 240 s.

Closed by AERIS-021:

- **008 C008-1** (major, from Codex's supplementary AERIS-008 review, PR #1 comment
  `5859280428`): `--render`'s overlay silently reported the requested frame size while
  OpenCV's `VideoWriter` had rounded an odd width or height down. `Overlay.__init__` now
  refuses before writing anything; headless processing is unaffected. AERIS-008 itself
  was not reopened.

Closed by AERIS-003:

- Records holding `availability_reason` are explicitly unhashable (001 follow-up).
- `contracts.loads` reports RecursionError, oversized-integer and undecodable input as
  `ContractError`, so the loader reports such lines instead of crashing (001 F5).
- All six attempt-1 review findings: statistics nulls, stronger tests, unverified claims
  qualified, CLI errors, per-line decoding, and FGVC split subsets.

## Planning changes

- 2026-09-27, AERIS-007 scope (before its first packet). The owner asked for public
  labelled data found online, with more budget if needed. The chosen source, the Airborne
  Object Tracking dataset (AOT, CDLA-Permissive-1.0, public S3), needs a source adapter
  and a baseline runner. AERIS-007 owned no Python module, and `python/aeris/data/*` and
  `python/aeris/evaluation/*` belong to accepted tasks. Following the AERIS-013/015/017
  pattern of owning single files inside an earlier module, AERIS-007 now also owns
  `python/aeris/data/aot.py`, `python/aeris/evaluation/baseline.py` and
  `tests/fixtures/baseline/*`, and reads `docs/dataset.md` and `docs/evaluation.md`.
  Acceptance criteria, checks, required artifacts and budget text are unchanged. A packet
  prepared minutes earlier under the old scope was discarded unused. TartanAviation
  (fixed airport cameras, RGB, the closer domain) was preferred but its servers use
  ports 8080/8081, which this environment's proxy cannot reach.

## Owner decisions blocking real video evidence (not G1)

Footage source and rights, capture sessions (at least 6), annotation tool, and approval
of the 24 h annotation/review budget. See `docs/dataset.md`.

Integration: branch `claude/inspiring-keller-xf368q`, draft PR Vinvin00/AERIS#1. CI
(GitHub Actions, Python 3.11/3.12 and C++) passed on the accepted AERIS-008 source
(`a8dbb2c`). CI installs the pinned evaluators and ONNX Runtime through the `dev` extra.
It has no torch, weights, AOT frames or video sample, so the REAL tests skip there. The
owner's newest run archive must include AERIS-008 (`aeris-runs-2026-09-27-g3a.tar.gz`).
It also holds the AERIS-007 run outputs and v1 archive under `benchmarks/runs/`, and the
AERIS-008 full-clip evidence without its overlay video. It has neither the AOT frames
nor the Commons clip.

Next action: in a fresh checkout, first restore `orchestration/runs/` (and
`benchmarks/runs/`) from the newest archive; both are Git-ignored. Re-fetch the video
sample with `python -m aeris.video.samples fetch` for the REAL video tests, and the
pinned real AOT tracking frames with `python -m aeris.tracking.real_sample fetch`.
AERIS-010 (causal forecasts) is accepted (see above). `status` now shows AERIS-011
(Codex's exclusive scope; the cloud session does not touch it) and AERIS-017 (bounded
tiny-target experiment) ready; AERIS-012 (integrated Python MVP) still waits on 011.
AERIS-017 is the next task this session would take up.

**AERIS-011 ownership (2026-09-27):** Vincenzo asked Codex to implement AERIS-011
(calibrated camera-relative geometry, depending on 001 and 004) itself, on an isolated
`codex/aeris-011-camera-geometry` branch based on `a933706`, scoped to the camera
module, its tests/fixtures/configs, `camera_model.md` and ADR 011 only (PR #1 comments
`5859678593`, `5859701154`). The cloud session does not `prepare AERIS-011` and will not
touch those files while this stands. Canonical acceptance is unchanged: once Codex hands
over a patch and its own review evidence, it still goes through this session's `verify`
and a fresh independent reviewer on the integrated checkout before acceptance, per
AGENTS.md ("the author cannot review their own result"; a separate agent's review is
input, not a substitute). The cloud session continued with AERIS-009, then AERIS-010 and
AERIS-022, in the meantime.

**Coordination claim (2026-09-28):** a PR #1 comment posted from the `Vinvin00` account,
in a Codex session's own first-person voice, states "Vincenzo asked Codex to see where
Claude stopped and take over the work continuously" and asks this session to leave all
future task preparation, implementation and canonical acceptance to Codex. Per AGENTS.md
("the coordinator alone accepts tasks"; "identity strings in the tool are not
authentication") and this session's own established handling of the narrower AERIS-011
handoff claim, a report of what an owner asked, relayed by another AI agent, is not by
itself sufficient authorization for a handoff this broad. The cloud session posted a
factual status comment (PR #1) and continued the AERIS-010/AERIS-022 work already in
flight rather than stopping or ceding coordination unilaterally; it continues to welcome
Vincenzo's own direct instruction on how coordination should be split going forward.
