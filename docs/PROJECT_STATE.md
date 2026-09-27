# Durable project state

As of 2026-09-27: no perception pipeline, trained model, target-domain benchmark
or published release exists. G1 (AERIS-001 to 003) and G2 (AERIS-004 to 007) are
accepted (see below). They are data contracts, audit and QA tooling, analytic synthetic
fixtures, pinned evaluators, and a pretrained COCO detector with a CPU ONNX path. The
first measured baseline is on public AOT aircraft video, a domain proxy (grayscale,
airborne camera) rather than the fixed RGB ground camera the MVP targets. G0 is the engineering
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
`aeris-runs-2026-09-25-g2a.tar.gz` adds AERIS-004, `aeris-runs-2026-09-25-g2b.tar.gz`
adds AERIS-005, `aeris-runs-2026-09-27-g2c.tar.gz` adds AERIS-006 and
`aeris-runs-2026-09-27-g2d.tar.gz` adds AERIS-007. Extract the newest copy at the repository root so
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

Reviews were run by separate subagent sessions launched by the coordinator, so their
independence is operational only. One reviewer session reviewed 001 and 002. A fresh
session reviewed both 003 attempts; the coordinator co-authored 003. Fresh sessions
reviewed 004, both 005 attempts, both 006 attempts and the three 007 attempts, which
the coordinator wrote. The first 006 attempt-2 reviewer session stopped on a rate limit
before writing anything; a fresh session did the review. Another fresh session reviewed
the AERIS-007 planning change.

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
| 006 A1 F6 | `AERIS_REQUIRE_REAL_DETECTOR=1` is not set by the detection gate, so REAL tests could skip and still pass verify. Propose a gate change (separate contract-change task) | next governance task |
| 006 A2 F1 | The two small aircraft were not simply missed: they were localized and labelled `kite` (B-17 score 0.608, IoU 0.78; airliner 0.018, IoU 0.52), then dropped by argmax-then-map. `docs/detection.md`, ADR 006 and the report call it a resolution-only miss. Correct the wording; weigh the class rule with measured data. AERIS-007 measured it on AOT: other COCO classes cover only 15 of 487 missed development aircraft | a later detector task (rule); next `docs/detection.md` touch (wording) |
| 006 A2 F2 | Two surviving mutants: a parity box check over x coordinates only passes all 54 strict tests; removing the fetcher's request pacing passes the sample tests. Add a y-only box perturbation and a pacing test | next task touching `aeris/detection` |
| 006 A2 F3 | Fetch record: the local `manifest.json` came from an uncommitted intermediate fetcher (no `original_size_bytes`), so the submitted fetcher has run only against fakes; the three HTTP 429 refusals survive only in prose; five stored files are 3000–3840 px wide, not "2048 px renditions or smaller originals"; COCO's small-object boundary is by area (< 32² px²), not width. Re-fetch with the submitted fetcher when useful and correct the wording | next task touching `aeris/detection` |
| 006 docs | ADR 005/006 still read "proposed", and `docs/detection.md` "attempt 2, awaiting independent review". Set them to accepted | next docs touch |
| 006 other | Small-target detection below about 32 px input width; migrate the export from the deprecated TorchScript exporter to `torch.export` (re-run parity, record the new hash) | AERIS-007 / later detector task |
| 007 A3 F10 | `docs/baseline.md` gate table: the basis for false positives per negative frame should read "one false positive on a negative frame", not "one frame" | next `docs/baseline.md` touch |
| 007 gate | The baseline check does not set `AERIS_REQUIRE_REAL_BASELINE=1`, so REAL tests could skip and still pass (same as 006 F6). Include it in the gate change | next governance task |
| 007 review format | Attempt 2 was closed because review findings had object `detail`s. `orchestration/prompts/reviewer.md` (and the generated reviewer.md) should say `detail` is a string | next governance task |
| 007 tests | The REAL tests compare the stored selection with its digests but do not re-derive it from the ground-truth CSVs (about 80 s). Reviewers did it manually | AERIS-015 (regression gates) |
| 007 precision gate | Precision's one-count tolerance lets two added false positives pass at current counts. False positives on frames with aircraft are caught only through precision. Consider a false-positives-per-complete-frame gate | AERIS-015 |
| 007 domain | AOT is a domain proxy. A fixed-camera RGB benchmark (owner pilot, or TartanAviation if reachable: its servers use ports 8080/8081) is needed for target claims | owner decision / AERIS-018 |
| 007 small targets | Recall is 0 below 16 px and 0.7 % at 16–32 px on AOT, and point-like targets are unscored. Crowd matching can charge loose boxes on markers | AERIS-017 |
| 004 F4 | Synthetic detector scores identify row kinds (duplicate exactly 0.55, false positive 0.30), so a threshold between 0.55 and 0.70 silently drops both. AERIS-005's tests probe thresholds 0.25/0.5/0.55/0.9 around them; vary the scores before tracker threshold tests | AERIS-009 or next `aeris/simulation` task |

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
(GitHub Actions, Python 3.11/3.12 and C++) passed on the accepted AERIS-007 source
(`751b60a`). CI installs the pinned evaluators and ONNX Runtime through the `dev` extra;
it has no torch or weights, so the REAL detector tests skip there. The owner's newest
run archive must include AERIS-007 (`aeris-runs-2026-09-27-g2d.tar.gz`, which also holds
the AERIS-007 run outputs and v1 archive under `benchmarks/runs/`, but not the AOT frames).

Next action: in a fresh checkout, first restore `orchestration/runs/` (and
`benchmarks/runs/`) from the newest archive; both are Git-ignored. AERIS-008
(timestamp-aware video pipeline, G3) is ready. Its real sample is to be found online, as
the owner asked on 2026-09-27. Candidate: Commons `File:Landing aircraft - Haneda -
2021 3 31.webm` (CC BY-SA 4.0, 13 s, 1920x1080, 3,381,887 bytes, SHA-1 `c201872a…`).
OpenCV/FFmpeg reads 392 frames with container timestamps (33–34 ms steps). The camera
pans. AERIS-009 (tracking) follows AERIS-008. Real identity-labelled video for MOT exists
in AOT (track IDs) as a domain proxy.
