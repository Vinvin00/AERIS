# PLAN-010 — Replay-based formal verification of AERIS-023 through AERIS-028

Status: executed; AERIS-023..028 accepted 2026-10-17. Planning review r1 APPROVE WITH CHANGES
(13 findings, tagged [F1]..[F13]); r2 APPROVE WITH CHANGES (six minor findings applied in r3,
tagged [R1]..[R6]); the r2 reviewer ratified the [F4] contract edits. Drafted as PLAN-008, then
PLAN-009; both numbers were taken by other sessions' plans before this file could be committed
(the shared checkout was owned by AERIS-031, then TOOL-002 work), so it is filed as PLAN-010.
Content is r3 unchanged apart from this header and the outcome section.

## Problem

AERIS-023..028 were implemented and committed to `main` before any orchestrator attempt was
open, interleaved with unaccepted external work (web demo `0fbe16a`, tiling `c480bf0`, live
panel in `6698ae6`, detector profiles `79a7cb3`/`7a3b8f4`), with accepted AERIS-029/030 (`659b565`,
`ee4f64c`, `bc2bb64`), and with post-hoc fixes from an informal independent review (`ec09d38`).
The orchestrator verifies the delta between a `prepare` snapshot and the working tree of one
checkout. AERIS-023 closed two attempts for this reason; one remains.

## Contract changes this plan asks to ratify [F4]

`ec09d38` edited `orchestration/tasks.json` without a reviewed planning change. This plan
depends on those edits and asks the reviewer to ratify them explicitly:

1. AERIS-027 `write_globs` += `tests/fixtures/tracking/pre_aeris027_*.json`, `docs/mvp.md`.
   Reason: the golden-state fixture that replaces a test which could not fail (review finding),
   and the `--min-score` wording in `docs/mvp.md`, which became wrong when sub-threshold
   detections started extending tracks.
2. AERIS-028 `write_globs` += `tests/ui/test_web.py`, `docs/mvp.md`. Reason: the live-panel
   heading-convention test and the `--camera-motion` CLI documentation.
3. AERIS-028 AC2: `(default none)` → `(the default since the owner decision of 2026-10-15; none
   remains available)`, plus "the live panel heading uses the tracks.csv convention" and
   "predictions.jsonl, tracks.csv and label-based evaluation are unchanged between modes".
   The default flip is an owner decision (ADR 028); the two added clauses make AC2 stricter.

No criterion is weakened; no check, gate, tool or prompt changes. AC1/AC3 of every task and all
other entries are unchanged.

## Method

1. **Isolated replay worktree `R`** under the session scratchpad, *not* under the project
   directory (inventory includes `.claude/`, and AERIS-031's baseline covers it) [F2]. `R` stays
   detached at `afe12c3` throughout; trees are materialised with `git checkout <state> -- .`
   plus deletions, never by moving HEAD, so every attempt's `base_commit` is `afe12c3` and any
   retry has an empty rebase range. `base_commit` therefore does not describe the replay bases;
   each attempt's `source-before/` does [F13].
2. **Interpreter**: `/Users/vincenzo/Desktop/Projects/AERIS/.venv/bin/python` (absolute; `R` has
   no `.venv`). Dry runs confirmed `aeris.__file__` resolves inside `R` (`check_tests.py`
   prepends `R` and `R/python`) [F13].
3. **What is copied into `R/orchestration/runs/`** [F1]: `state.json` (sha256 recorded at copy
   time), every accepted task's evidence directory, and the whole `AERIS-023/` folder. Excluded:
   `workspaces/` (7 GB, includes a registered worktree), `AERIS-031/` (its open attempt owns the
   *shared* checkout, not `R`; its write_globs `scripts/qut_video_pilot.py`,
   `tests/unit/test_qut_video_pilot.py`, … do not overlap the replay), and every other unaccepted
   or closed attempt. This keeps the one-writer-per-checkout rule visible: `R` is a separate
   checkout with exactly one open attempt at a time.
4. **Media for real-model tests** are copied into `R`'s tool-ignored prefixes only
   (`IGNORED_PREFIXES`: `models/onnx/`, `data/raw/`), not gitignored paths such as `.claude/`
   [F13]. Skip counts depend on which media are present [R3]: each report cites the skip counts
   and reasons from its own check logs (the dry runs with weights and the FGVC sample copied
   showed one `unit` skip, the real-sample smoke test).
5. **Constant contract per attempt.** `tasks.json`, `gates.json`, `prompts/`, `tools/`,
   `AGENTS.md`, `CONTRACTS.md` are identical at `prepare` and `verify`. AERIS-023 uses its
   attempt-01 snapshot (its entry is byte-identical to `main`'s; `tools/`, `gates.json`,
   `prompts/`, `AGENTS.md`, `tests/test_orchestration.py` unchanged since `afe12c3`). AERIS-024..028
   use `tasks.json` from `main` `ddf9e5c` (sha256 `c54e19d2…7929`), which contains the edits above.
6. **States are built from patches, never whole-file copies** [F7], recorded as commits on a
   local scratch branch `replay-dry` (dry run) — reproducible from the patch files listed in the
   evidence. Dry-run results (orchestrator's own `inventory`/`changed`, all of the task's checks
   with the absolute interpreter, inventory unchanged afterwards) [F8]:

   | Task | Base | Delta (all inside write_globs) | Dry run |
   |---|---|---|---|
   | 023 (attempt-03, `--retry`) | `attempt-01/source-before/` exactly (empty delta vs its `base.json`) | `git diff afe12c3 0fbe16a -- <023 globs>`: 7 files | PASS (5 checks; unit 1 skip) |
   | 024 | tree of `0fbe16a` + `tasks.json` | `git diff 0fbe16a d47d053 -- <globs>` + adapted 024 fixes: 5 files | PASS |
   | 025 | A024 | `git diff d47d053 e2aea3b -- <globs>`: 7 files | PASS |
   | 026 | A025 | `git diff e2aea3b c1be200 -- <globs>`: 7 files | PASS |
   | 027 | A026 + all of `c480bf0` + non-027 hunks of `6698ae6` | 027 hunks of `6698ae6`, ADR, 027 fixes: 8 files | PASS |
   | 028 | A027 | `git diff 47aad0d 4e10199 -- <globs>` (3-way) + 028 fixes: 11 files | PASS |

   A026 equals `c1be200` except the carried 024 fixes and `tasks.json`. A027's `python/` equals
   `6698ae6`'s except the carried 024 docstring, so base ∪ delta reproduces `6698ae6` [F6].
7. **Adapted, coordinator-written code not on `main`** [F5], shown to each reviewer as new code:
   - 024 `tests/ui/test_web.py`: `ec09d38` wraps the slow-inference timing test in
     `mock.patch("aeris.web.live.timestamp_plan", …)` and asserts `timestamp_source`; neither
     exists before AERIS-030. Replay adds only `times[0] == 0.0` and strict increase (verified:
     fails with `t = position / fps`). The `cli.py` docstring and ADR 024 hunks are verbatim.
   - 027 `tests/integration/test_pipeline.py` off-switch test and the `docs/mvp.md` `--min-score`
     sentence are verbatim subsets of `ec09d38`; its 028 hunks in those files are deferred to 028.
   - 028 `tests/ui/test_web.py` `LivePanelTests`: `timestamp_plan` wrapper dropped (unpaced
     session, so the wrapper had no effect even on `main`; the API arrives with AERIS-030); assertions unchanged (verified: fails with the old
     `atan2(vy, vx)`). `live.py` gets only `ec09d38`'s heading hunk. `test_pipeline.py`, `cli.py`,
     `docs/mvp.md`, `camera_motion.py`, `docs/trajectory_estimation.md`, ADR 028 end byte-identical
     to `ec09d38`.
8. **Base-construction audit for reviewers** [F6]: each packet's review inputs include
   `git diff <previous after-state> <this base>` (empty for 025, 026, 028; for 027, `c480bf0` and the
   `6698ae6` panel). For 024 the real list is [R2]: deletes `.claude/launch.json`; modifies
   `docs/web.md`, `orchestration/tasks.json`, `python/aeris/web/server.py`, `static/app.js`,
   `static/index.html`, `static/style.css`; adds `python/aeris/web/live.py` (no `pyproject.toml` or
   Qt change: those were already in the 023 snapshot). The 024 reviewer is told that `live.py` and
   `docs/web.md` are 024 write-glob files that arrive through the base from the external `0fbe16a`
   web-demo work. The 027 reviewer must confirm the base holds no association, tracker or
   low-score logic.
9. **Replay vs `main`** [F7, R1]: `main` HEAD is `c6dcff8` at r3. Backing out the later external
   commits (`c6dcff8`, `7a3b8f4` — both `profiles.py` only —, `ddf9e5c`, `ee4f64c`, `659b565` cleanly;
   `79a7cb3` and `bc2bb64` per file) leaves 17 of 22 task files byte-identical to A028; the rest
   differ only by `79a7cb3` (profiles; the `7eb4161` merge added no resolution of its own), `bc2bb64`
   (AERIS-030 bearings/PTS) and the logged adaptations (`test_web.py`: 0 unexplained added lines).
   Re-check this immediately before execution if `main` has moved again. Any code change made during a replay retry must
   also land on `main` as its own commit and be carried into later bases.
10. **Roles and provenance** [F10, F11]. Reports use author ID `claude-opus-replay-coordinator`
    and name provenance: Codex `codex-aeris-023-implementer` (023 attempt-01), Claude Sonnet commits
    (`0fbe16a`, `6698ae6` web parts), Claude Opus commits (`d47d053`, `e2aea3b`, `c1be200`,
    `6698ae6` 027 parts, `47aad0d`, `4e10199`, `ec09d38`; this coordinator session wrote them) and
    the replay adaptations. The coordinator never edits a review and may not resolve or
    down-grade review findings; acceptance rests only on `evaluate` passing with the independent
    `review.json` [R4]. Reviewers are fresh agents that did not write any of this code and are
    not the informal reviewer behind `ec09d38`; each `review.json` records the reviewer's session
    identity. The 023 reviewer also receives attempt-01's `report.json`, and the 023 report
    declares the web-demo `draw_states` extraction in `cli.py` as non-023 content in scope.
11. **Later-code evidence** [F12]: ADR 027's table includes `estimate` rows measured at
    `bc2bb64`; the 027 report's limitations label them, and the reviewer confirms no 027
    criterion depends on them. The direction regression without compensation (p90 174.5°) is a
    known 027 property, recorded in ADR 027.
12. **Durable replay evidence** [R5]: before each review, the coordinator copies that task's
    patches, the adaptations log, `dry_check.py`, the base inventory and the dry-run output into
    the attempt folder, and names `replay-dry` commits in the report. The `replay-dry` branch is
    kept until copy-back is complete. The 028 reviewer also confirms that `--camera-motion none`
    reproduces the A027 written outputs (`predictions.jsonl`, `tracks.csv`) [R6].
13. **Failure handling.** A failed check or `revise` closes the attempt as the tool does.
    AERIS-023 gets no fourth attempt: on failure, stop and escalate with the smallest reproducer.
14. **Copy-back** [F9], only when no orchestrator command is running in the shared checkout and
    coordinated with AERIS-031's owner: copy `AERIS-023/attempt-03/` and `AERIS-024/`..`AERIS-028/`
    into the shared `runs/`, add only the six `state.json` keys, then verify every pre-existing key
    is byte-identical, each new entry's `report_sha256`/`review_sha256` matches the copied files,
    and `status` lists the six as accepted. Recorded `fingerprint`s describe `R`'s replay trees,
    not the shared checkout.
15. **Integration** [F13]: in a clean worktree at `main` HEAD (not the shared checkout), run the
    full suite, `ruff`, `check_tests.py`, `orchestrate lint` and a CLI smoke run on
    `tangier-landing-6s.mp4`; record HEAD, `git status` and results in the outcome section.

## Limits

Accepted states are historical replays, not current files. `main` also holds unaccepted work
(web demo and Qt plugin change from `0fbe16a` — under separate review —, tiling, live panel,
detector profiles) that this plan does not accept. No perception accuracy is claimed.

## Outcome (2026-10-17)

Executed in replay worktree `replay-exec` (HEAD `afe12c3` throughout) with a copied `runs/` subset;
all states built from patches on local branches `replay-dry`, `replay-dry-2`, `replay-dry-3`.

| Task | Attempts | Result | Formal-review findings that forced a retry |
|---|---|---|---|
| 023 | 3 (1–2 closed earlier) | accepted | none |
| 024 | 2 | accepted | AC2 call sites untested -> `ForecastNoiseCallSiteTests` |
| 025 | 1 | accepted | none |
| 026 | 2 | accepted | live gate untested; ADR 026 claimed k = 1.0 removed all reversed arrows (one remained) -> live-gate test, independent velocity-std test, corrected ADR table |
| 027 | 1 | accepted | none (first reviewer cut off by an API limit before writing; a fresh reviewer completed) |
| 028 | 3 | accepted | 1: masking/downscale and map-back untested -> estimator, batch and live tests; 2: new live test flaky (stub keyed on processed-frame count while the paced loop drops frames) -> media-time keying, 3 px tolerance |

Deviations from the method: retries added coordinator-written tests, so the later replay states were
rebuilt by cherry-picking onto the new after-states (`replay-dry-2`, `replay-dry-3`) and re-dry-run
before use; an `ec09d38` defect surfaced on the way (`LivePanelTests` unpaced, wall-clock timing made it
order-dependent) and was fixed in the 028 delta. Every review-driven change (tests, ADR 026/028
corrections, two `metrics.json`/`benchmark.json` metadata strings) was then ported onto `main` and
mutation-checked there. Open minors are listed in `docs/PROJECT_STATE.md`. Integration results are
recorded in the commit that copies the evidence back.
