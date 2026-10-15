# AERIS-027 — Low-score detections extend existing tracks

Status: implementation for independent review. Date: 2026-10-15.

Detections below the operating threshold (0.3, untuned) were dropped before tracking, so a
confirmed aircraft whose score dipped for a few frames coasted on prediction alone or was
lost. `KalmanTracker.step(..., high_score=...)` now matches in two passes (the ByteTrack
idea): detections at or above `high_score` are matched to all tracks and may start new
tracks; the remaining detections are then matched, with the same chi-square gate, only to
confirmed or lost tracks left unmatched. Low-score detections never start a track and
never confirm a tentative one. The default `high_score = 0.0` keeps the previous behaviour.

The batch CLI passes every detection the detector kept (its 0.1 score floor) with
`high_score = --min-score`. Detection metrics and `detections_at_or_above_min_score` still
use only detections at or above the threshold. `metrics.json` adds
`low_score_track_updates`, and the manifest records `low_score_pass`. The live demo uses
the same rule. Measurement noise is unchanged for low-score detections; there is still no
validated confidence-to-noise model.

Smoke comparison on `tangier-landing-6s.mp4` (one track, 2x2 tiles, scored against later
detections per AERIS-025; not ground truth):

| low-score pass | confirmed rows coasted | low-score updates | 1 s forecasts scored | median 1 s error |
|---|---|---|---|---|
| off | 19 of 55 | 0 | 23 | 34 px |
| on | 1 of 55 | 18 | 44 | 37 px |

**Direction regression (found by independent review, reproduced 2026-10-15 at bc2bb64).**
The table above omitted direction. Keeping tracks measured through noisy stretches lets the
AERIS-026 arrow gate pass arrows it previously hid, and some point backwards. Same clip,
1 s constant-velocity arrows actually drawn, scored against later detections:

| camera motion | low-score pass | coasted | arrows drawn | direction median / p90 |
|---|---|---|---|---|
| none | off | 19 of 55 | 18 | 3.5° / 7.8° |
| none | on | 1 of 55 | 28 | 4.4° / 174.5° |
| estimate | off | 19 of 55 | 35 | 1.9° / 2.9° |
| estimate | on | 1 of 55 | 54 | 2.0° / 5.9° |

Without camera-motion compensation the pass undoes ADR 026's "removes the reversed
arrows"; with it (AERIS-028, now the default) the reversals stay suppressed and coverage
rises from 35 to 54 arrows at a p90 of 5.9° instead of 2.9°. The arrow threshold k was
selected without this pass and without compensation and should be re-selected on more
footage. One clip and one track: this shows the plumbing works and coasting drops. It is not
evidence of better tracking accuracy. Identity switches caused by a low-score false
positive inside the gate remain possible and need labeled footage (AERIS-021) to measure.

Provenance: the implementation landed in commit 6698ae6 together with unrelated web
console work from a concurrent session in the same checkout. Code and tests are as
written for this task. Tests: `tests/unit/test_tracking.py::LowScorePassTests`,
`tests/unit/test_tracking.py::PreLowScoreBehaviourTests` (AC1: with the default threshold the
tracker reproduces golden states recorded from the pre-change tracker at c1be200 on a
cluttered, irregularly timed scene, `tests/fixtures/tracking/pre_aeris027_cluttered_states.json`;
it fails if any detection is treated as low-score by default; a companion test shows the
pass changes that scene), `tests/integration/test_pipeline.py::...test_low_score_pass_off_drops_sub_threshold_detections_before_tracking`
(the off switch: fails if the flag is ignored), and
`tests/integration/test_pipeline.py::...test_low_score_detections_extend_tracks_without_counting_as_detections`
(fails with `low_score_pass=False`). One existing test double's `step` override was
widened to pass keyword arguments through; its assertions are unchanged. A first
"default keeps previous behaviour" test compared the new code with itself and could not
fail; review caught it and it was replaced by the golden test. The `duplicates_dropped` and
`gated_pairs` counters now also count low-score detections (internal, not exported).
