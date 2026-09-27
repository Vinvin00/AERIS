# Kalman tracking and gated Hungarian association (AERIS-009)

Status: attempt 1, awaiting independent review.

`aeris.tracking.tracker` turns per-frame `DetectionRecord`s into identity-stable
`TrackState`s over time: a constant-velocity Kalman filter per track, gated Hungarian
association in a confirmed-first cascade, and explicit lifecycle transitions in elapsed
seconds. Decisions and alternatives are in ADR 009. This module never sees ground truth
at runtime; it consumes only the current and past frames of one sequence.

```
tracker = Tracker(sequence_id, config_from_dict(load_config()))
for timestamp_s, detections in frames:            # strictly increasing timestamps
    states = tracker.step(timestamp_s, detections) # every live TrackState
```

## State, motion model and association (AC1, AC2)

State is `[cx, cy, vx, vy, w, h]` (CONTRACTS.md "Track state"). Width and height are
part of the *measured* vector `[cx, cy, w, h]`, not just carried along: when two tracks'
predicted centres coincide (a true crossing), their sizes are often the only thing that
still tells them apart, and the tests below rely on exactly that.

Each `step(timestamp_s, detections)`:

1. Validates `timestamp_s` (finite, strictly greater than the previous call) and that
   every detection belongs to this tracker's sequence and carries this frame's timestamp.
2. Suppresses duplicate detections of the same object: greedy IoU non-max suppression
   within the frame, highest score kept, threshold 0.5 (TrackEval's own distractor IoU).
3. Predicts every live track forward by the elapsed time (`Q(dt)` below).
4. Runs gated Hungarian assignment in **two passes**: already-confirmed tracks match
   against every detection first, then tentative tracks match against what is left.
   A pair is only accepted inside the gate: Mahalanobis-squared distance against the
   track's own innovation covariance, gated at `scipy.stats.chi2.ppf(0.99, 4)` (a 1%
   false-alarm rate over the 4-dimensional measurement).
5. Updates matched tracks (Joseph-form covariance update), marks unmatched tracks
   missed, and starts a new tentative track for every still-unmatched detection.
6. Deletes any track whose time since its last match exceeds `lost_timeout_s`.
7. Returns every remaining track's `TrackState`.

**Why the two-pass cascade.** A flat, one-shot Hungarian assignment over all tracks and
detections together has a real failure mode, found on the AERIS-004 fixture itself: a
track just spawned from a duplicate detection has a wide, barely-constrained covariance,
which can score a *smaller* Mahalanobis cost to the next real detection than an
established track whose covariance has settled after many updates — letting the
duplicate's spurious track steal the established one's next detection, and with it its
identity, outright. Matching confirmed tracks first removes the competition.

**Lifecycle**, in elapsed seconds, never frame count:

| Transition | When |
|---|---|
| (none) `->` tentative | A detection is not matched to a live track |
| tentative `->` confirmed | The track's matched-frame count reaches `min_hits_to_confirm` (irreversible) |
| confirmed `->` lost | The track goes unmatched this frame |
| lost `->` confirmed | The track matches again |
| tentative or lost `->` deleted | Time since the last match exceeds `lost_timeout_s` |

`TrackState.observed` is `True` only on a step where the track was actually matched;
otherwise its state is a pure prediction. `age_s` and `time_since_seen_s` are computed
from timestamps, so a run with irregular frame spacing (dropped frames, variable frame
rate) behaves the same as one with even spacing over the same elapsed time.

**Process and measurement noise** (`configs/tracker/baseline.json`):

- `Q(dt)`: a discretized white-noise-acceleration model for the (x, vx) and (y, vy)
  pairs (`accel_std_px_s2`), plus a term for width/height whose variance grows linearly
  with `dt` (`size_std_px_sqrt_s`). Both explicitly depend on elapsed time.
- `R`: fixed from `position_std_px` and `size_std_px`. CONTRACTS.md requires a
  detector-score-to-`R` mapping to be "documented and validated"; none exists yet, so
  detector score is **not** used to modulate `R` — a documented limitation, not an
  oversight.
- Every predicted and updated covariance is symmetrized and eigenvalue-clipped at a
  float-noise tolerance (`_symmetrize_psd`); it raises `TrackingError` if an eigenvalue
  is negative beyond that tolerance, rather than silently returning an invalid matrix.
- A track's box origin is shifted, not shrunk, to stay non-negative when its centre
  estimate sits close to the image edge (`BBox` requires `x_min, y_min >= 0`).

## Real evidence (AC3)

Tracking metrics need identity-labelled ground truth: `evaluate_tracking` (AERIS-005)
runs pinned TrackEval HOTA, CLEAR and Identity only on sequences where every frame is
`complete` and every non-ignored object carries a `track_id`. AOT (already adapted by
AERIS-007, `aeris.data.aot`) has both; the Haneda clip (AERIS-008) does not, so it
cannot back a tracking-performance claim.

**Pinned real sample** (`tests/fixtures/tracking/real_samples.json`): 23 contiguous
frames (indices 18-40, part1, CDLA-Permissive-1.0), flight
`69904ae4f18d46ec8886e8fd6179171b-camera0`, one scored `Helicopter1` visible throughout
(16-45 px wide). Frames are fetched by key and checked against a pinned SHA-256 (not
committed, per AGENTS.md):

```
python -m aeris.tracking.real_sample fetch
```

reuses `aeris.data.aot.fetch`'s S3 ETag and SHA-256 checks (AERIS-007); an already-present
file whose SHA-256 matches the pin is reused without a request. `RealAotSequenceTests`
checks the same pins itself and skips (or, under `AERIS_REQUIRE_REAL_TRACKING=1`, fails)
if the frames are not already present — it does not fetch during a test run, matching
AERIS-006/007/008's REAL tests.

Two explicitly separate real results, not one, so a reader can see which factor drives
the difference:

| | Input | HOTA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|
| **Oracle ablation** | the flight's own ground-truth boxes, fed to the tracker directly (score 1.0, labelled "ablation, not a detector") | 0.699 | 0.957 | 0.978 | **0** |
| **End-to-end** | the accepted AERIS-006 ONNX detector's real output on the same frames | 0.0 | 0.0 | 0.0 | 0 |

- **Oracle ablation** measures the tracker's own real-world performance, decoupled from
  detector recall: zero identity switches over the whole sequence, on real (not
  synthetic) geometry. This is the same "oracle" pattern CONTRACTS.md already describes
  for forecast evaluation, not a new one, and is clearly labelled as an ablation, never a
  deployment claim.
- **End-to-end** finds nothing: the real detector produces 0 detections on this
  16-45 px real target. This is the honest, expected result, not a bug — AERIS-007
  already measured near-zero AOT recall below 32 px, and reporting a favourable number
  here would have meant hunting for a rarer, larger real target instead of using the
  plainly available one.

## Tests (`tests/unit/test_tracking.py`)

**Unit tests (SYNTHETIC):**

- **Kalman math**, hand-built, noiseless: constant-velocity prediction is exact under
  variable `dt` (a 1 s gap and a 10 s gap from the same state); process noise grows with
  elapsed time, not frame count; covariance stays finite, symmetric and PSD over 30
  random steps; `_symmetrize_psd` clips float-noise asymmetry but raises on a genuine
  negative eigenvalue.
- **Empty inputs:** no detections and no tracks is a no-op; detections with no existing
  tracks all spawn tentative tracks; an existing track survives one empty frame inside
  its timeout.
- **Validation:** a non-increasing or non-finite timestamp, a detection from the wrong
  sequence, and a detection whose timestamp does not match the step's are all refused.
- **Duplicate suppression:** the lower-scoring of two overlapping boxes is dropped;
  non-overlapping and below-threshold-overlap boxes are both kept.
- **AERIS-004 fixtures** (`sim-cv-crossing`, `sim-empty-birth`), mapped to source track
  IDs by nearest true centre (test bookkeeping only, not part of the tracker):
  - the crossing tracks keep separate, stable IDs through and after the crossing frame;
  - t1's single missed detection (frame 4) keeps the same ID, going `lost` then
    `confirmed` again;
  - the duplicate (frame 9, on stationary t3) and the false positive (frame 2) never
    reach `confirmed`;
  - t3 stays continuously `confirmed` through the duplicate frame (the cascade's job);
  - final positions match the scenario's analytic constant-velocity ground truth;
  - the empty-birth track is silent in frames 0-1, tentative at birth, confirmed by its
    second detection, `lost` (not deleted) through two empty frames inside the timeout,
    and would be deleted if the timeout were shorter than the gap.

**REAL tests** (`AERIS_REQUIRE_REAL_TRACKING=1` makes skips fail): the oracle ablation
(0 identity switches, MOTA and IDF1 both `>= 0.9`) and the end-to-end real detector run
(asserts the evaluator actually ran and returned a combined result — not a quality
claim, since the honest result here is near-empty).

## Outputs

`python -m aeris.tracking.tracker frames.jsonl detections.jsonl --out DIR` writes
`tracks.jsonl` (one line per live `TrackState` per frame) for one sequence; this is a
convenience CLI for inspection, not a required artifact of any accepted evidence.

## Limitations

- **No detector-score-to-`R` mapping.** Detector score never modulates measurement
  noise; CONTRACTS.md requires such a mapping to be validated, and no validation exists.
- **No appearance model.** Identity through a crossing relies on position and size only;
  a crossing between two similarly-sized, similarly-moving objects is not covered by any
  test here and may not be resolvable by this design at all.
- **Real end-to-end evidence is a single, real, honestly-negative result.** The
  detector's real recall on this flight's small target is zero; a real end-to-end result
  with actual real matches would need a larger, rarer real target or a later detector.
- **Duplicate suppression's IoU threshold** (0.5) does not, by itself, suppress the
  fixture's own jittered duplicate (measured IoU 0.44); the confirmed-first cascade is
  the layer that actually keeps identity stable in that case. A lower threshold was not
  chosen instead, to avoid tuning a general-purpose parameter to one fixture's noise
  draw.
- **Track IDs are per-`Tracker` instance,** monotonically increasing strings; nothing
  persists across separate `Tracker` instances or process restarts.
- **`orchestration/gates.json`'s `tracking` check does not set**
  `AERIS_REQUIRE_REAL_TRACKING=1` (the same gap already flagged for 006/007/008);
  `verify` was run with it set explicitly.
