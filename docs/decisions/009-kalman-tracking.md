# ADR 009 — Constant-velocity Kalman tracking with a confirmed-first association cascade

Status: proposed for AERIS-009 attempt 1 review. Date: 2026-09-27.

Context: AERIS-009 turns per-frame detections (AERIS-006/008) into identity-stable
tracks over time, gated on real elapsed time rather than frame count, and reports
tracking metrics (AERIS-005's pinned TrackEval wrapper) on real identity-labelled video.

Decision:

- **State and motion model.** `[cx, cy, vx, vy, w, h]` (CONTRACTS.md), constant velocity
  for the box centre. Width and height are tracked too and are part of the measurement
  vector `[cx, cy, w, h]`, not left out: on the crossing scenario below, two tracks'
  predicted centres coincide exactly at the crossing instant, and only their sizes (16x8
  vs 12x12 px) tell them apart. Dropping size from the gate would make that crossing
  unrecoverable without appearance features.
- **Process and measurement noise.** `Q(dt)` is a discretized white-noise-acceleration
  model for each of the (x, vx) and (y, vy) pairs, plus a term for width/height whose
  variance grows linearly with `dt`; every term depends on elapsed time, never frame
  count. `R` is fixed from configured standard deviations. CONTRACTS.md requires a
  detector-score-to-`R` mapping to be "documented and validated"; none exists yet, so
  score is not used to modulate `R` at all — `configs/tracker/baseline.json`'s
  `scope_note` says so, and this is a limitation, not an oversight.
- **Numerical stability.** Covariance updates use Joseph's form; every predicted and
  updated covariance is symmetrized and its eigenvalues clipped at a float-noise
  tolerance (`_symmetrize_psd`), raising if an eigenvalue is negative beyond that
  tolerance. A track's estimated box is shifted, not shrunk, to keep a non-negative
  origin when its centre estimate sits close to the image edge (CONTRACTS.md's `BBox`
  requires `x_min, y_min >= 0`).
- **Association: a two-pass, confirmed-first cascade,** not one flat Hungarian
  assignment. Already-confirmed tracks are matched against every detection first; only
  then do tentative tracks compete for what is left. Rationale, found empirically on the
  AERIS-004 crossing fixture's duplicate-detection case: a track that was just spawned
  from a duplicate has a wide, barely-constrained covariance, which can score a
  *smaller* Mahalanobis cost to the next real detection than an established track whose
  covariance has settled — letting a one-frame duplicate steal an established track's
  identity outright. A single flat assignment pass does not protect against this; the
  cascade does, by construction, at no cost to the crossing case (confirmed tracks never
  compete with each other for priority, only with tentative ones).
- **Duplicate suppression.** Detections are greedily IoU-deduplicated within a frame
  (highest score kept) before association, at threshold 0.5 (matching TrackEval's own
  distractor IoU). This alone did not fix the case above on the fixture's actual jittered
  boxes (measured IoU 0.44, just under the threshold); it is kept anyway as a
  self-justifying, independent defence against genuinely overlapping duplicates,
  documented as such rather than tuned to this one fixture's noise draw.
- **Lifecycle: `tentative -> confirmed -> lost -> deleted`, in seconds.** A track is
  confirmed once its hit count reaches `min_hits_to_confirm` (irreversible); a confirmed
  track that goes unmatched becomes `lost` and flips back to `confirmed` the moment it
  matches again; any track (tentative or lost) is deleted once its time since last match
  exceeds `lost_timeout_s`. Both thresholds are configured, not hardcoded.
- **Gating.** Mahalanobis-squared distance against each track's own innovation
  covariance, gated at the chi-square quantile for a 1% false-alarm rate over the
  4-dimensional measurement (`scipy.stats.chi2.ppf(0.99, 4)`), not a hand-picked pixel
  radius.
- **Real evidence (AC3), two explicitly separate results, not one.** Both use 23 pinned,
  contiguous frames of one real AOT flight (`tests/fixtures/tracking/real_samples.json`;
  frames 18-40 of `69904ae4f18d46ec8886e8fd6179171b-camera0`, part1, a scored Helicopter
  visible throughout, 16-45 px wide):
  - **Oracle ablation:** the flight's own ground-truth boxes are fed to the tracker
    directly (score 1.0, provenance says "oracle ground truth (ablation, not a
    detector)"). This measures the tracker's own real-world performance, decoupled from
    detector recall: HOTA 0.699, MOTA 0.957, IDF1 0.978, **0 identity switches**, on
    real, non-synthetic geometry. This is the established "oracle" pattern already used
    for forecast evaluation (docs/CONTRACTS.md's "oracle-identity forecast protocol"),
    not a new precedent.
  - **End-to-end real detector + tracker:** the accepted AERIS-006 ONNX detector, on the
    same real frames, finds nothing (0 detections; HOTA/MOTA/IDF1 all 0). This is the
    honest, expected result, not a bug: AERIS-007 already measured near-zero AOT recall
    below 32 px, and this helicopter is 16-45 px throughout. Reporting a favourable
    number here would have meant hunting for a larger, rarer real target instead of
    using the plainly available one — the kind of curation AGENTS.md forbids.
  - Both are on the *same* real flight so a reader can see exactly which factor (detector
    recall, not the tracker) drives the difference.

Alternatives rejected:

- **IoU-only association (no motion model).** Common in simple trackers (SORT's default),
  but fails outright on the crossing scenario at the crossing frame itself, where the two
  boxes may not even overlap the "wrong" track's box, giving no signal at all; a
  Kalman-predicted position keeps working through a crossing precisely because it
  extrapolates each track's own motion.
- **Re-weighting association by hit count instead of a hard cascade.** A single cost
  matrix with a hand-tuned "confirmed bonus" subtracted from established tracks' costs
  would also fix the observed failure, but the bonus's size is an unprincipled free
  parameter with no natural scale; the two-pass cascade needs no such constant and is a
  well-established pattern (DeepSORT's matching cascade prioritises tracks the same way,
  by recency/confirmation).
- **Modulating `R` by detector score.** Rejected per CONTRACTS.md: no validated mapping
  exists. A future task could add one against a real precision/recall curve.
- **A single real-video result (detector only).** Reporting only the near-zero
  end-to-end number would honestly describe the current detector's limits but say
  nothing about whether the *new* component (the tracker) works on real geometry; the
  oracle ablation exists to answer that question honestly, clearly labelled as an
  ablation rather than a deployment claim.

Consequences:

- Any future change to the detector's score calibration should revisit whether a
  validated score-to-`R` mapping is now possible; until then, R is a fixed, engineering
  default (`configs/tracker/baseline.json`), not tuned or validated against real data.
- The real detector's zero recall on this flight means AC3's end-to-end path is not
  exercised beyond "it runs and reports honestly" on this specific evidence; a future
  task with a larger real-detector-visible target (AERIS-017's tiny-target work, or a
  fixed-camera domain closer to the MVP target) would give a more informative end-to-end
  number.
- `orchestration/gates.json`'s `tracking` check does not yet set
  `AERIS_REQUIRE_REAL_TRACKING=1` (the same governance gap already flagged for 006/007/008);
  `verify` was run with it set explicitly.
