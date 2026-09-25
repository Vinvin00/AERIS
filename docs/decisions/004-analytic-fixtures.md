# ADR 004 — Analytic ground-truth fixtures

Status: proposed for AERIS-004 review. Date: 2026-09-25.

Decision: `python/aeris/simulation/fixtures.py` generates two deterministic, SYNTHETIC
fixture families under `tests/fixtures/simulation/`. Each family has an `inputs/`
directory that runtime code may read and a `truth/` directory that only evaluators and
tests may read. `manifest.json` gives each artifact's role, SHA-256, size and record
count, plus the seed and a scope label.

- **Image space.** `sim-cv-crossing` has three constant-velocity boxes, variable delta
  times, and dropped frames 6 and 7. Two tracks cross at t = 0.5 s, one track is
  hidden for two frames, and one is stationary. `sim-empty-birth` has complete empty
  frames before and after a visible track. Inputs are contract-v1 `FrameRecord` and
  `DetectionRecord` lines plus rendered PGM frames. Truth is an AERIS-003 canonical
  dataset (`dataset.json`, `annotations.jsonl`), which `aeris.data.validate` accepts,
  and `expected.json`. `expected.json` holds per-frame analytic states (including
  hidden ones), visible spans, frame-index gaps, delta times, empty frames, crossing
  and overlap events with IoU, heading and speed, detection sources, and
  constant-velocity forecast targets at the contract horizons.
- **Projection.** A calibrated camera (K plus OpenCV Brown–Conrady distortion) sees a
  target moving at constant velocity in the camera frame. Inputs are `camera.json`
  (calibration, conventions and a crop/resize model-input transform) and
  `observations.jsonl` (distorted marker pixels and their box). Truth is scene
  positions, unit rays, horizontal and vertical angular extents, and the line-of-sight
  angular rate |r × v| / |r|². It also holds pixel/ray round-trip points, an on-axis
  reference extent, and transform check points.

Conventions:

- **Pixel coordinates.** Pixel (i, j) spans [i, i+1) × [j, j+1), so the top-left pixel
  centre is (0.5, 0.5). This is the contract's half-open convention. OpenCV puts that
  centre at (0, 0), so an OpenCV principal point is (cx − 0.5, cy − 0.5). The tests
  apply that shift when comparing with `cv2.projectPoints`/`cv2.undistortPoints`.
- **Scene units.** They are arbitrary; only angular quantities are meaningful. Range
  appears only as evaluator truth and implies no metric claim.
- **Heading.** atan2(−vy, vx), counterclockwise from image right, in [0, 360). Below
  1e-9 px/s it is null with a reason.

Reproducibility:

- **Exact positions.** Image-space timestamps are dyadic and sizes even, so centres
  and boxes are exact in binary floating point.
- **Detector noise.** It is an Irwin–Hall sum of four uniforms taken from SHA-256 of
  (seed, key), using only IEEE arithmetic. That keeps it independent of platform,
  NumPy version and libm. Each frame's noise is keyed by its own identifiers, so
  inputs up to time t never depend on later frames.
- **Trigonometric values.** Only headings and angular extents need libm. They are
  rounded to 9 decimal degrees and 12 decimal radians, so a last-bit difference
  between platforms changes the committed bytes only with negligible probability.
- **Images.** Frames are binary PGM, which is byte-deterministic, unlike PNG or zlib
  output. They are not committed: their hashes are pinned in the manifest and
  `--images` writes them. JSON uses sorted keys and repr floats.

Reason: later tasks need exact, reviewable expected values with no truth in their
inputs. AERIS-005 needs metric edge cases, AERIS-009 crossings, gaps and duplicates,
AERIS-010 forecast targets, and AERIS-011 projection, undistortion and transform
round trips. Checking against independent implementations (OpenCV, closed forms)
keeps the fixtures from agreeing only with themselves.

Consequences: fixture changes are deliberate. Regenerate with
`PYTHONPATH=python python -m aeris.simulation.fixtures tests/fixtures/simulation`, then
review the diff and the manifest hashes. These fixtures are not perception, tracking,
forecast or calibration evidence, and real-video claims still need AERIS-007 onwards.
Filter-covariance fixtures are left to AERIS-009/010, which own the noise model.
