# ADR 004 — Analytic fixture boundary

Status: proposed for AERIS-004 review. Date: 2026-09-30.

Use a standard-library deterministic generator in `aeris.simulation.fixtures`.
`tests/fixtures/simulation/input/` contains only PPM observations, frame metadata,
and a fixed ideal pinhole camera calibration. `truth/` contains contract-v1 motion
annotations and separate world-coordinate projection truth. Evaluators may read
truth; detector, tracker, and predictor inputs must be built only from `input/`.
The manifest records a fixed seed, file SHA-256 digests, units, derivation, and
synthetic scope. Regenerate from the repository root with:

```sh
PYTHONPATH=python python3 -c 'from pathlib import Path; from aeris.simulation.fixtures import generate; generate(Path("tests/fixtures/simulation"))'
```

The image-space fixture has two identities at irregular times. One disappears at
0.25 s, both overlap at their path crossing at 0.5 s, and the last frame is a complete
negative. Its
unobserved state during the gap is defined by the analytic trajectory but is not
an annotation at that frame. The projection fixture uses ideal K with fx=fy=100 px,
principal point (32,24) px, no distortion, and three specified world points at
10 m depth. These are exact toy geometry checks, not camera calibration evidence.

The PPM sprites are simple visible squares, not representative aircraft imagery.
Passing these fixtures establishes contract and arithmetic behavior only; it says
nothing about real detection, tracking, uncertainty calibration, or forecast quality.
