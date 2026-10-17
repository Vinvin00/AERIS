# AERIS-028 — Scene-relative forecasts under camera motion

Status: implementation for independent review. Date: 2026-10-15.

All motion was measured in image pixels. When the camera pans to follow an aircraft, which
is common in recorded aircraft footage, the aircraft barely moves in the image: the arrow was
hidden by the AERIS-026 gate or pointed the wrong way, and headings described the camera
operator rather than the aircraft. It was first opt-in for the CLI. **Owner decision
2026-10-15, after independent review:** `--camera-motion estimate` is now the CLI default
(use `none` for a known fixed camera); the web live view always uses it. Reason: the
AERIS-027 low-score pass reintroduces reversed arrows without compensation (see ADR 027).
On a fixed camera a correct fit is near identity and a failed step is identity, but a wrong
fit on moving background (water, clouds) can add false motion there; this is a deliberate
change to the fixed-camera MVP default, not evidence that compensation is harmless.

Design:
- `CameraMotion` estimates a similarity (translation, rotation, uniform scale) between
  consecutive processed frames from up to 400 Shi-Tomasi corners tracked with pyramidal
  Lucas-Kanade and fitted with RANSAC (2 px threshold, at least 20 inliers), on a copy
  downscaled to at most 640 px. Detection boxes, enlarged by 20%, are masked out. Transforms
  chain into a frame-to-reference map. A similarity, not a homography, because the footage
  is far-field and a homography over sparse sky corners is poorly constrained.
- The tracker still runs in image pixels; association needs image positions, and a
  following camera keeps image motion small, which suits it.
- A separate constant-velocity forecaster receives each assigned detection center mapped to
  the reference frame. It drives the arrow, the heading gate and, in the live view, the
  panel; arrow endpoints are mapped back to the current frame. Written `PredictionRecord`s
  remain the original image-space forecasters, so the predictions contract and label-based
  evaluation are unchanged.
- A failed step (featureless sky, blank frame) is counted and treated as zero camera
  motion. That reproduces the uncompensated behaviour for the step and is never presented
  as a measurement. `pipeline.process_video` gained an optional `frame_hook` so the CLI can
  estimate motion during the single decode pass.

Evidence. Synthetic: a 174 px pan over 29 frames is recovered within 1.5 px with a moving,
masked aircraft in view. An aircraft fixed in the scene under a 4 px/frame pan draws arrows
without compensation and none with it. Featureless frames fail as intended.
`tangier-landing-6s.mp4` (one track, 2x2 tiles, scored against later detections per AERIS-025,
not ground truth; image-space and scene-relative scores are in different frames, so this is
not a like-for-like accuracy comparison):

| mode | motion steps estimated/failed | 1 s arrows drawn | direction median / p90 |
|---|---|---|---|
| none | — | 28 | 4.4° / 174.5° |
| estimate | 59 / 0 | 54 | 2.0° / 5.9° |

Rendered overlays show the camera panning left with the aircraft; with compensation the
arrow points left along the flight path on every sampled frame. Counting every frame (2x2
tiles, overlay drawing captured from the render pass): without compensation 29 of the 58
tracked frames have no arrow, including frames 44–51 and 56–59 in the panning stretch; with
it only frames 2–4 (track warm-up) lack one. (Corrected after formal review, 2026-10-16: the
earlier text said the uncompensated arrow was missing at frames 45 and 55; frame 55 has one.)
The live view on the same clip estimated 57 of 57 steps in one run; the count depends on
dropped frames.

Review follow-ups (2026-10-15): the pan integration test now also asserts that
`predictions.jsonl` and `tracks.csv` are byte-identical between `none` and `estimate`; the live
panel heading used `atan2(vy, vx)` (clockwise, 90 = down) while `tracks.csv` and CONTRACTS use
`atan2(-vy, vx)` counterclockwise; it now matches, tested by
`tests/ui/test_web.py::LivePanelTests`, which also covers the live gate and the live
camera-motion failure path on blank frames. The module docstring no longer says estimation is
"never worse": that holds only for failed steps. Motion-estimation time runs inside the frame
hook and is not attributed to any stage in `benchmark.json` (wall clock only).

Formal-review follow-ups (2026-10-16): tests now cover masking with a strongly textured
target that dominates the corners (at 640x360 and at 1280x720, where the fit runs on a
downscaled copy and the mask must be scaled), an HD clip with uneven pan, rotation and zoom
checked against the true transform at the frame corners (fails if the chain order is
swapped or the downscale conversion is skipped), the batch arrow mapped back to the current
frame under a pan with scene motion, and the live panel heading and 1 s forecast on a
textured panning clip. `metrics.json` names the coordinate frame of
`forecast_vs_detections_displayed` (reference frame with camera motion, original frame without),
and `benchmark.json` notes that motion estimation is not attributed to a stage. A failed
step keeps the accumulated transform, so it equals the uncompensated case for that step only.

Limits: errors accumulate along the chain; a moving background (waves, clouds in wind,
other aircraft not detected) biases the estimate; zoom is modelled only as uniform scale;
parallax is ignored. No physical heading, speed or accuracy claim follows.
