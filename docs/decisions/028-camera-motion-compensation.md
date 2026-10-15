# AERIS-028 — Scene-relative forecasts under camera motion

Status: implementation for independent review. Date: 2026-10-15.

All motion was measured in image pixels. When the camera pans to follow an aircraft, which
is common in recorded aircraft footage, the aircraft barely moves in the image: the arrow was
hidden by the AERIS-026 gate or pointed the wrong way, and headings described the camera
operator rather than the aircraft. The project MVP assumes a fixed camera, so this is opt-in
for the CLI (`--camera-motion estimate`, default `none`); the web live view always uses it.

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
arrow points left along the flight path on every sampled frame, without it the arrow is
missing at frames 45 and 55. The live view on the same clip estimated 57 of 57 steps.

Limits: errors accumulate along the chain; a moving background (waves, clouds in wind,
other aircraft not detected) biases the estimate; zoom is modelled only as uniform scale;
parallax is ignored. No physical heading, speed or accuracy claim follows.
