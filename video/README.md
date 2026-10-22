# AERIS demo video (Remotion)

Portfolio/social cut built from real AERIS runs at 30 fps.

- Boxes: observed rows of the run's `tracks.csv` on that exact frame (coasted predictions not drawn).
- Trails: the same observed track centers over the previous 40 frames.
- Forecast arrows: exactly the 1 s arrows AERIS's own overlay draws (camera-motion compensated,
  gated by `heading_resolved`, ADR 026). `run_aeris.py` runs the unchanged CLI and saves the items
  passed to `cli.render_overlay` as `forecast.json`; nothing is recomputed.
- Numbers: `metrics.json` (`forecast_vs_detections_displayed`, constant velocity, 1.0 s horizon; scored
  against later AERIS detections, not ground truth) and `benchmark.json` (inference p50).

## Rebuild

1. Sources (Wikimedia Commons, SHA-1 as in `python/aeris/ui/samples.py` and `web/demo_clips.py`)
   re-encoded to 30 fps in `build/demo-video-work/clips/`: `formation.mp4` (full 10 s),
   `red-arrows.mp4` (first 20 s), `air-force-one.mp4` (first 630 frames).
2. AERIS runs (`W=build/demo-video-work`, `PYTHONPATH=python`):
   ```sh
   .venv/bin/python video/run_aeris.py $W/clips/formation.mp4 --output $W/runs/v3-formation --profile fast-zoom
   .venv/bin/python video/run_aeris.py $W/clips/red-arrows.mp4 --output $W/runs/v3-red-arrows --profile accurate
   .venv/bin/python video/run_aeris.py $W/clips/air-force-one.mp4 --output $W/runs/v3-air-force-one --profile accurate
   ```
3. `cd video && npm install && npm run data && npm run render` → `out/aeris-demo-16x9.mp4`.
   `npm run studio` opens the live editor.

## Windows used (from per-frame checks)

| Scene | Run | Window | Check |
|---|---|---|---|
| Blue Angels | v3-formation (fast-zoom) | 6.5–10 s | 4 boxes on every frame, 4 track IDs; arrows from 8.1 s, when all four pass the gate |
| Red Arrows | v3-red-arrows (accurate) | 8.57–12.37 s | 9 boxes on 99/102 frames after lock (10 on two frames, 8 on one); arrows on all 9 tracks on every frame from 1.2 s; camera-compensated, so they run ~1,300 px off-screen (heading error 1.3°, position error 132 px, shown in the panel) |
| Air Force One | v3-air-force-one (accurate) | 0–4.1 s | 1 box on 123/123 frames, 1 track ID; arrow drawn on 85/87 frames from 1.3 s |

Media, data and renders are ignored by Git (`public/`, `src/data/`, `out/`).
