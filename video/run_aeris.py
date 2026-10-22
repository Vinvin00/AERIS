"""Run the AERIS CLI unchanged, and also save what its overlay draws (forecast.json).

`python -m aeris` keeps its displayed 1 s forecast arrows (camera-motion compensated and
gated by heading_resolved, ADR 026) only inside overlay.mp4. This wraps cli.render_overlay to
write the same per-frame items to <output>/forecast.json before rendering. Nothing is recomputed.

    PYTHONPATH=python .venv/bin/python video/run_aeris.py <clip> --output <dir> --profile <p>
"""
import json
import sys
from pathlib import Path

from aeris import cli

_render = cli.render_overlay


def render_and_save(video, out, drawn, fps, *rest):
    frames = {}
    for index, items in drawn.items():
        frames[index] = [[s.track_id, s.lifecycle, bool(s.observed), round(s.center[0], 1), round(s.center[1], 1),
                          None if f is None else round(float(f[0]), 1), None if f is None else round(float(f[1]), 1)]
                         for s, f in items]
    Path(out).parent.joinpath("forecast.json").write_text(json.dumps(frames, separators=(",", ":")))
    return _render(video, out, drawn, fps, *rest)


cli.render_overlay = render_and_save
sys.exit(cli.main())
