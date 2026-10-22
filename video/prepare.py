"""Copy 30 fps clips and AERIS output into the Remotion project.

Run `video/run_aeris.py` per clip first (see README.md). Per frame, only tracks the tracker marks
observed (a real detection that frame) are exported. Forecast arrows are the ones AERIS's own
overlay draws (forecast.json, gated per ADR 026); nothing is recomputed here.
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build" / "demo-video-work"
HERE = Path(__file__).parent
RUNS = {"formation": "v3-formation", "red-arrows": "v3-red-arrows", "air-force-one": "v3-air-force-one"}


def num(v, nd=1):
    return None if v in ("", None) else round(float(v), nd)


for clip, run_name in RUNS.items():
    run = WORK / "runs" / run_name
    metrics = json.loads((run / "metrics.json").read_text())
    bench = json.loads((run / "benchmark.json").read_text())
    drawn = json.loads((run / "forecast.json").read_text())
    n = metrics["counts"]["frames_processed"]
    frames = [[] for _ in range(n)]
    for r in csv.DictReader((run / "tracks.csv").open()):
        if r["observed"] != "1":
            continue
        i = int(r["frame_index"])
        arrow = next((d for d in drawn.get(str(i), []) if d[0] == r["track_id"]), None)
        frames[i].append([
            r["track_id"], *(num(r[k]) for k in ("x_min_px", "y_min_px", "x_max_px", "y_max_px", "cx_px", "cy_px")),
            num(r["speed_px_s"], 0), num(r["image_heading_deg"], 0),
            None if arrow is None else arrow[5], None if arrow is None else arrow[6],
        ])
    one_s = metrics["forecast_vs_detections_displayed"]["models"]["constant_velocity"]["1.0"]
    stats = {
        "frames": n,
        "detections": metrics["counts"]["detections_at_or_above_min_score"],
        "arrows": sum(1 for f in drawn.values() for d in f if d[5] is not None),
        "inferMs": round(bench["stages"]["inference"]["p50_s"] * 1000),
        "fcErrPx": num(one_s["median_error_px"], 0),
        "fcDirDeg": num(one_s["median_direction_error_deg"], 1),
        "fcScored": one_s["scored"],
    }
    (HERE / "src" / "data").mkdir(parents=True, exist_ok=True)
    (HERE / "src" / "data" / f"{clip}.json").write_text(json.dumps({"frames": frames, "stats": stats}, separators=(",", ":")))
    (HERE / "public").mkdir(exist_ok=True)
    shutil.copy(WORK / "clips" / f"{clip}.mp4", HERE / "public" / f"{clip}.mp4")
    print(f"{clip} <- {run_name}: {sum(1 for f in frames if f)}/{n} frames with a detection, {stats}")
