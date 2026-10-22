#!/usr/bin/env python3
"""Build the hosted, pre-recorded demo site: real pipeline runs, no backend.

    PYTHONPATH=python .venv/bin/python tools/build_static_demo.py [--out build/demo-site]

Downloads each clip from Wikimedia Commons (checksum-checked by aeris.ui.samples), runs
`python -m aeris` once per clip, and copies the browser-playable results next to the
console files. Media stays out of Git (`build/` is ignored); deploy the output folder.

The hosted demo runs with the opt-in AERIS-033 tracker config `manoeuvre-a3000-s0-full` (owner
decision, 2026-10-22): it keeps camera-followed jets on far fewer track IDs (F-35 clip: 55 -> 6),
but failed the full QUT held-back check (ADR 034), so the CLI and local web default stay
`baseline.json`. The page states this. Runs are cached per tracker config under the work folder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from aeris.detection.profiles import PROFILES  # noqa: E402
from aeris.ui.samples import prepare_sample  # noqa: E402
from aeris.web.demo_clips import BY_SLUG  # noqa: E402
from aeris.web.server import ASSETS, browser_preview  # noqa: E402

SLUGS = ["formation", "hkia-late", "tokyo-small", "b52-dubai", "f35-airshow",
         "red-arrows", "air-force-one", "spitfire"]
PROFILE = "fast-zoom"
# The hosted demo opts into the AERIS-033 tracker; the CLI and local web default stay baseline.json.
TRACKER_CONFIG = Path("configs/tracker/manoeuvre-a3000-s0-full.json")
TRACKER_NOTE = ("Tracker: manoeuvre-a3000-s0-full, an opt-in setting that keeps a fast, camera-followed "
                "aircraft on far fewer track IDs. On labelled QUT aircraft video it passed the tuning clips "
                "but failed the full held-back check (ADR 034), so it is not the AERIS default.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build/demo-site")
    parser.add_argument("--work", type=Path, default=ROOT / "build/demo-work")
    parser.add_argument("--slugs", nargs="+", default=SLUGS, choices=sorted(BY_SLUG))
    args = parser.parse_args()
    cache = args.work / "cache"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "python"), "ORT_DISABLE_TELEMETRY": "1"}
    shutil.rmtree(args.out, ignore_errors=True)
    samples = []
    for slug in args.slugs:
        sample = BY_SLUG[slug]
        print(f"[{slug}] preparing", flush=True)
        clip = prepare_sample(sample, cache)
        run = args.work / "runs" / TRACKER_CONFIG.stem / slug   # keyed by tracker: never reuse other runs
        if not (run / "metrics.json").is_file():
            shutil.rmtree(run, ignore_errors=True)
            subprocess.run([sys.executable, "-m", "aeris", str(clip), "--output", str(run), "--profile", PROFILE,
                            "--tracker-config", str(TRACKER_CONFIG),
                            "--input-note", "Unlabeled hosted demonstration; no accuracy claim"],
                           cwd=ROOT, env=env, check=True)
        target = args.out / "demo" / slug
        target.mkdir(parents=True)
        shutil.copy(browser_preview(clip, cache / "previews"), target / "original.mp4")
        shutil.copy(browser_preview(run / "overlay.mp4", cache / "previews"), target / "overlay.mp4")
        for name in ("metrics.json", "tracks.csv", "run_manifest.json"):
            shutil.copy(run / name, target / name)
        counts = json.loads((run / "metrics.json").read_text())["counts"]
        samples.append({
            "slug": slug, "title": sample.title, "description": sample.description, "author": sample.author,
            "license": sample.license, "page": sample.page,
            "observation": (f"{counts['detections_at_or_above_min_score']} detections in "
                            f"{counts['frames_processed']} frames, {counts['tracks_created']} track "
                            f"{'ID' if counts['tracks_created'] == 1 else 'IDs'} "
                            f"(one offline run, {PROFILE} profile, {TRACKER_CONFIG.stem} tracker)."),
        })
    (args.out / "demo/manifest.json").write_text(
        json.dumps({"samples": samples, "profile_note": PROFILES[PROFILE].note,
                    "tracker_config": TRACKER_CONFIG.name, "tracker_note": TRACKER_NOTE}, indent=1))
    for name in ("app.js", "style.css", "static-api.js"):
        shutil.copy(ASSETS / name, args.out / name)
    page = (ASSETS / "index.html").read_text()
    for old, new in [
        ('<html lang="en">', '<html lang="en" data-static>'),
        ('<script src="/app.js" defer></script>', '<script src="/static-api.js" defer></script>\n  <script src="/app.js" defer></script>'),
        ("Runs on your machine. One click to try.",
         "Pre-recorded results from the real pipeline. One click to try. " + TRACKER_NOTE),
        ("Drop in your own clip", "Run it yourself"),
    ]:
        assert old in page, old
        page = page.replace(old, new)
    (args.out / "index.html").write_text(page)
    print(f"built {args.out} ({sum(f.stat().st_size for f in args.out.rglob('*') if f.is_file()) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
