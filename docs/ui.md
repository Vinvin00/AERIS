# Local desktop demo

The app opens on a dark analysis-console landing page that explains what AERIS does and how to try it.
**Open sample library** opens the recorded sample picker; **Use my video**
opens a local file chooser. The second screen keeps preview, run controls, status,
and results together so a first-time visitor can follow the whole demo. On smaller
desktop windows the workspace scrolls vertically; the sample picker and run controls
remain visible at 900×600. The output panel appears when a run begins or a saved
result is loaded. Its counts are runtime outputs, not measured accuracy.

Run the Qt app with `PYTHONPATH=python .venv/bin/python -m aeris.ui.app` after
installing the `desktop` extra (`.venv/bin/python -m pip install -e '.[desktop]'`).
The pinned local detector weights and `ffmpeg`/`ffprobe` are required to run a new
analysis; see `models/README.md`. The app opens without them and explains what is
missing. It is a local desktop app, not a hosted web UI or an upload service.

The picker lists seven **recorded**, unlabeled excerpts. On this checkout it uses
the existing Git-ignored smoke-test clips and overlays under
`benchmarks/runs/real-media-smoke-2026-10-13/`. On another machine, **Use sample**
downloads the original from Wikimedia Commons only when selected, limits each source
to 35 MB, verifies the published SHA-1, and uses `ffmpeg` to make a 10 fps excerpt in
the app cache. A source is never accepted after a checksum mismatch. A saved local
overlay is a preview of the prior local run; **Run AERIS** generates a new immutable
run under `benchmarks/runs/`. User-selected videos stay local. The UI never supplies
annotations, replay detections, or future observations to the runtime.

| Picker item | Source and attribution | Terms | Local smoke observation |
|---|---|---|---|
| Formation | [Lucastphotography](https://commons.wikimedia.org/wiki/File:USAF_Blue_Angels_in_close_formation.webm) | CC BY 4.0 | Four tracks |
| C-5 320×240 | [US Air Force](https://commons.wikimedia.org/wiki/File:C-5_Galaxy.ogv) | US public domain | Aircraft detected; identity split |
| HKIA later and earlier | [IamCristYe](https://commons.wikimedia.org/wiki/File:Airplane_Landing_at_South_Runway,_HKIA_20210215.webm) | CC BY-SA 4.0 | Later detected, earlier missed |
| A380 | [LeoTor](https://commons.wikimedia.org/wiki/File:A380_Taking_off_from_YYZ.webm) | CC BY 4.0 | Detected |
| Ferry gulls | [Subhashish Panigrahi](https://commons.wikimedia.org/wiki/File:Birds_flying_behind_a_ferry.webm) | CC BY-SA 3.0 | One brief bird false positive |
| Bird flock | [Earthwild TV](https://commons.wikimedia.org/wiki/File:Graceful_Birds_in_the_Sky-Amazing_Bird_Sound-User_FREE_Videos.webm) | CC BY 3.0 | Several brief bird false positives |

Credit the author, link the source and license, and identify modifications in any
published edit. Share-alike terms apply to adaptations of the BY-SA clips. No video,
checkpoint, or credential is committed to Git. The catalog in
`python/aeris/ui/samples.py` carries source URLs and checksums; local observations
and artifact hashes are in the Git-ignored
`benchmarks/runs/real-media-smoke-2026-10-13/validation_report.json`.

These are unannotated visual smoke tests. Detection counts are not precision or
recall. AP, HOTA, IDF1, and forecast accuracy remain unavailable without complete
ground truth. The bird clips cannot substantiate a claim that birds are always
rejected. The detector's operating threshold is the untuned CLI default.

The child process keeps analysis off the UI thread. **Cancel** terminates a run;
each successful run gets a new output folder. Failure text stays visible. Preview
decoding uses OpenCV, while all timing and outputs come from the CLI. A fresh run
can differ from the saved local smoke artifact if code, model or environment changes.
