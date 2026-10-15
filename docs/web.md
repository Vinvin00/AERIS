# Local browser demo

Run `PYTHONPATH=python .venv/bin/python -m aeris.web.server` and open
`http://127.0.0.1:8765`. Use `--no-open` to keep the browser closed or `--port 0`
to choose an available loopback port. The server listens only on `127.0.0.1`.
It uses the same AERIS CLI as the Qt app and requires the pinned local model and
`ffprobe` for a new analysis. The page still opens and explains any missing
dependency. No video, checkpoint, or credential is added to Git.

The sample library uses the existing seven-item catalog and its recorded footage
provenance in [ui.md](ui.md). A selected source is retrieved on demand, capped at
35 MB, checksum checked, and excerpted by `ffmpeg` when no local excerpt exists.
A user-selected video is copied into a temporary local folder before analysis;
the 500 MB limit protects the local demo server. Runs are immutable under
`benchmarks/runs/web-demo-*`. The browser can download the overlay, metrics, tracks,
and manifest from the current run; the output path is also shown.
The page makes an H.264 copy in the temporary cache when a source or overlay uses
a codec the browser cannot play. This preview copy does not replace the original
CLI artifact, which remains available for download and provenance checks.

The app displays run counts and null truth-dependent metric reasons from
`metrics.json`. These are unannotated smoke demonstrations, not perception
benchmarks. It never sends labels, future observations, or replay detections to
the runtime. The API is restricted to loopback, checks the request origin for
mutating requests, and exposes only the selected video and allowlisted artifacts.

This is a local web app. It is not a hosted upload service. Closing the server
ends the session; temporary uploaded videos can be deleted from the OS temp
folder after use.

## Live mode

Selecting a recorded clip and pressing **Play with live detection** (or the hero demo
button) runs the pipeline as the video plays; the whole file is not processed first.
**Export full run files** keeps the batch CLI path. A secondary row runs the detector, tracker and 1 s forecast frame by
frame on a camera (`camera:N`), an `rtsp://`/`http(s)://` stream, or the selected clip
played at real time, and streams the annotated feed to the page as MJPEG
(`python/aeris/web/live.py`). Timestamps are seconds since the session started: media
time (frame position / container fps) for a paced clip, so inference jitter does not
appear as motion, and monotonic arrival time for a camera or stream (AERIS-024). Forecasts,
heading and the prediction-error score are computed in scene-relative pixels: camera
motion is estimated from background corners each frame and removed, then forecasts are
mapped back onto the current frame (AERIS-028). If inference is slower than the source, frames are dropped and counted. Live
sessions write no run artifacts and compute no metrics; rates shown are wall-clock on
this machine, not a benchmark. Verified on a paced file only; a physical camera or
network stream has not been exercised.

## Verification and handoff

- Decision: keep the web server on loopback and call the existing recorded-video
  CLI in a child process. Browser previews use temporary H.264 copies; immutable
  CLI artifacts keep their original bytes.
- Verified locally on 2026-10-14: `PYTHONPATH=python .venv/bin/python -m pytest -q`
  passed 252 tests; `.venv/bin/python tools/check_tests.py` passed 252 tests;
  `python3 tools/orchestrate.py lint`, Ruff on the changed Python, `node --check`
  on `app.js`, and `git diff --check` passed. A one-click A380 run completed in
  the browser with a decodable 960×540 overlay, 60 processed frames, 60 detections
  above the untuned threshold, one track ID, and truth-dependent metrics null
  because no labels were supplied. The browser layout had no horizontal overflow
  at 390 px width. These are smoke checks, not perception quality measurements.
- Independent review found and prompted fixes for Host validation, failed-upload
  selection state, preview error handling, keyboard file controls, and the
  distinction between observed and predicted-only track states. Re-review found
  no remaining blocking code issue.
- Unresolved: sample preparation has no cancel action; repeated local uploads and
  preview copies can consume disk until the OS temp folder is cleaned. No hosted
  service or real-data benchmark is provided. The new web slice has no
  orchestration task packet or formal acceptance receipt.
- Exact next action for project coordination: add a separately reviewed web UI
  task to `orchestration/tasks.json`, preserve the current patch as its starting
  implementation, then run `prepare`, `verify`, independent review, and `accept`
  against a fresh baseline. Changed files are `pyproject.toml`,
  `python/aeris/ui/__init__.py`, `python/aeris/web/`, `tests/ui/test_web.py`, and
  this document.
