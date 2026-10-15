"""Loopback-only web interface for the existing AERIS CLI.

The browser sees only video and run artifacts selected in this session. No labels,
replay detections, or future observations are supplied to the CLI.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Lock, Thread
from urllib.parse import unquote, urlsplit
from uuid import uuid4
import webbrowser

from aeris.ui.samples import BY_SLUG, ROOT, SAMPLES, prepare_sample
from aeris.estimation.camera import Calibration, CameraModel
from aeris.estimation.world_bearing import FixedCameraPose
from aeris.web.live import TILES, LiveSession, load_stages

ASSETS = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 500_000_000
ALLOWED_ARTIFACTS = {"overlay.mp4", "metrics.json", "tracks.csv", "run_manifest.json"}


def browser_preview(path: Path, cache: Path) -> Path:
    """Make a browser-playable local copy when the CLI/source uses another codec."""
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        return path
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() == "h264":
        return path
    fingerprint = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    name = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    preview = cache / f"{name}.mp4"
    if preview.is_file():
        return preview
    temporary = cache / f"{name}.{uuid4().hex}.partial.mp4"
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(temporary)],
            capture_output=True, text=True, timeout=300, check=False,
        )
        if result.returncode != 0 or not temporary.is_file():
            raise RuntimeError(f"Browser preview conversion failed: {result.stderr[-300:]}")
        temporary.replace(preview)
    finally:
        temporary.unlink(missing_ok=True)
    return preview


def try_browser_preview(path: Path, cache: Path) -> tuple[Path | None, str | None]:
    try:
        return browser_preview(path, cache), None
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)


class DemoState:
    def __init__(self, *, camera: CameraModel | None = None, pose: FixedCameraPose | None = None,
                 current_zoom: float | None = None) -> None:
        self.camera, self.pose, self.current_zoom = camera, pose, current_zoom
        self.lock = Lock()
        self.selected_video: Path | None = None
        self.preview_video: Path | None = None
        self.preview_overlay: Path | None = None
        self.selected_name: str | None = None
        self.source_page: str | None = None
        self.run_dir: Path | None = None
        self.status = "idle"
        self.message = "Choose a recorded sample or your own video."
        self.process: subprocess.Popen[str] | None = None
        self.live: LiveSession | None = None
        self.cancel_requested = False
        self.log = ""
        self.cache = Path(tempfile.gettempdir()) / "aeris-web-samples"
        self.uploads = Path(tempfile.gettempdir()) / "aeris-web-uploads"

    def snapshot(self) -> dict:
        with self.lock:
            result = {
                "status": self.status,
                "message": self.message,
                "selected_name": self.selected_name,
                "source_page": self.source_page,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "log": self.log[-6000:],
                "video_url": "/api/video" if self.preview_video else None,
                "overlay_url": "/api/preview/overlay" if self.preview_overlay else None,
                "metrics": None,
                "live": self.live.stats if self.live else None,
            }
            directory = self.run_dir
        if directory and (directory / "metrics.json").is_file():
            try:
                result["metrics"] = json.loads((directory / "metrics.json").read_text())
            except (OSError, ValueError):
                pass
        return result

    def select_sample(self, slug: str) -> None:
        sample = BY_SLUG[slug]
        with self.lock:
            if self.status in {"preparing", "running"}:
                raise RuntimeError("Wait for the current operation or cancel it.")
            self.status = "preparing"
            self.message = f"Preparing {sample.title}…"
            self.selected_video = None
            self.preview_video = None
            self.preview_overlay = None
            self.selected_name = None
            self.run_dir = None
            self.log = ""
            self.source_page = sample.page

        def work() -> None:
            try:
                path = prepare_sample(sample, self.cache)
                preview, warning = try_browser_preview(path, self.cache / "previews")
                with self.lock:
                    self.selected_video = path
                    self.preview_video = preview
                    self.selected_name = sample.title
                    self.status = "ready"
                    self.message = (f"Sample ready. Browser preview unavailable: {warning}"
                                    if warning else "Sample ready. Run analysis to create a new result.")
            except Exception as exc:
                with self.lock:
                    self.status = "failed"
                    self.message = f"Sample unavailable: {exc}"

        Thread(target=work, daemon=True).start()

    def select_upload(self, name: str, stream, size: int) -> None:
        if not 0 < size <= MAX_UPLOAD_BYTES:
            raise ValueError("Video must be between 1 byte and 500 MB.")
        suffix = Path(name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".webm", ".avi", ".ogv"}:
            raise ValueError("Choose an MP4, MOV, WebM, AVI, or OGV video.")
        with self.lock:
            if self.status in {"preparing", "running"}:
                raise RuntimeError("Wait for the current operation or cancel it.")
            self.status = "preparing"
            self.message = "Copying video locally…"
            self.selected_video = None
            self.preview_video = None
            self.preview_overlay = None
            self.selected_name = None
            self.source_page = None
            self.run_dir = None
            self.log = ""
        self.uploads.mkdir(parents=True, exist_ok=True)
        target = self.uploads / f"{uuid4().hex}{suffix}"
        try:
            with target.open("wb") as output:
                remaining = size
                while remaining:
                    chunk = stream.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ValueError("Upload ended before the declared size.")
                    output.write(chunk)
                    remaining -= len(chunk)
            preview, warning = try_browser_preview(target, self.cache / "previews")
            with self.lock:
                self.selected_video = target
                self.preview_video = preview
                self.preview_overlay = None
                self.selected_name = Path(name).name
                self.source_page = None
                self.run_dir = None
                self.log = ""
                self.status = "ready"
                self.message = (f"Video ready. Browser preview unavailable: {warning}"
                                if warning else "Video ready. Accuracy will be unavailable without labels.")
        except Exception as exc:
            target.unlink(missing_ok=True)
            with self.lock:
                self.status = "failed"
                self.message = f"Could not prepare video: {exc}"
            raise

    def run(self) -> None:
        self.stop_live()
        with self.lock:
            if self.status not in {"ready", "complete", "failed", "cancelled"} or not self.selected_video:
                raise RuntimeError("Choose a video before running analysis.")
            if not (ROOT / "models/onnx/yolox_tiny.onnx").is_file():
                raise RuntimeError("Model weights are missing. See models/README.md.")
            if not shutil.which("ffprobe"):
                raise RuntimeError("ffprobe is missing. Install ffmpeg first.")
            video = self.selected_video
            self.run_dir = ROOT / "benchmarks/runs" / (
                "web-demo-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                + "-" + uuid4().hex[:8]
            )
            output = self.run_dir
            self.status = "running"
            self.message = "Running AERIS locally…"
            self.log = ""
            self.preview_overlay = None
            self.cancel_requested = False

        def work() -> None:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
            env["ORT_DISABLE_TELEMETRY"] = "1"
            command = [sys.executable, "-m", "aeris", str(video), "--output", str(output), "--tiles", str(TILES),
                       "--input-note", "Unlabeled local web demonstration; no accuracy claim"]
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                with self.lock:
                    self.process = process
                    if self.cancel_requested:
                        process.terminate()
                stdout, _ = process.communicate()
                preview_overlay = None
                if process.returncode == 0 and (output / "overlay.mp4").is_file():
                    try:
                        preview_overlay = browser_preview(output / "overlay.mp4",
                                                          self.cache / "previews")
                    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                        stdout += f"\nBrowser preview unavailable: {exc}"
                with self.lock:
                    self.log = stdout[-6000:]
                    self.process = None
                    self.preview_overlay = preview_overlay
                    if self.cancel_requested:
                        self.status = "cancelled"
                        self.message = "Run cancelled. No completed result was published."
                    elif process.returncode == 0 and (output / "metrics.json").is_file():
                        self.status = "complete"
                        self.message = "Run complete. Review the overlay and outputs below."
                    else:
                        self.status = "failed"
                        self.message = f"Run failed (exit {process.returncode})."
            except OSError as exc:
                with self.lock:
                    self.process = None
                    self.status = "failed"
                    self.message = f"Could not start AERIS: {exc}"

        Thread(target=work, daemon=True).start()

    def start_live(self, source: str) -> None:
        """``camera:N``, an rtsp/http(s) URL, or ``selected`` (the chosen video, paced real time)."""
        if source.startswith("camera:") and source[7:].isdigit():
            target, paced = int(source[7:]), False
        elif source.startswith(("rtsp://", "http://", "https://")):
            target, paced = source, False
        elif source == "selected":
            with self.lock:
                if not self.selected_video:
                    raise RuntimeError("Choose a video first.")
                target, paced = str(self.selected_video), True
        else:
            raise ValueError("Source must be camera:N, an rtsp/http(s) URL, or selected.")
        with self.lock:
            if self.status == "running":
                raise RuntimeError("A recorded-video analysis is running.")
        self.stop_live()
        try:
            stages = load_stages()
        except Exception as exc:
            raise RuntimeError(f"Cannot load the detector: {exc}") from exc
        session = LiveSession(target, stages, paced=paced, label=source,
                              camera=self.camera, pose=self.pose, current_zoom=self.current_zoom)
        with self.lock:
            self.live = session
        session.start()

    def stop_live(self) -> None:
        with self.lock:
            session = self.live
        if session:
            session.stop()

    def cancel(self) -> None:
        with self.lock:
            if self.status != "running":
                raise RuntimeError("No analysis is running.")
            self.cancel_requested = True
            process = self.process
            self.message = "Cancelling run…"
        if process:
            process.terminate()


def handler_for(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AERISWeb/1.0"

        def log_message(self, format, *args):
            if urlsplit(self.path).path != "/api/state":
                super().log_message(format, *args)

        def _json(self, value: dict, status=HTTPStatus.OK):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _valid_host(self) -> bool:
            port = self.server.server_port
            return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _file(self, path: Path):
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range", "")
            if requested.startswith("bytes="):
                try:
                    first, last = requested[6:].split("-", 1)
                    start = int(first) if first else max(0, size - int(last))
                    end = int(last) if last else size - 1
                    if not 0 <= start <= end < size:
                        raise ValueError
                except ValueError:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
            partial = bool(requested)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = source.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self):
            if not self._valid_host():
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid request host")
                return
            path = urlsplit(self.path).path
            if path == "/api/state":
                self._json(state.snapshot())
            elif path == "/api/samples":
                self._json({"samples": [
                    {"slug": s.slug, "title": s.title, "description": s.description,
                     "author": s.author, "license": s.license, "page": s.page,
                     "observation": s.observation} for s in SAMPLES
                ]})
            elif path == "/api/live/stream":
                with state.lock:
                    session = state.live
                if not session:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for jpeg in session.frames():
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif path == "/api/video":
                with state.lock:
                    video = state.preview_video
                self._file(video) if video else self.send_error(HTTPStatus.NOT_FOUND)
            elif path == "/api/preview/overlay":
                with state.lock:
                    overlay = state.preview_overlay
                self._file(overlay) if overlay else self.send_error(HTTPStatus.NOT_FOUND)
            elif path.startswith("/api/artifact/"):
                name = path.removeprefix("/api/artifact/")
                with state.lock:
                    directory = state.run_dir
                if name in ALLOWED_ARTIFACTS and directory:
                    self._file(directory / name)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            else:
                name = "index.html" if path == "/" else path.lstrip("/")
                if name not in {"index.html", "app.js", "style.css"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._file(ASSETS / name)

        def do_POST(self):
            if not self._valid_host():
                self._json({"error": "Invalid request host."}, HTTPStatus.FORBIDDEN)
                return
            origin = self.headers.get("Origin", "")
            port = self.server.server_port
            allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if origin not in allowed:
                self._json({"error": "Invalid request origin."}, HTTPStatus.FORBIDDEN)
                return
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    raise ValueError("Invalid request length.")
                if path == "/api/upload":
                    if length > MAX_UPLOAD_BYTES:
                        raise ValueError("Video exceeds 500 MB.")
                    name = unquote(self.headers.get("X-Filename", "video.mp4"))
                    state.select_upload(name, self.rfile, length)
                else:
                    if length > 4096:
                        raise ValueError("Request is too large.")
                    data = json.loads(self.rfile.read(length) or b"{}")
                    if path == "/api/select":
                        slug = data.get("slug")
                        if slug not in BY_SLUG:
                            raise ValueError("Unknown sample.")
                        state.select_sample(slug)
                    elif path == "/api/run":
                        state.run()
                    elif path == "/api/cancel":
                        state.cancel()
                    elif path == "/api/live/start":
                        state.start_live(str(data.get("source", "")))
                    elif path == "/api/live/stop":
                        state.stop_live()
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                self._json(state.snapshot())
            except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local AERIS browser demo")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--calibration", type=Path, help="fixed-camera calibration JSON for live bearings")
    parser.add_argument("--pose", type=Path, help="surveyed fixed-camera pose JSON for live bearings")
    parser.add_argument("--current-zoom", type=float, help="operator-asserted current zoom for live bearings")
    args = parser.parse_args(argv)
    try:
        calibration = Calibration.from_dict(json.loads(args.calibration.read_text())) if args.calibration else None
        pose = FixedCameraPose.from_dict(json.loads(args.pose.read_text())) if args.pose else None
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"invalid live calibration or pose: {exc}")
    if (calibration is None) != (pose is None):
        parser.error("--calibration and --pose must be supplied together")
    if args.current_zoom is not None and (not math.isfinite(args.current_zoom) or args.current_zoom <= 0):
        parser.error("--current-zoom must be finite and positive")
    state = DemoState(camera=CameraModel(calibration), pose=pose, current_zoom=args.current_zoom)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(state))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"AERIS web demo: {url}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
