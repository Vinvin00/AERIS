"""Timestamp-aware video pipeline (AERIS-008).

    decode -> timestamp -> preprocess -> inference -> postprocess -> contract records
                                                                  -> optional overlay

Decoding uses OpenCV's FFmpeg backend (LGPL FFmpeg bundled with opencv-python-headless).
Each decoded frame's time comes from its presentation timestamp: after `read()`, OpenCV
reports FFmpeg's best-effort presentation time of that frame (the frame PTS when the
stream has one) as `CAP_PROP_POS_MSEC`. Timestamps are seconds from the first decoded
frame. That is only a presentation timestamp when the container stores a time for that
frame; otherwise FFmpeg derives one from a nominal frame rate. `pts` mode therefore
accepts only:

- Matroska/WebM, whose block times AERIS reads itself (without FFmpeg) and checks frame
  by frame against the decoded times; files with laced or invisible video blocks, or more
  than one video track, are refused, since frames inside a lace share one stored time;
- MP4/QuickTime, whose sample tables store a duration (and composition offset) for every
  sample; FFmpeg's times come from those tables and are not re-parsed here.

AVI, MPEG-TS (a PTS is required only every 0.7 s), Ogg, raw streams and unknown files
(by magic bytes) are refused. Every delta must be strictly positive; a
zero or negative delta, or a missing timestamp, stops the run (`TimestampError`). The only
fallback is explicit: `--timestamps inferred_fps --fps F` gives `frame_index / F`, and
every frame record says `timestamp_source = "inferred_fps"`.

Each processed frame records decode, preprocess, inference, postprocess and render
wall-clock seconds, `total` (one wall-clock span from the start of its decode to the end
of its processing) and `other` (total minus the named stages: timestamp check, pixel
hash, record building). Throughput is processed frames over the loop's wall time. Frames
decoded but not processed are counted by reason. Outputs, in
the output directory:

- `frames.jsonl`: contract `FrameRecord`s (`annotation_status = "unlabeled"`; the image
  checksum is the SHA-256 of the decoded BGR pixels);
- `detections.jsonl`: contract `DetectionRecord`s in original-frame pixels;
- `timings.json`: per-frame stage times and summaries (warm-up frames excluded);
- `run.json`: input, timestamp policy, counts, detector, environment, code revision,
  output digests, status and any error;
- `overlay.mp4` (with `--render`): processed frames with detections scoring at least 0.3
  drawn (a display choice only), written at a constant rate; the true times are in
  `frames.jsonl`.

    python -m aeris.video.pipeline run INPUT --out DIR [--detector onnx|fake]
        [--timestamps pts|inferred_fps] [--fps F] [--every N] [--max-frames N] [--render]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from aeris.data.contracts import SCHEMA_VERSION, FrameRecord
from aeris.detection import baseline as det

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("decode", "preprocess", "inference", "postprocess", "render", "other", "total")
PTS_CONTAINERS = ("matroska", "mp4")
STORED_TIME_TOLERANCE_MS = 1e-3
TIMESTAMP_MODES = ("pts", "inferred_fps")
OVERLAY_FOURCC = "mp4v"
OVERLAY_MIN_SCORE = 0.3  # display only; detections.jsonl keeps every detection above the detector floor


class PipelineError(RuntimeError):
    """The run cannot continue; the message names the frame and the reason."""


class TimestampError(PipelineError):
    """A frame has no usable timestamp, or time does not strictly increase."""


class TimestampPolicy:
    """Seconds from the first decoded frame, validated frame by frame.

    Called with every decoded frame, including frames that are then skipped, so the whole
    stream's timing is checked.
    """

    def __init__(self, mode: str, fps: float | None = None):
        if mode not in TIMESTAMP_MODES:
            raise PipelineError(f"timestamps: expected one of {TIMESTAMP_MODES}, got {mode!r}")
        if mode == "inferred_fps":
            if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
                raise PipelineError("inferred_fps timestamps need an explicit finite fps > 0")
        elif fps is not None:
            raise PipelineError("fps is only used with inferred_fps timestamps")
        self.mode, self.fps = mode, fps
        self.first_ms: float | None = None
        self.previous: float | None = None
        self.deltas: list[float] = []

    @property
    def source(self) -> str:
        return "decode_pts" if self.mode == "pts" else "inferred_fps"

    def __call__(self, index: int, pts_ms: float | None) -> float:
        if self.mode == "inferred_fps":
            timestamp = index / self.fps
        else:
            if pts_ms is None or not math.isfinite(pts_ms) or pts_ms < 0:
                raise TimestampError(f"frame {index}: no usable presentation timestamp ({pts_ms!r}); "
                                     "use --timestamps inferred_fps --fps F to state a frame rate")
            if self.first_ms is None:
                self.first_ms = pts_ms
            timestamp = (pts_ms - self.first_ms) / 1000.0
        if self.previous is not None:
            delta = timestamp - self.previous
            if not delta > 0:
                raise TimestampError(f"frame {index}: timestamp delta {delta * 1000:.3f} ms is not "
                                     f"positive ({self.previous:.6f} s -> {timestamp:.6f} s)")
            self.deltas.append(delta)
        self.previous = timestamp
        return timestamp

    def summary(self) -> dict:
        deltas = self.deltas
        return {"mode": self.mode, "source": self.source, "fps": self.fps,
                "first_pts_ms": self.first_ms, "deltas": len(deltas),
                "delta_s_min": min(deltas) if deltas else None,
                "delta_s_median": float(np.median(deltas)) if deltas else None,
                "delta_s_max": max(deltas) if deltas else None}


# ---------------------------------------------------------------- detectors

class OnnxDetector:
    """The AERIS-006 baseline (configs/detector/baseline.json) split into stages."""

    def __init__(self, threads: int = 1):
        config = det.load_config()
        self.settings = det.settings_from_config(config)
        export = config["export"]["aeris_export"]
        onnx = det.require_artifact(ROOT / export["path"], export["sha256"])
        self.model = det.OnnxModel(onnx, self.settings, threads=threads)
        self.provenance = det.provenance(config, f"onnxruntime CPUExecutionProvider threads={threads}")
        self.info = {"kind": "onnx", "onnx_sha256": export["sha256"], "providers": self.model.providers,
                     "threads": threads, "precision": "float32",
                     "input_size": [self.settings.input_width, self.settings.input_height]}

    def preprocess(self, image):
        return det.preprocess(image, self.settings)

    def infer(self, tensor):
        return np.asarray(self.model(tensor))

    def postprocess(self, raw, letterbox):
        return det.postprocess(raw, letterbox, self.settings)


class FakeDetector:
    """SYNTHETIC deterministic stand-in: no model and no pixels read.

    For the k-th processed frame it returns one 20 x 10 px aircraft box at
    x = 37k mod (W - 19), y = 23k mod (H - 9) with score 0.5 + (k mod 5) / 10, except that
    every fourth frame (k mod 4 == 3) has no detection. `fail_at=k` raises at frame k.
    """

    def __init__(self, fail_at: int | None = None):
        self.settings = det.settings_from_config(det.load_config())
        self.provenance = "aeris-008 fake detector (SYNTHETIC, deterministic)"
        self.info = {"kind": "fake", "fail_at": fail_at}
        self.calls, self.fail_at = 0, fail_at

    def preprocess(self, image):
        height, width = image.shape[:2]
        return None, det.Letterbox(width, height, width, height)

    def infer(self, tensor):
        k, self.calls = self.calls, self.calls + 1
        if self.fail_at is not None and k == self.fail_at:
            raise RuntimeError(f"fake detector failure at processed frame {k}")
        return k

    def postprocess(self, k, letterbox):
        if k % 4 == 3:
            return []
        bw, bh = min(20, letterbox.width), min(10, letterbox.height)
        x = (37 * k) % (letterbox.width - bw + 1)
        y = (23 * k) % (letterbox.height - bh + 1)
        return [det.Detection((float(x), float(y), float(x + bw), float(y + bh)),
                              0.5 + (k % 5) / 10, 0, 4, k)]


# ---------------------------------------------------------------- decode and render

def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def container_format(path: Path) -> str:
    """The container named by the file's magic bytes (`unknown` if none matches)."""
    with open(path, "rb") as handle:
        head = handle.read(377)
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska"
    if head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        return "mp4"
    if len(head) == 377 and head[0] == head[188] == head[376] == 0x47:
        return "mpegts"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if head[:4] == b"OggS":
        return "ogg"
    return "unknown"


# Matroska element IDs (with their length markers) used by matroska_timeline.
_EBML, _SEGMENT, _INFO, _TIMECODE_SCALE = 0x1A45DFA3, 0x18538067, 0x1549A966, 0x2AD7B1
_TRACKS, _TRACK_ENTRY, _TRACK_NUMBER, _TRACK_TYPE = 0x1654AE6B, 0xAE, 0xD7, 0x83
_CLUSTER, _CLUSTER_TIME, _SIMPLE_BLOCK, _BLOCK_GROUP, _BLOCK = 0x1F43B675, 0xE7, 0xA3, 0xA0, 0xA1
_SEGMENT_CHILDREN = {0x114D9B74, _INFO, _TRACKS, _CLUSTER, 0x1C53BB6B, 0x1941A469, 0x1043A770,
                     0x1254C367, 0xEC, 0xBF}


def _vint(handle, keep_marker: bool = False) -> tuple[int, int]:
    first = handle.read(1)
    if not first:
        raise EOFError
    length = 9 - first[0].bit_length()
    if first[0] == 0 or length > 8:
        raise PipelineError("Matroska: invalid variable-length integer")
    rest = handle.read(length - 1)
    if len(rest) != length - 1:
        raise EOFError
    value = first[0] if keep_marker else first[0] & ((1 << (8 - length)) - 1)
    for byte in rest:
        value = (value << 8) | byte
    return value, length


def _element(handle):
    """(id, size or None if unknown, body offset) of the element at the current offset."""
    element, _ = _vint(handle, keep_marker=True)
    size, length = _vint(handle)
    return element, (None if size == (1 << (7 * length)) - 1 else size), handle.tell()


def _uint(handle, offset: int, size: int) -> int:
    handle.seek(offset)
    data = handle.read(size)
    if len(data) != size or size > 8:
        raise PipelineError("Matroska: truncated or oversized integer element")
    return int.from_bytes(data, "big")


def _children(handle, start: int, end: int):
    offset = start
    while offset < end:
        handle.seek(offset)
        element, size, body = _element(handle)
        if size is None:
            raise PipelineError("Matroska: unknown-size element inside a sized element")
        yield element, size, body
        offset = body + size


def matroska_timeline(path: Path) -> dict:
    """Stored presentation times (ms, ascending) of the one video track's blocks.

    Read directly from the file, without FFmpeg. Refuses files AERIS cannot map one stored
    time to each decoded frame for: no or several video tracks, laced blocks (frames in a
    lace share one time), invisible blocks (not output as frames). A truncated tail ends
    the scan; the decoder stops there too.
    """
    total = path.stat().st_size
    scale, tracks, blocks = 1_000_000, {}, []

    def block(handle, body, cluster_time):
        if cluster_time is None:
            raise PipelineError("Matroska: block before its cluster time")
        handle.seek(body)
        track, _ = _vint(handle)
        head = handle.read(3)
        if len(head) != 3:
            raise EOFError
        blocks.append((track, cluster_time + int.from_bytes(head[:2], "big", signed=True), head[2]))

    with open(path, "rb") as handle:
        try:
            element, size, body = _element(handle)
            if element != _EBML or size is None:
                raise PipelineError("Matroska: missing EBML header")
            handle.seek(body + size)
            element, size, body = _element(handle)
            if element != _SEGMENT:
                raise PipelineError("Matroska: missing Segment")
            end = total if size is None else min(total, body + size)
            offset = body
            while offset < end:
                handle.seek(offset)
                element, size, body = _element(handle)
                if element == _CLUSTER:
                    cluster_end = None if size is None else body + size
                    cluster_time, child = None, body
                    while cluster_end is None or child < cluster_end:
                        handle.seek(child)
                        try:
                            kind, csize, cbody = _element(handle)
                        except EOFError:
                            child = total
                            break
                        if cluster_end is None and kind in _SEGMENT_CHILDREN:
                            break  # an unknown-size cluster ends at the next top-level element
                        if csize is None:
                            raise PipelineError("Matroska: unknown-size element inside a cluster")
                        if kind == _CLUSTER_TIME:
                            cluster_time = _uint(handle, cbody, csize)
                        elif kind == _SIMPLE_BLOCK:
                            block(handle, cbody, cluster_time)
                        elif kind == _BLOCK_GROUP:
                            for inner, isize, ibody in _children(handle, cbody, cbody + csize):
                                if inner == _BLOCK:
                                    block(handle, ibody, cluster_time)
                        child = cbody + csize
                    offset = child if cluster_end is None else cluster_end
                    continue
                if size is None:
                    raise PipelineError("Matroska: unknown-size element other than a cluster")
                if element == _INFO:
                    for inner, isize, ibody in _children(handle, body, body + size):
                        if inner == _TIMECODE_SCALE:
                            scale = _uint(handle, ibody, isize)
                elif element == _TRACKS:
                    for inner, isize, ibody in _children(handle, body, body + size):
                        if inner == _TRACK_ENTRY:
                            fields = {k: _uint(handle, b, n) for k, n, b in _children(handle, ibody, ibody + isize)
                                      if k in (_TRACK_NUMBER, _TRACK_TYPE)}
                            tracks[fields.get(_TRACK_NUMBER)] = fields.get(_TRACK_TYPE)
                offset = body + size
        except EOFError:
            pass  # truncated file: keep what was read
    video = [number for number, kind in tracks.items() if kind == 1]
    if len(video) != 1 or video[0] is None:
        raise PipelineError(f"Matroska: expected exactly one video track, found {len(video)}")
    own = [(time, flags) for track, time, flags in blocks if track == video[0]]
    if any(flags & 0x06 for _, flags in own):
        raise PipelineError("Matroska: laced video blocks; frames inside a lace have no stored time of their own")
    if any(flags & 0x08 for _, flags in own):
        raise PipelineError("Matroska: invisible video blocks; stored and decoded frames would not align")
    return {"track": video[0], "timecode_scale_ns": scale,
            "times_ms": sorted(time * scale / 1e6 for time, _ in own)}


def open_video(path: Path):
    """An OpenCV FFmpeg capture and the container's reported properties."""
    if not path.is_file():
        raise PipelineError(f"{path}: no such file")
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise PipelineError(f"{path}: OpenCV's FFmpeg backend cannot open it as video")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    info = {"path": str(path), "name": path.name, "container": container_format(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size, "backend": cap.getBackendName(),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "nominal_fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or None}
    return cap, info


class Overlay:
    """Processed frames with detections drawn, at a constant rate (for viewing only)."""

    def __init__(self, path: Path, fps: float, width: int, height: int,
                 min_score: float = OVERLAY_MIN_SCORE):
        self.path, self.fps, self.size, self.frames = path, fps, (width, height), 0
        self.min_score = min_score
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*OVERLAY_FOURCC), fps, self.size)
        if not self.writer.isOpened():
            raise PipelineError(f"{path}: cannot open an {OVERLAY_FOURCC} video writer")

    def write(self, image, records, timestamp_s: float) -> None:
        canvas = image.copy()
        for record in records:
            if record.score < self.min_score:
                continue
            b = record.bbox
            cv2.rectangle(canvas, (int(b.x_min), int(b.y_min)), (int(math.ceil(b.x_max)) - 1,
                          int(math.ceil(b.y_max)) - 1), (0, 255, 0), 2)
            cv2.putText(canvas, f"{record.score:.2f}", (int(b.x_min), max(12, int(b.y_min) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, f"t={timestamp_s:.3f}s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        self.writer.write(canvas)
        self.frames += 1

    def close(self) -> dict:
        self.writer.release()
        return {"path": self.path.name, "codec": OVERLAY_FOURCC, "fps": self.fps,
                "width": self.size[0], "height": self.size[1], "frames": self.frames,
                "min_score_drawn": self.min_score}


# ---------------------------------------------------------------- run

def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": int(arr.size), "mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max())}


def environment() -> dict:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        import onnxruntime
        versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        versions["onnxruntime"] = None
    return {"platform": platform.platform(), "cpu": cpu, **versions}


def code_revision() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD").strip() or None,
            "dirty_paths": [line[3:] for line in status.splitlines() if len(line) > 3]}


def _jsonl(path: Path, records) -> None:
    path.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")


def run(path, detector, out_dir, *, timestamps: str = "pts", fps: float | None = None, every: int = 1,
        max_frames: int | None = None, render: bool = False, sequence_id: str | None = None,
        warmup_frames: int = 1, clock=time.perf_counter, argv=None) -> dict:
    """Decode, time and detect every `every`-th frame; write the outputs; return run.json."""
    if type(every) is not int or every < 1:
        raise PipelineError("every: expected an integer >= 1")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise PipelineError("max_frames: expected an integer >= 1 or None")
    if type(warmup_frames) is not int or warmup_frames < 0:
        raise PipelineError("warmup_frames: expected an integer >= 0")
    path, out_dir = Path(path), Path(out_dir)
    policy = TimestampPolicy(timestamps, fps)
    if not path.is_file():
        raise PipelineError(f"{path}: no such file")
    container = container_format(path)
    if policy.mode == "pts" and container not in PTS_CONTAINERS:
        raise PipelineError(f"{path.name}: a {container} file does not store a presentation time "
                            "for every frame (FFmpeg would derive missing ones from a nominal frame "
                            "rate); use --timestamps inferred_fps --fps F to state a frame rate explicitly")
    stored = None
    if policy.mode == "pts" and container == "matroska":
        try:
            timeline = matroska_timeline(path)
        except PipelineError as exc:
            raise PipelineError(f"{path.name}: {exc}; use --timestamps inferred_fps --fps F to state "
                                "a frame rate explicitly") from exc
        stored = [t - timeline["times_ms"][0] for t in timeline["times_ms"]] if timeline["times_ms"] else []
    cap, info = open_video(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequence = sequence_id or re.sub(r"\s+", "_", path.stem)
    frames, detections, per_frame, decode_all = [], [], [], []
    skipped: Counter = Counter()
    overlay, overlay_info, status, error, failure = None, None, "ok", None, None
    decoded, stopped_at_max, reference, index = 0, False, None, None
    loop_start = clock()
    try:
        while True:
            frame_start = clock()
            ok, image = cap.read()
            decode_s = clock() - frame_start
            if not ok:
                break
            index = decoded
            decoded += 1
            if reference is None:
                reference = image.shape
            elif image.shape != reference:
                raise PipelineError(f"frame {index}: size {image.shape} differs from the first frame {reference}")
            timestamp = policy(index, cap.get(cv2.CAP_PROP_POS_MSEC))
            if stored is not None:
                if index >= len(stored):
                    raise TimestampError(f"frame {index}: the decoder produced more frames than the "
                                         f"container stores ({len(stored)})")
                if abs(timestamp * 1000 - stored[index]) > STORED_TIME_TOLERANCE_MS:
                    raise TimestampError(f"frame {index}: decoded time {timestamp * 1000:.3f} ms differs "
                                         f"from the stored block time {stored[index]:.3f} ms")
            decode_all.append(decode_s)
            if index % every:
                skipped["stride"] += 1
                continue
            height, width = image.shape[:2]
            frame = FrameRecord(
                SCHEMA_VERSION, sequence, index, timestamp, policy.source, width, height,
                f"{path.name}#frame={index}", hashlib.sha256(image.tobytes()).hexdigest(), None,
                "unlabeled", {"camera_id": "not recorded for this input"})
            start = clock()
            tensor, letterbox = detector.preprocess(image)
            pre_s = clock() - start
            start = clock()
            raw = detector.infer(tensor)
            inf_s = clock() - start
            start = clock()
            found = detector.postprocess(raw, letterbox)
            post_s = clock() - start
            records = det.to_records(found, frame, letterbox, detector.settings, detector.provenance)
            render_s = None
            if render:
                start = clock()
                if overlay is None:
                    rate = (info["nominal_fps"] / every if info["nominal_fps"] > 0 else 25.0)
                    overlay = Overlay(out_dir / "overlay.mp4", rate, width, height)
                overlay.write(image, records, timestamp)
                render_s = clock() - start
            frames.append(frame)
            detections.extend(records)
            total_s = clock() - frame_start
            per_frame.append({"frame_index": index, "timestamp_s": timestamp, "decode": decode_s,
                              "preprocess": pre_s, "inference": inf_s, "postprocess": post_s,
                              "render": render_s,
                              "other": total_s - (decode_s + pre_s + inf_s + post_s + (render_s or 0.0)),
                              "total": total_s, "detections": len(records)})
            if max_frames is not None and len(frames) >= max_frames:
                stopped_at_max = True
                break
    except PipelineError as exc:
        status, error, failure = "failed", str(exc), exc
    except Exception as exc:  # detector or contract failure: record where it happened
        status, error = "failed", f"frame {index}: {type(exc).__name__}: {exc}"
        failure = PipelineError(error)
        failure.__cause__ = exc
    finally:
        cap.release()
        if overlay is not None:
            overlay_info = overlay.close()
    loop_wall_s = clock() - loop_start
    warnings = []
    if (status == "ok" and not stopped_at_max and info["frame_count_reported"] > 0
            and decoded < info["frame_count_reported"]):
        warnings.append(f"stream ended after {decoded} frames; the container reports "
                        f"{info['frame_count_reported']}")
    kept = per_frame[warmup_frames:]
    timings = {"clock": "time.perf_counter wall seconds", "warmup_frames_excluded": min(warmup_frames, len(per_frame)),
               "percentiles": "numpy.percentile, linear interpolation",
               "summary": {stage: summarize([f[stage] for f in kept if f[stage] is not None]) for stage in STAGES},
               "decode_all_decoded_frames": summarize(decode_all),
               "loop_wall_s": loop_wall_s,
               "processed_frames_per_second": (len(per_frame) / loop_wall_s
                                               if per_frame and loop_wall_s > 0 else None),
               "throughput_note": "processed frames / loop wall time, including warm-up, decoding "
                                  "of stride-skipped frames and writing the overlay",
               "per_frame": per_frame}
    _jsonl(out_dir / "frames.jsonl", frames)
    _jsonl(out_dir / "detections.jsonl", detections)
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = {name: file_sha256(out_dir / name) for name in ("frames.jsonl", "detections.jsonl", "timings.json")}
    if overlay_info is not None:
        outputs["overlay.mp4"] = file_sha256(out_dir / "overlay.mp4")
    checked = {"matroska": ("every decoded frame against the stored block times, read without FFmpeg"
                            if stored is not None else None),
               "mp4": "FFmpeg's times from the MP4 sample tables (stored per sample; not re-parsed)"}
    summary = {"schema_version": 1, "task": "AERIS-008", "status": status, "error": error,
               "input": info, "sequence_id": sequence,
               "timestamps": {**policy.summary(),
                              "stored_times_checked": (checked.get(container) if policy.mode == "pts" else None)},
               "counts": {"decoded": decoded, "processed": len(frames), "skipped": dict(skipped),
                          "stopped_at_max_frames": stopped_at_max, "detections": len(detections)},
               "settings": {"every": every, "max_frames": max_frames, "render": render,
                            "warmup_frames": warmup_frames},
               "detector": {**detector.info, "provenance": detector.provenance},
               "overlay": overlay_info, "warnings": warnings, "environment": environment(),
               "code": code_revision(), "argv": argv, "outputs": outputs}
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="decode, time and detect one video")
    go.add_argument("input", type=Path)
    go.add_argument("--out", type=Path, required=True, help="output directory (not in Git)")
    go.add_argument("--detector", choices=("onnx", "fake"), default="onnx")
    go.add_argument("--threads", type=int, default=1)
    go.add_argument("--timestamps", choices=TIMESTAMP_MODES, default="pts")
    go.add_argument("--fps", type=float, default=None, help="only with --timestamps inferred_fps")
    go.add_argument("--every", type=int, default=1, help="process every N-th decoded frame")
    go.add_argument("--max-frames", type=int, default=None)
    go.add_argument("--render", action="store_true", help="also write overlay.mp4")
    args = parser.parse_args(argv)
    try:
        detector = OnnxDetector(args.threads) if args.detector == "onnx" else FakeDetector()
        summary = run(args.input, detector, args.out, timestamps=args.timestamps, fps=args.fps,
                      every=args.every, max_frames=args.max_frames, render=args.render,
                      argv=sys.argv if argv is None else ["aeris.video.pipeline", *argv])
    except (PipelineError, det.DetectorError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = summary["counts"]
    print(f"ok  decoded {counts['decoded']}  processed {counts['processed']}  "
          f"detections {counts['detections']}  timestamps {summary['timestamps']['source']}")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
