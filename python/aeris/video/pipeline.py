"""Timestamp-aware recorded-video pipeline: decode -> detect stages -> optional overlay.

Timestamps come from container presentation times (via ``ffprobe``) when available.
Otherwise, if the policy allows it, they are inferred from the nominal frame rate and
marked ``inferred_fps``. Zero, negative, non-finite or mixed-source timestamps fail.
Stage timings are wall-clock ``perf_counter`` seconds on this machine, not a benchmark.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from aeris.data.contracts import BBox, DetectionRecord

STAGES = ("decode", "preprocess", "inference", "postprocess", "render", "total")
POLICIES = ("pts", "pts_or_inferred")


class VideoError(ValueError):
    """Input video, timestamps or detector output violate the pipeline contract."""


class Stages(Protocol):
    """Three detector stages so each can be timed separately."""
    name: str

    def preprocess(self, image_bgr: np.ndarray): ...
    def infer(self, prepared): ...
    def postprocess(self, raw, prepared, *, width: int, height: int, sequence_id: str,
                    frame_index: int, timestamp_s: float) -> list[DetectionRecord]: ...


class BaselineStages:
    """The AERIS-006 YOLOX baseline split into preprocess/infer/postprocess."""

    def __init__(self, config: dict, runner, *, tiles: int = 1, gate: float | None = None):
        """``tiles=N`` adds an N x N grid of overlapping crops to the full-frame pass, so
        small, distant aircraft keep enough pixels at the model input. Costs N*N + 1
        forward passes per frame; tile and full-frame boxes are merged by NMS.

        ``gate`` makes tiling a cascade: the full frame runs first and the tiles run only
        when it found no aircraft scoring >= ``gate``. Keep ``gate`` >= the operating
        min-score, or weak full-frame hits suppress the zoom and distant aircraft are lost.
        A confident large aircraft also suppresses the zoom, so a small one beside it can
        be missed; use ``gate=None`` when that matters more than speed."""
        from aeris.detection import baseline
        self._b, self.config, self.runner, self.tiles, self.gate = baseline, config, runner, tiles, gate
        self.name = (f"{config['model']['name']}/{runner.name}" + (f"+tiles{tiles}x{tiles}" if tiles > 1 else "")
                     + (f"+cascade{gate}" if tiles > 1 and gate is not None else ""))

    def _crops(self, height, width):
        yield 0, 0, height, width
        if self.tiles > 1:
            n = self.tiles
            th, tw = int(height / n * 1.25), int(width / n * 1.25)  # 25% overlap so edge-cut aircraft appear whole once
            if max(th, tw) <= self.config["input"]["size"]:
                return  # frame already fits the model input; tiles would only upscale
            for i in range(n):
                for j in range(n):
                    y0, x0 = round(i * (height - th) / (n - 1)), round(j * (width - tw) / (n - 1))
                    yield y0, x0, th, tw

    def preprocess(self, image_bgr):
        size, pad = self.config["input"]["size"], self.config["input"]["pad_value"]
        if self.tiles == 1:
            return self._b.preprocess(image_bgr, size, pad)
        return [(y0, x0, h, w, *self._b.preprocess(image_bgr[y0:y0 + h, x0:x0 + w], size, pad))
                for y0, x0, h, w in self._crops(*image_bgr.shape[:2])]

    def infer(self, prepared):
        if self.tiles == 1:
            return self.runner(prepared[0])
        raws = [self.runner(prepared[0][4])]    # crop 0 is the full frame
        if self.gate is not None and len(prepared) > 1:
            _, _, h, w, _, ratio = prepared[0]
            if any(f[2] >= self.gate for f in self._b.postprocess(raws[0], ratio, w, h, self.config)):
                return raws                     # _found zips over the crops that ran
        return raws + [self.runner(p[4]) for p in prepared[1:]]

    def _found(self, raw, prepared, width, height):
        if self.tiles == 1:
            return self._b.postprocess(raw, prepared[1], width, height, self.config)
        found = []
        for (y0, x0, h, w, _, ratio), r in zip(prepared, raw):
            m = 0.02 * max(h, w)
            for box, class_id, score, coco in self._b.postprocess(r, ratio, w, h, self.config):
                # a box cut by an interior tile edge is a fragment of something larger; the
                # full-frame pass or an overlapping tile sees it whole, so drop the fragment
                if ((x0 > 0 and box[0] <= m) or (x0 + w < width and box[2] >= w - m)
                        or (y0 > 0 and box[1] <= m) or (y0 + h < height and box[3] >= h - m)):
                    continue
                found.append(([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0], class_id, score, coco))
        if not found:
            return []
        boxes = np.array([f[0] for f in found], dtype=np.float32)
        scores = np.array([f[2] for f in found], dtype=np.float32)
        merged = [found[k] for k in self._b.nms(boxes, scores, self.config["postprocess"]["nms_iou"])]
        return merged[:self.config["postprocess"]["max_detections"]]

    def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
        found = self._found(raw, prepared, width, height)
        provenance = f"{self.config['model']['name']}@{self.config['model']['sha256'][:12]}/{self.runner.name}"
        return [DetectionRecord(1, sequence_id, frame_index, timestamp_s, width, height,
                                BBox(*box), class_id, self.config["class_mapping_version"],
                                score, provenance, {})
                for box, class_id, score, _ in found]


def validate_timestamps(times: list[float]) -> None:
    """Strictly increasing, finite, nonnegative; zero or negative deltas are errors."""
    for i, t in enumerate(times):
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t < 0:
            raise VideoError(f"timestamp {i}: must be finite and nonnegative, got {t!r}")
        if i and t <= times[i - 1]:
            raise VideoError(f"timestamp {i}: delta {t - times[i - 1]:.9g} s is not positive")


def require_single_timestamp_source(sources) -> str:
    """Reject a sequence mixing timestamp sources (for example decode_pts and inferred_fps)."""
    distinct = set(sources)
    if len(distinct) != 1:
        raise VideoError(f"timestamp sources must be one value, got {sorted(map(str, distinct))}")
    return distinct.pop()


def probe_pts(path: Path) -> list[float] | None:
    """Presentation times of video stream 0 in display order, or None if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "frame=best_effort_timestamp_time", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120)
        if done.returncode != 0:
            return None
        frames = json.loads(done.stdout or "{}").get("frames", [])
        values = [f.get("best_effort_timestamp_time") for f in frames]
        if not values or any(v in (None, "N/A") for v in values):
            return None
        return [float(v) for v in values]
    except (subprocess.TimeoutExpired, OSError, ValueError, AttributeError):
        return None  # treated as unavailable: marked fallback or refusal by policy


def timestamp_plan(path: Path, fps: float, policy: str):
    """Return ``(pts_or_None, source, reason)`` before decoding; PTS are validated here."""
    if policy not in POLICIES:
        raise VideoError(f"timestamp policy must be one of {POLICIES}")
    pts = probe_pts(path)
    if pts is not None:
        validate_timestamps(pts)
        return pts, "decode_pts", None
    if policy == "pts":
        raise VideoError("container presentation times unavailable and policy forbids inference")
    if not math.isfinite(fps) or fps <= 0:
        raise VideoError("no usable frame rate to infer timestamps")
    return None, "inferred_fps", f"ffprobe pts unavailable; inferred from nominal {fps:g} fps"


def draw_overlay(image: np.ndarray, detections: list[DetectionRecord], timestamp_s: float) -> np.ndarray:
    out = image.copy()
    for d in detections:
        p1, p2 = (round(d.bbox.x_min), round(d.bbox.y_min)), (round(d.bbox.x_max), round(d.bbox.y_max))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cv2.putText(out, f"{d.class_id} {d.score:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, f"t={timestamp_s:.3f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _summary(values: list[float], warmup: int) -> dict:
    sample = values[warmup:]
    if not sample:
        return {"samples": 0, "warmup_excluded": min(warmup, len(values)), "p50_s": None,
                "p95_s": None, "total_s": float(sum(values))}
    return {"samples": len(sample), "warmup_excluded": warmup,
            "p50_s": float(np.percentile(sample, 50)), "p95_s": float(np.percentile(sample, 95)),
            "total_s": float(sum(values))}


def process_video(path: str | Path, stages: Stages, *, sequence_id: str,
                  output_video: str | Path | None = None, timestamp_policy: str = "pts_or_inferred",
                  stride: int = 1, max_frames: int | None = None, warmup_frames: int = 1,
                  frame_hook=None) -> dict:
    """Decode a recorded video, run detection on every ``stride``-th frame, optionally render.

    ``frame_hook(index, image_bgr, detections, timestamp_s)``, if given, is called for each
    processed frame after detection, before any overlay is drawn (for example to estimate camera
    motion, or to run the tracker and tracker-guided crops on this frame).

    Returns detections, per-frame timings, per-stage summaries and skipped-frame counts.
    Any decode, timestamp, dimension, detector or writer failure raises ``VideoError``.
    """
    path = Path(path)
    if stride < 1 or (max_frames is not None and max_frames < 1) or warmup_frames < 0:
        raise VideoError("stride and max_frames must be >= 1 and warmup_frames >= 0")
    if not path.is_file():
        raise VideoError(f"video not found: {path}")
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    pts, source, reason = timestamp_plan(path, fps, timestamp_policy)
    writer = None
    size = None
    per_stage = {name: [] for name in STAGES}
    frames_out, detections, skipped = [], [], {"stride": 0}
    index = 0
    reached_eof = False
    try:
        while max_frames is None or index < max_frames:
            t0 = time.perf_counter()
            ok, frame = capture.read()
            t_decode = time.perf_counter() - t0
            if not ok:
                reached_eof = True
                break
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise VideoError(f"frame {index}: expected 3-channel color")
            height, width = frame.shape[:2]
            if size is None:
                size = (width, height)
                if output_video is not None:
                    rate = fps if math.isfinite(fps) and fps > 0 else 1.0
                    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             rate, size)
                    if not writer.isOpened():
                        raise VideoError(f"cannot open output video: {output_video}")
            elif size != (width, height):
                raise VideoError(f"frame {index}: dimensions change mid-stream")
            if pts is not None:
                if index >= len(pts):
                    raise VideoError(f"decoded more frames than ffprobe's {len(pts)}")
                timestamp = pts[index]
            else:
                timestamp = index / fps
                if index and timestamp <= (index - 1) / fps:
                    raise VideoError(f"frame {index}: inferred timestamp does not increase")
            if index % stride:
                skipped["stride"] += 1
                if writer is not None:
                    writer.write(frame)
                index += 1
                continue
            marks = {"decode": t_decode}
            try:
                t0 = time.perf_counter()
                prepared = stages.preprocess(frame)
                marks["preprocess"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                raw = stages.infer(prepared)
                marks["inference"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                found = stages.postprocess(raw, prepared, width=width, height=height,
                                           sequence_id=sequence_id, frame_index=index,
                                           timestamp_s=timestamp)
                marks["postprocess"] = time.perf_counter() - t0
            except Exception as exc:
                raise VideoError(f"detector failed on frame {index}: {exc}") from exc
            for d in found:
                if (d.timestamp_s, d.width, d.height, d.frame_index) != (timestamp, width, height, index):
                    raise VideoError(f"detector returned a record not matching frame {index}")
            detections.extend(found)
            if frame_hook is not None:
                frame_hook(index, frame, found, timestamp)
            marks["render"] = 0.0
            if writer is not None:
                t0 = time.perf_counter()
                writer.write(draw_overlay(frame, found, timestamp))
                marks["render"] = time.perf_counter() - t0
            marks["total"] = sum(marks.values())
            for name in STAGES:
                per_stage[name].append(marks[name])
            frames_out.append({"frame_index": index, "timestamp_s": timestamp,
                               "detections": len(found), "timings_s": marks})
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if index == 0:
        raise VideoError("video produced no decodable frames")
    if pts is not None and reached_eof and index != len(pts):
        raise VideoError(f"ffprobe reports {len(pts)} frames but decoding gave {index}")
    if output_video is not None:
        check = cv2.VideoCapture(str(output_video))
        ok, probe = check.read()
        check_count = check.get(cv2.CAP_PROP_FRAME_COUNT)
        check.release()
        if (not ok or probe.shape[:2] != (size[1], size[0])
                or int(check_count) != index):
            raise VideoError("rendered output is unreadable, has different dimensions or frame count")
    width, height = size
    times = [pts[i] if pts is not None else i / fps for i in range(index)]
    return {
        "video": str(path), "sequence_id": sequence_id, "detector": stages.name,
        "width": width, "height": height, "nominal_fps": fps if math.isfinite(fps) else None,
        "timestamp_source": source, "timestamp_reason": reason, "timestamp_policy": timestamp_policy,
        "first_timestamp_s": times[0], "decoded_frames": index,
        "processed_frames": len(frames_out), "skipped_frames": skipped,
        "rendered": output_video is not None,
        "render_note": None if output_video is not None else "render disabled; render timings are 0",
        "stage_summary": {n: _summary(per_stage[n], warmup_frames) for n in STAGES},
        "wall_clock_s": time.perf_counter() - started, "frames": frames_out,
        "detections": detections,
        "timing_scope": "perf_counter wall-clock on this machine; decode excludes ffprobe; not a benchmark",
    }
