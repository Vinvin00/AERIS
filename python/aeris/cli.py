"""``aeris input.mp4 --output DIR``: recorded video -> detections -> tracks -> forecasts.

Writes ``overlay.mp4``, ``tracks.csv``, ``predictions.jsonl``, ``metrics.json``,
``benchmark.json`` and ``run_manifest.json`` into a new directory (never into an existing
one, and never partially: files are staged in a sibling directory and renamed on success).
Truth-dependent metrics are computed only when ``--annotations`` supplies labels; otherwise
they are null with a reason. Nothing here is a validated accuracy claim: the detector
operating score (``--min-score``) is an untuned default, the tracker and forecaster use
untuned defaults, and the only real-model evidence is smoke-level.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata

import cv2
import numpy as np

from aeris import __version__
from aeris.data.contracts import BBox, DetectionRecord
from aeris.detection import profiles
from aeris.estimation.camera import Calibration, CameraModel
from aeris.estimation.camera_motion import CameraMotion, apply_transform
from aeris.estimation.trajectory import MODELS as FORECAST_MODELS
from aeris.estimation.trajectory import TrajectoryForecaster, evaluate_forecasts, forecast_vs_detections
from aeris.evaluation.metrics import TrackObservation, detection_metrics, tracking_metrics
from aeris.tracking.tracker import CONFIG_PATH as TRACKER_CONFIG_PATH, TENTATIVE, KalmanTracker
from aeris.tracking.tracker import TrackerError, load_config as load_tracker_config
from aeris.video import pipeline

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIN_SCORE = 0.3     # untuned; NOT validation-selected
ARROW_MIN_SPEED_SIGMAS = 1.0  # draw the 1 s arrow once speed >= 1 velocity std; one-clip choice, see ADR 026
ARTIFACTS = ("overlay.mp4", "tracks.csv", "predictions.jsonl", "metrics.json", "benchmark.json")
TRACK_COLUMNS = (
    "sequence_id", "frame_index", "timestamp_s", "track_id", "class_id", "lifecycle", "observed",
    "cx_px", "cy_px", "vx_px_s", "vy_px_s", "w_px", "h_px", "x_min_px", "y_min_px", "x_max_px",
    "y_max_px", "age_s", "time_since_seen_s", "image_heading_deg", "speed_px_s",
    "angular_width_deg", "angular_height_deg", "angular_rate_deg_s", "availability_reason")


class CliError(ValueError):
    """User-facing failure; no output directory is created."""


class ReplayStages:
    """Deterministic detector for tests and fixture replay: detections come from a JSON file.

    Format: ``{"frames": {"<frame_index>": [{"bbox": [x0,y0,x1,y1], "class_id": 0, "score": 0.9}]}}``.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        try:
            self.frames = json.loads(raw)["frames"]
        except (ValueError, KeyError, TypeError) as exc:
            raise CliError(f"unreadable replay detections {path}: {exc}") from exc
        self.name = f"replay:{self.sha256[:12]}"

    def preprocess(self, image_bgr):
        return None

    def infer(self, prepared):
        return None

    def postprocess(self, raw, prepared, *, width, height, sequence_id, frame_index, timestamp_s):
        records = []
        for item in self.frames.get(str(frame_index), []):
            x0, y0, x1, y1 = item["bbox"]
            records.append(DetectionRecord(
                1, sequence_id, frame_index, timestamp_s, width, height, BBox(x0, y0, x1, y1),
                int(item["class_id"]), "aeris-aircraft-v1", float(item["score"]),
                f"{self.name}", {}))
        return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> str:
    """Content hash of the package source and configs (independent of Git state)."""
    entries = []
    for pattern in ("python/aeris/**/*.py", "configs/**/*.json"):
        for path in sorted(ROOT.glob(pattern)):
            if "__pycache__" not in path.parts:
                entries.append((path.relative_to(ROOT).as_posix(), sha256_file(path)))
    return hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=30)
    try:
        head, status = run("rev-parse", "HEAD"), run("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {"code_revision": None, "source_dirty": None, "reason": "git unavailable"}
    if head.returncode != 0 or status.returncode != 0:
        return {"code_revision": None, "source_dirty": None, "reason": "not a Git checkout"}
    return {"code_revision": head.stdout.strip(), "source_dirty": bool(status.stdout.strip()),
            "reason": None}


def package_versions() -> dict:
    found = {}
    for name in ("numpy", "opencv-python-headless", "scipy", "onnxruntime", "pycocotools", "trackeval"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def describe_stages(stages) -> dict:
    if isinstance(stages, ReplayStages):
        return {"kind": "replay", "detections_file": stages.path.name,
                "detections_sha256": stages.sha256, "model_sha256": None,
                "execution_provider": None, "precision": None, "input_resolution": None}
    config = getattr(stages, "config", None)
    runner = getattr(stages, "runner", None)
    if config is None or runner is None:
        return {"kind": type(stages).__name__, "model_sha256": None, "execution_provider": None,
                "precision": None, "input_resolution": None}
    providers = getattr(getattr(runner, "session", None), "get_providers", lambda: None)()
    size = config["input"]["size"]
    return {"kind": "baseline", "name": config["model"]["name"], "model_sha256": config["model"]["sha256"],
            "detector_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "runtime": runner.name, "execution_provider": providers or [runner.name],
            "precision": "float32", "input_resolution": [size, size], "profile": getattr(stages, "profile", None),
            "tiles": getattr(stages, "tiles", 1), "cascade_gate": getattr(stages, "gate", None),
            "weight_license": config["model"]["weight_license"]}


def _summary(values):
    if not values:
        return {"samples": 0, "p50_s": None, "p95_s": None, "total_s": 0.0}
    return {"samples": len(values), "p50_s": float(np.percentile(values, 50)),
            "p95_s": float(np.percentile(values, 95)), "total_s": float(sum(values))}


def _cell(value):
    if value is None:
        return ""
    return repr(float(value)) if isinstance(value, float) else str(value)


def measurement_sigma_px(state, tracker_cfg: dict) -> float:
    """Forecaster measurement noise: the tracker's position-noise fraction and pixel floor, but
    applied to the *filtered* box's larger side and shared by both axes (the forecaster takes one
    scalar), so it is conservative for elongated boxes. Untuned."""
    return max(tracker_cfg["r_floor_px"], tracker_cfg["r_position_frac"] * max(state.size))


def heading_resolved(forecaster, track_id: str, t: float) -> bool:
    """True when the forecaster's speed clearly exceeds its velocity uncertainty, so the
    direction of the forecast arrow is meaningful; hides arrows on new or slow tracks."""
    (vx, vy), (sx, sy) = forecaster.velocity(track_id, t)
    return math.hypot(vx, vy) >= ARROW_MIN_SPEED_SIGMAS * max(sx, sy)


def draw_states(frame, items) -> None:
    """Draw track boxes, IDs, lifecycle and the 1 s forecast line onto ``frame`` in place."""
    colors = {"tentative": (160, 160, 160), "confirmed": (0, 255, 0), "lost": (0, 165, 255)}
    for state, forecast in items:
        color = colors[state.lifecycle]
        if state.bbox is not None:
            b = state.bbox
            cv2.rectangle(frame, (round(b.x_min), round(b.y_min)),
                          (round(b.x_max), round(b.y_max)), color, 2)
        cx, cy = round(state.center[0]), round(state.center[1])
        label = f"{state.track_id} {state.lifecycle[0]}" + ("" if state.observed else " pred")
        cv2.putText(frame, label, (max(0, cx - 20), max(12, cy - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if forecast is not None:
            cv2.line(frame, (cx, cy), (round(forecast[0]), round(forecast[1])), color, 1)


def render_overlay(video: Path, out: Path, drawn: dict[int, list], fps: float) -> float:
    """Second pass: draw track boxes, IDs, lifecycle and the 1 s constant-velocity forecast."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise CliError(f"cannot reopen {video} for rendering")
    started, writer, index = time.perf_counter(), None, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps if fps and math.isfinite(fps) else 1.0, (width, height))
                if not writer.isOpened():
                    raise CliError(f"cannot open overlay writer {out}")
            draw_states(frame, drawn.get(index, []))
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if index == 0:
        raise CliError("no frames to render")
    return time.perf_counter() - started


def run_pipeline(input_video, output_dir, *, stages, sequence_id=None, calibration=None,
                 annotations=None, min_score=DEFAULT_MIN_SCORE, timestamp_policy="pts_or_inferred",
                 stride=1, max_frames=None, warmup_frames=1, render=True, command=None,
                 dataset_info=None, input_note=None, low_score_pass=True, camera_motion="estimate",
                 tracker_config: Path | None = None) -> dict:
    """Run the whole pipeline and write the artifact set atomically. Returns the manifest."""
    input_video, output_dir = Path(input_video), Path(output_dir)
    if not input_video.is_file():
        raise CliError(f"input video not found: {input_video}")
    if output_dir.exists():
        raise CliError(f"output directory exists: {output_dir} (runs are immutable; use a new one)")
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise CliError("--min-score must be in [0, 1]")
    if camera_motion not in ("none", "estimate"):
        raise CliError("--camera-motion must be 'none' or 'estimate'")
    sequence_id = sequence_id or "".join(c if c.isalnum() or c in "-_." else "-"
                                         for c in input_video.stem) or "sequence"
    started = time.perf_counter()
    motion, to_ref = (CameraMotion() if camera_motion == "estimate" else None), {}

    def estimate_motion(index, image, found):     # frame -> reference transform, per processed frame
        motion.update(image, [d.bbox for d in found])
        to_ref[index] = motion.to_ref.copy()

    try:
        result = pipeline.process_video(input_video, stages, sequence_id=sequence_id,
                                        output_video=None, timestamp_policy=timestamp_policy,
                                        stride=stride, max_frames=max_frames,
                                        warmup_frames=warmup_frames,
                                        frame_hook=estimate_motion if motion else None)
    except pipeline.VideoError as exc:
        raise CliError(str(exc)) from exc
    by_frame, all_by_frame = {}, {}    # at/above min_score; everything the detector kept
    for det in result["detections"]:
        all_by_frame.setdefault(det.frame_index, []).append(det)
        if det.score >= min_score:
            by_frame.setdefault(det.frame_index, []).append(det)

    camera = CameraModel(calibration)
    tracker_path = Path(tracker_config) if tracker_config else TRACKER_CONFIG_PATH
    tracker = KalmanTracker(sequence_id, load_tracker_config(tracker_path))
    forecasters = {name: TrajectoryForecaster(sequence_id, name) for name in FORECAST_MODELS}
    track_rows, predictions, drawn, previous = [], [], {}, {}
    measured = defaultdict(list)    # track_id -> [(t, assigned detection center)]
    displayed = set()               # (track_id, issue time) whose arrow was drawn
    # With camera motion, a separate constant-velocity forecaster runs in reference-frame pixels
    # (scene-relative) and drives the arrow; written predictions stay in original-frame pixels.
    scene = TrajectoryForecaster(sequence_id, "constant_velocity") if motion else None
    scene_measured, scene_predictions = defaultdict(list), []
    arrow = scene or forecasters["constant_velocity"]
    all_states, track_times, forecast_times = [], [], []
    for frame in result["frames"]:
        index, t = frame["frame_index"], frame["timestamp_s"]
        t0 = time.perf_counter()
        if low_score_pass:     # sub-threshold detections may only extend existing tracks
            states = tracker.step(t, all_by_frame.get(index, []), high_score=min_score)
        else:
            states = tracker.step(t, by_frame.get(index, []))
        track_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        frame_forecasts = {}
        for state in states:
            if state.observed:
                z = state.measurement_center_px
                if z is None:
                    raise CliError(f"observed track {state.track_id} has no assigned measurement")
                measured[state.track_id].append((t, z))
                for f in forecasters.values():
                    f.update(state.track_id, t, z, measurement_sigma_px(state, tracker.cfg))
                if scene:
                    z_ref = apply_transform(to_ref[index], z)
                    scene_measured[state.track_id].append((t, z_ref))
                    scene.update(state.track_id, t, z_ref, measurement_sigma_px(state, tracker.cfg))
        for state in states:
            if state.lifecycle == TENTATIVE:
                continue
            records = {name: f.forecast(state.track_id, t) for name, f in forecasters.items()}
            for name in forecasters:
                predictions.extend(records[name])
            if scene:
                arrow_records = scene.forecast(state.track_id, t)
                scene_predictions.extend(arrow_records)
            else:
                arrow_records = records["constant_velocity"]
            one_s = next(r for r in arrow_records if r.horizon_s == 1.0)
            if one_s.center and heading_resolved(arrow, state.track_id, t):
                frame_forecasts[state.track_id] = (one_s.center if not scene else
                                                   apply_transform(np.linalg.inv(to_ref[index]), one_s.center))
                displayed.add((state.track_id, t))
        forecast_times.append(time.perf_counter() - t0)
        drawn[index] = [(s, frame_forecasts.get(s.track_id)) for s in states]
        for state in states:
            reasons = []
            heading = camera.image_heading(*state.velocity_px_s)
            if heading.heading_deg is None:
                reasons.append(heading.availability_reason["heading_deg"])
            width_deg = height_deg = rate_deg = None
            if state.bbox is not None:
                extent = camera.bbox_angular_extent(state.bbox)
                width_deg, height_deg = extent.width_deg, extent.height_deg
                if extent.width_rad is None:
                    reasons.append(extent.availability_reason.get("angular")
                                   or "; ".join(extent.availability_reason.values()))
            else:
                reasons.append("bbox unavailable; no angular extent")
            if state.observed and state.track_id in previous:
                t_prev, c_prev = previous[state.track_id]
                rate = camera.angular_rate(c_prev, t_prev, state.center, t)
                rate_deg = rate.rate_deg_s
            if state.observed:
                previous[state.track_id] = (t, state.center)
            box = state.bbox
            track_rows.append({
                "sequence_id": sequence_id, "frame_index": index, "timestamp_s": t,
                "track_id": state.track_id, "class_id": state.class_id, "lifecycle": state.lifecycle,
                "observed": int(state.observed), "cx_px": state.center[0], "cy_px": state.center[1],
                "vx_px_s": state.velocity_px_s[0], "vy_px_s": state.velocity_px_s[1],
                "w_px": state.size[0], "h_px": state.size[1],
                "x_min_px": box.x_min if box else None, "y_min_px": box.y_min if box else None,
                "x_max_px": box.x_max if box else None, "y_max_px": box.y_max if box else None,
                "age_s": state.age_s, "time_since_seen_s": state.time_since_seen_s,
                "image_heading_deg": heading.heading_deg, "speed_px_s": heading.speed_px_s,
                "angular_width_deg": width_deg, "angular_height_deg": height_deg,
                "angular_rate_deg_s": rate_deg,
                "availability_reason": "; ".join(dict.fromkeys(r for r in reasons if r))})
        all_states.append((index, states))

    unlabeled = {"available": False, "reason": "no ground truth supplied (--annotations)"}
    truth_metrics = {"detection": {**unlabeled, "ap50": None, "ap50_95": None},
                     "tracking": {**unlabeled, "hota": None, "idf1": None},
                     "forecast": {**unlabeled, "models": None}}
    if annotations is not None:
        annotation_by_frame = {}
        for annotation in annotations:
            key = (annotation.frame.sequence_id, annotation.frame.frame_index)
            if key in annotation_by_frame:
                raise CliError(f"duplicate annotation frame: {key}")
            annotation_by_frame[key] = annotation
        aligned = []
        for frame in result["frames"]:
            key = (sequence_id, frame["frame_index"])
            annotation = annotation_by_frame.get(key)
            if annotation is None:
                raise CliError(f"annotation missing for processed frame: {key}")
            if (annotation.frame.width, annotation.frame.height) != (result["width"], result["height"]):
                raise CliError(f"annotation dimensions differ from video for frame: {key}")
            aligned.append(annotation)
        annotations = aligned
        observations = [TrackObservation(sequence_id, index, s.track_id, s.bbox, s.class_id, s.observed)
                        for index, states in all_states for s in states
                        if s.lifecycle != TENTATIVE and s.bbox is not None]
        kept = [d for ds in by_frame.values() for d in ds]
        truth_metrics = {
            "detection": detection_metrics(annotations, kept, score_threshold=min_score),
            "tracking": tracking_metrics(annotations, observations),
            "forecast": evaluate_forecasts(annotations, predictions, models=list(FORECAST_MODELS))}

    counts = {"frames_decoded": result["decoded_frames"], "frames_processed": result["processed_frames"],
              "frames_skipped": result["skipped_frames"],
              "detections_total": len(result["detections"]),
              "detections_at_or_above_min_score": sum(len(v) for v in by_frame.values()),
              "tracks_created": tracker.counters["births"], "track_rows": len(track_rows),
              "low_score_track_updates": tracker.counters["low_score_matches"],
              "predictions": len(predictions),
              "predictions_unavailable": sum(1 for p in predictions if p.center is None)}
    metrics = {"schema_version": 1, "sequence_id": sequence_id, "counts": counts,
               "operating_score_threshold": min_score,
               "operating_score_note": "untuned default, not selected on validation data",
               "truth_dependent": truth_metrics,
               "forecast_vs_detections": forecast_vs_detections(
                   measured, predictions, models=list(FORECAST_MODELS)),
               "forecast_vs_detections_displayed": forecast_vs_detections(
                   scene_measured if scene else measured,
                   [p for p in (scene_predictions if scene else predictions)
                    if (p.track_id, p.issue_time_s) in displayed], models=["constant_velocity"]),
               "forecast_vs_detections_displayed_frame": ("reference frame (scene-relative)" if scene
                                                          else "original frame"),
               "forecast_vs_detections_scene": forecast_vs_detections(
                   scene_measured, scene_predictions, models=["constant_velocity"]) if scene else
                   {"available": False, "reason": "camera motion not estimated (--camera-motion none)"},
               "camera_motion": {"mode": camera_motion, **(motion.counters if motion else {}),
                                 "note": "failed steps are treated as no camera motion; untuned, "
                                         "image-based, errors accumulate along the sequence"},
               "arrow_min_speed_sigmas": ARROW_MIN_SPEED_SIGMAS,
               "note": "Truth-dependent metrics are null unless labels are supplied; no accuracy is claimed."}

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=parent))
    try:
        render_s = 0.0
        if render:
            render_s = render_overlay(input_video, stage / "overlay.mp4", drawn, result["nominal_fps"])
            check = cv2.VideoCapture(str(stage / "overlay.mp4"))
            ok, probe = check.read()
            frame_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            check.release()
            if (not ok or probe.shape[:2] != (result["height"], result["width"])
                    or frame_count != result["decoded_frames"]):
                raise CliError("overlay is unreadable or has the wrong dimensions or frame count")
        with open(stage / "tracks.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in track_rows:
                writer.writerow({k: _cell(v) for k, v in row.items()})
        with open(stage / "predictions.jsonl", "w") as handle:
            for record in predictions:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        (stage / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
        wall = time.perf_counter() - started
        benchmark = {
            "schema_version": 1,
            "scope": "offline recorded file; wall-clock perf_counter on this machine; not a benchmark",
            "camera_motion_note": ("camera-motion estimation runs inside the decode pass and is not attributed "
                                   "to any stage; it is included in wall_clock_s only"),
            "stages": {**{k: result["stage_summary"][k] for k in
                          ("decode", "preprocess", "inference", "postprocess", "total")},
                       "track": _summary(track_times[warmup_frames:]),
                       "forecast": _summary(forecast_times[warmup_frames:])},
            "render_overlay_total_s": render_s if render else None,
            "render_note": None if render else "render disabled",
            "warmup_frames_excluded": warmup_frames, "samples_after_warmup": max(
                0, result["processed_frames"] - warmup_frames),
            "wall_clock_s": wall, "frames_dropped": 0,
            "queueing": "not applicable: offline file, no real-time source",
            "hardware": {"machine": platform.machine(), "processor": platform.processor() or None,
                         "os": platform.platform()}}
        (stage / "benchmark.json").write_text(
            json.dumps(benchmark, indent=2, sort_keys=True, allow_nan=False) + "\n")
        git = git_state()
        locks = {p.name: sha256_file(p) for p in sorted((ROOT / "requirements").glob("*.txt"))}
        manifest = {
            "schema_version": 1, "aeris_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": command if command is not None else [sys.executable, "-m", "aeris", *sys.argv[1:]],
            "code_revision": git["code_revision"], "source_dirty": git["source_dirty"],
            "git_reason": git["reason"], "source_fingerprint": source_fingerprint(),
            "input": {"path_name": input_video.name, "note": input_note,
                      "sha256": sha256_file(input_video),
                      "size_bytes": input_video.stat().st_size, "width": result["width"],
                      "height": result["height"], "nominal_fps": result["nominal_fps"]},
            "sequence_id": sequence_id,
            "detector": describe_stages(stages),
            "tracker_config": tracker_path.name,
            "tracker_config_sha256": sha256_file(tracker_path),
            "forecast_models": sorted(FORECAST_MODELS),
            "calibration": ({"calibration_id": calibration.calibration_id, "camera_id": calibration.camera_id,
                             "quality": calibration.quality} if calibration else
                            {"calibration_id": None, "reason": "no calibration supplied (pixel-only mode)"}),
            "dataset": dataset_info or {"dataset_id": None, "split": None,
                                        "reason": "no annotations supplied"},
            "timestamp": {"policy": result["timestamp_policy"], "source": result["timestamp_source"],
                          "reason": result["timestamp_reason"]},
            "min_score": min_score, "low_score_pass": low_score_pass, "camera_motion": camera_motion, "stride": stride, "max_frames": max_frames,
            "warmup_frames": warmup_frames, "samples_after_warmup": benchmark["samples_after_warmup"],
            "random_seed": None, "seed_reason": "no random operations",
            "environment": {"os": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "packages": package_versions()},
            "dependency_lock_sha256": locks,
            "artifact_sha256": {name: sha256_file(stage / name) for name in ARTIFACTS
                                if (stage / name).exists()},
            "accuracy": None,
            "accuracy_reason": ("labels supplied; see metrics.json" if annotations is not None
                                else "no labels supplied; no accuracy computed")}
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aeris", description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="recorded video file")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="calibration JSON (K, distortion, ...); omit for pixel-only mode")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="canonical dataset directory with labels; omit for unlabeled input")
    parser.add_argument("--detections", type=Path, default=None,
                        help="replay detections JSON instead of running the model (deterministic)")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="detector config (default configs/detector/baseline.json)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--tracker-config", type=Path, default=None,
                        help="tracker config JSON (default configs/tracker/baseline.json; see docs/tracking.md)")
    parser.add_argument("--camera-motion", choices=("none", "estimate"), default="estimate",
                        help="estimate background motion so the forecast arrow is scene-relative "
                             "(default; use 'none' for a known fixed camera); written predictions "
                             "and tracks stay in image pixels")
    parser.add_argument("--profile", choices=tuple(profiles.PROFILES), default=profiles.DEFAULT_PROFILE,
                        help="detector profile: model, zoom tiles and cascade (see aeris.detection.profiles)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument("--timestamp-policy", choices=pipeline.POLICIES, default="pts_or_inferred")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--input-note", default=None,
                        help="free-text provenance recorded in the manifest, e.g. 'derived from one photo'")
    return parser


def _load_calibration(path):
    if path is None:
        return None
    try:
        return Calibration.from_dict(json.loads(Path(path).read_text()))
    except (OSError, ValueError, TypeError) as exc:
        raise CliError(f"unusable calibration {path}: {exc}") from exc


def _load_annotations(path):
    if path is None:
        return None, None
    from aeris.data.validate import read_annotations, read_index
    index, _, index_problems = read_index(Path(path))
    pairs, problems = read_annotations(Path(path))
    if index is None or index_problems or problems:
        raise CliError(f"unusable annotations {path}: {(index_problems + problems)[:3]}")
    info = {"dataset_id": index.get("dataset_id"), "split": sorted({s["split"] for s in index["sequences"]}),
            "dataset_json_sha256": sha256_file(Path(path) / "dataset.json"),
            "annotations_sha256": sha256_file(Path(path) / "annotations.jsonl")}
    return [a for _, a in pairs], info


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.detections is not None:
            stages = ReplayStages(args.detections)
        else:
            from aeris.detection import baseline
            try:
                stages = profiles.build(args.profile, args.model_config)
            except (baseline.DetectorError, ImportError) as exc:
                raise CliError(f"cannot load the detector: {exc} (see models/README.md, or pass --detections)") from exc
            if stages.gate is not None and stages.gate < args.min_score:
                raise CliError(f"profile {args.profile!r} gates zoom at {stages.gate}, below --min-score "
                               f"{args.min_score}: weak full-frame hits would suppress the zoom")
        annotations, dataset_info = _load_annotations(args.annotations)
        manifest = run_pipeline(
            args.input, args.output, stages=stages, sequence_id=args.sequence_id,
            calibration=_load_calibration(args.calibration), annotations=annotations,
            min_score=args.min_score, timestamp_policy=args.timestamp_policy, stride=args.stride,
            max_frames=args.max_frames, warmup_frames=args.warmup_frames, render=not args.no_render,
            command=[sys.executable, "-m", "aeris", *(argv if argv is not None else sys.argv[1:])],
            dataset_info=dataset_info, input_note=args.input_note, camera_motion=args.camera_motion,
            tracker_config=args.tracker_config)
    except (CliError, TrackerError, ValueError, OSError) as exc:
        print(f"aeris: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "artifacts": sorted(manifest["artifact_sha256"]),
                      "run_manifest": str(args.output / "run_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
