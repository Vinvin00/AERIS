"""Deterministic analytic fixtures. SYNTHETIC: they validate math and plumbing only.

Two families, each split into perception inputs and evaluator-only truth:

  image_space/inputs/  frames.jsonl (FrameRecord), detections.jsonl (DetectionRecord) and
                       images/*.pgm (rendered frames; written only on request)
  image_space/truth/   dataset.json + annotations.jsonl (AERIS-003 canonical layout) and
                       expected.json (analytic states, gaps, events, forecast targets)
  projection/inputs/   camera.json (calibration, model-input transform), observations.jsonl
  projection/truth/    truth.json (scene positions, rays, angular extents and rates)

Coordinates are original-frame continuous pixels: pixel (i, j) spans [i, i+1) x [j, j+1),
so the top-left pixel centre is (0.5, 0.5). Image-space timestamps are dyadic, so
centres and boxes are exact in binary floating point. Detector noise comes from SHA-256
of (seed, key) and IEEE arithmetic only. Values that need trigonometry are rounded
(ANGLE_DECIMALS, DEGREE_DECIMALS) so a last-bit libm difference between platforms cannot
change artifact bytes except with negligible probability. Inputs for frames up to time t
never depend on later frames. Scene units are arbitrary; only ratios matter.

Usage: python -m aeris.simulation.fixtures OUT_DIR [--seed N] [--images]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from aeris.data.contracts import (
    PREDICTION_HORIZONS_S, AnnotationRecord, BBox, DetectionRecord, FrameRecord,
    GroundTruthObject,
)

GENERATOR_VERSION = 1
DEFAULT_SEED = 20260925
SCOPE = ("SYNTHETIC analytic fixture: validates math and plumbing only; it is not "
         "perception, tracking, forecasting or geometry performance evidence")
DETECTOR = ("SYNTHETIC analytic fixture detector (aeris.simulation.fixtures "
            f"v{GENERATOR_VERSION}); not a trained model")
CLASS_MAPPING_VERSION = "aeris-aircraft-v1"
IMAGE_CAMERA_ID = "sim-cam-image"
NOISE_SIGMA_PX = 0.5
ANGLE_DECIMALS = 12
DEGREE_DECIMALS = 9
MIN_SPEED_PX_S = 1e-9
PIXEL_CONVENTION = (
    "continuous original-frame pixels: pixel (i, j) spans [i, i+1) x [j, j+1) and the "
    "top-left pixel centre is (0.5, 0.5); OpenCV puts that centre at (0, 0), so an "
    "OpenCV principal point is (cx - 0.5, cy - 0.5)")


@dataclass(frozen=True)
class Track:
    """Constant-velocity box: centre at time 0 (px), velocity (px/s), size (px)."""

    track_id: str
    center: tuple[float, float]
    velocity: tuple[float, float]
    size: tuple[float, float]
    hidden: frozenset[int] = frozenset()

    def center_at(self, t: float) -> tuple[float, float]:
        return (self.center[0] + self.velocity[0] * t, self.center[1] + self.velocity[1] * t)

    def bbox_at(self, t: float) -> BBox:
        (x, y), (w, h) = self.center_at(t), self.size
        return BBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2)


@dataclass(frozen=True)
class Scenario:
    sequence_id: str
    width: int
    height: int
    frames: tuple[tuple[int, float], ...]
    tracks: tuple[Track, ...]
    missed: frozenset[tuple[int, str]] = frozenset()
    duplicates: frozenset[tuple[int, str]] = frozenset()
    false_positives: tuple[tuple[int, BBox], ...] = ()


IMAGE_SCENARIOS = (
    # Variable delta times; frames 6 and 7 dropped; t1 and t2 cross at t = 0.5 s (frame 8);
    # t2 hidden in frames 11-12; t3 stationary; one miss, one duplicate, one false positive.
    Scenario(
        "sim-cv-crossing", 160, 120,
        frames=((0, 0.0), (1, 0.0625), (2, 0.125), (3, 0.15625), (4, 0.25), (5, 0.3125),
                (8, 0.5), (9, 0.5625), (10, 0.625), (11, 0.75), (12, 0.8125), (13, 0.875),
                (14, 1.0)),
        tracks=(Track("t1", (20.0, 40.0), (96.0, 32.0), (16.0, 8.0)),
                Track("t2", (116.0, 72.0), (-96.0, -32.0), (12.0, 12.0), frozenset({11, 12})),
                Track("t3", (136.0, 20.0), (0.0, 0.0), (10.0, 10.0))),
        missed=frozenset({(4, "t1")}),
        duplicates=frozenset({(9, "t3")}),
        false_positives=((2, BBox(36.0, 97.0, 44.0, 103.0)),)),
    # Complete empty frames around a track that is visible only in frames 2-5.
    Scenario(
        "sim-empty-birth", 160, 120,
        frames=tuple((index, index * 0.125) for index in range(8)),
        tracks=(Track("b1", (30.0, 90.0), (64.0, -48.0), (14.0, 7.0), frozenset({0, 1, 6, 7})),)),
)

CAMERA = {
    "camera_id": "sim-cam-calibrated", "calibration_id": "sim-cal-v1", "width": 640, "height": 480,
    "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0,
    "k1": -0.12, "k2": 0.03, "p1": 0.001, "p2": -0.0005, "k3": 0.0,
}
MODEL_INPUT_CROP_XYWH = (64, 48, 512, 384)
MODEL_INPUT_RESIZE_WH = (256, 192)
TARGET_CENTER = (-400.0, -120.0, 3000.0)   # scene units at t = 0, camera frame
TARGET_VELOCITY = (150.0, 20.0, -400.0)    # scene units per second (approaching)
TARGET_HALF_SPAN, TARGET_HALF_HEIGHT = 18.0, 3.0
PROJECTION_TIMES = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0)
ROUND_TRIP_NORMALIZED = tuple((x, y) for y in (-0.45, 0.0, 0.45)
                              for x in (-0.6, -0.3, 0.0, 0.3, 0.6))
REFERENCE_ON_AXIS = {"half_span": 18.0, "depth": 2000.0}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unit(seed: int, *key) -> float:
    """Uniform in [0, 1) from SHA-256 of (seed, key); independent of platform and numpy."""
    digest = hashlib.sha256(json.dumps([seed, *key]).encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2.0 ** 53


def _noise(seed: int, *key) -> float:
    """Zero-mean noise with standard deviation NOISE_SIGMA_PX: an Irwin-Hall sum of four
    uniforms (approximately Gaussian, bounded at +/-2*sqrt(3) sigma), IEEE arithmetic only."""
    return (sum(_unit(seed, *key, k) for k in range(4)) - 2.0) * NOISE_SIGMA_PX * math.sqrt(3.0)


def _jitter(box: BBox, seed: int, *key) -> BBox:
    return BBox(*(value + _noise(seed, *key, name) for name, value in
                  (("x_min", box.x_min), ("y_min", box.y_min), ("x_max", box.x_max),
                   ("y_max", box.y_max))))


def _visible(track: Track, frame_index: int) -> bool:
    return frame_index not in track.hidden


def _heading_deg(velocity: tuple[float, float]) -> float | None:
    """Image heading atan2(-vy, vx), counterclockwise from image right, in [0, 360)."""
    if math.hypot(*velocity) < MIN_SPEED_PX_S:
        return None
    return round(math.degrees(math.atan2(-velocity[1], velocity[0])) % 360.0, DEGREE_DECIMALS)


def _render(scenario: Scenario, frame_index: int, t: float) -> bytes:
    """Binary PGM: a faint per-frame texture plus filled boxes of the visible tracks.

    A pixel is filled when its centre lies in the half-open truth box. Rendering is
    illustrative; the truth boxes are exact by construction.
    """
    ys, xs = np.mgrid[0:scenario.height, 0:scenario.width]
    image = (200 + (xs + 3 * ys + 7 * frame_index) % 17).astype(np.uint8)
    for track in scenario.tracks:
        if _visible(track, frame_index):
            box = track.bbox_at(t)
            image[(xs + 0.5 >= box.x_min) & (xs + 0.5 < box.x_max)
                  & (ys + 0.5 >= box.y_min) & (ys + 0.5 < box.y_max)] = 60
    return f"P5\n{scenario.width} {scenario.height}\n255\n".encode() + image.tobytes()


def _detections(scenario: Scenario, seed: int, frame_index: int, t: float) -> list[tuple]:
    """(bbox, score, source track or None, kind), ordered left to right as a detector might."""
    found = []
    sid = scenario.sequence_id
    for track in scenario.tracks:
        if not _visible(track, frame_index) or (frame_index, track.track_id) in scenario.missed:
            continue
        truth = track.bbox_at(t)
        score = 0.7 + 0.25 * _unit(seed, sid, frame_index, track.track_id, "score")
        found.append((_jitter(truth, seed, sid, frame_index, track.track_id, "box"), score,
                      track.track_id, "true_positive"))
        if (frame_index, track.track_id) in scenario.duplicates:
            shifted = BBox(truth.x_min + 2, truth.y_min + 1, truth.x_max + 2, truth.y_max + 1)
            found.append((_jitter(shifted, seed, sid, frame_index, track.track_id, "duplicate"),
                          0.55, track.track_id, "duplicate"))
    found += [(box, 0.3, None, "false_positive") for index, box in scenario.false_positives
              if index == frame_index]
    return sorted(found, key=lambda item: (item[0].x_min, item[0].y_min))


def _iou(a: BBox, b: BBox) -> float:
    width = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    height = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    area = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min)
    return inter / (area - inter)


def _open_interval(offset: float, rate: float, limit: float) -> tuple[float, float] | None:
    """Times t with |offset + rate * t| < limit (None: never; infinite bounds: always)."""
    if rate == 0:
        return (-math.inf, math.inf) if abs(offset) < limit else None
    ends = sorted(((-limit - offset) / rate, (limit - offset) / rate))
    return ends[0], ends[1]


def _pair_events(scenario: Scenario) -> list[dict]:
    """Analytic centre crossings and box-overlap spans for each track pair."""
    events = []
    start, end = scenario.frames[0][1], scenario.frames[-1][1]
    tracks = scenario.tracks
    for i, first in enumerate(tracks):
        for second in tracks[i + 1:]:
            offset = [second.center[k] - first.center[k] for k in range(2)]
            rate = [second.velocity[k] - first.velocity[k] for k in range(2)]
            spans = [_open_interval(offset[k], rate[k], (first.size[k] + second.size[k]) / 2)
                     for k in range(2)]
            if None in spans:
                continue
            low, high = max(spans[0][0], spans[1][0], start), min(spans[0][1], spans[1][1], end)
            if low >= high:
                continue
            event = {"track_ids": [first.track_id, second.track_id],
                     "box_overlap_span_s": [low, high],
                     "overlapping_frames": [
                         {"frame_index": index, "timestamp_s": t, "iou": iou}
                         for index, t in scenario.frames
                         if (iou := _iou(first.bbox_at(t), second.bbox_at(t))) > 0]}
            moving = [k for k in range(2) if rate[k] != 0]
            times = {-offset[k] / rate[k] for k in moving}
            if moving and len(times) == 1 and all(offset[k] == 0 for k in range(2) if k not in moving):
                crossing = times.pop()
                if start <= crossing <= end:
                    event.update(type="crossing", crossing_time_s=crossing,
                                 crossing_center_px=list(first.center_at(crossing)),
                                 crossing_frame_index=next(
                                     (index for index, t in scenario.frames if t == crossing), None))
            event.setdefault("type", "box_overlap")
            events.append(event)
    return events


def _visible_spans(scenario: Scenario, track: Track) -> list[dict]:
    spans, current = [], None
    for index, t in scenario.frames:
        if _visible(track, index):
            if current is None:
                current = {"first_frame_index": index, "start_s": t}
            current.update(last_frame_index=index, end_s=t)
        elif current is not None:
            spans.append(current)
            current = None
    return spans + ([current] if current else [])


def _frame_gaps(scenario: Scenario) -> list[dict]:
    return [{"after_frame_index": a, "next_frame_index": b,
             "missing_frame_indices": list(range(a + 1, b)), "delta_t_s": tb - ta}
            for (a, ta), (b, tb) in zip(scenario.frames, scenario.frames[1:]) if b != a + 1]


def _expected_sequence(scenario: Scenario, detection_rows: list[dict]) -> dict:
    last_time = scenario.frames[-1][1]
    tracks = []
    for track in scenario.tracks:
        speed = math.hypot(*track.velocity)
        heading = _heading_deg(track.velocity)
        tracks.append({
            "track_id": track.track_id, "class_id": 0,
            "motion": {"model": "constant_velocity", "reference_time_s": 0.0,
                       "center_px": list(track.center), "velocity_px_s": list(track.velocity),
                       "size_px": list(track.size)},
            "speed_px_s": speed, "heading_deg": heading,
            "availability_reason": {} if heading is not None else {
                "heading_deg": f"speed below {MIN_SPEED_PX_S} px/s: image heading is undefined"},
            "visible_spans": _visible_spans(scenario, track),
            "states": [{"frame_index": index, "timestamp_s": t, "visible": _visible(track, index),
                        "center_px": list(track.center_at(t)), "velocity_px_s": list(track.velocity),
                        "bbox": track.bbox_at(t).to_dict()} for index, t in scenario.frames],
        })
    forecasts = [{"track_id": track.track_id, "issue_frame_index": index, "issue_time_s": t,
                  "horizon_s": horizon, "target_time_s": t + horizon,
                  "center_px": list(track.center_at(t + horizon)),
                  "target_within_sequence": t + horizon <= last_time}
                 for index, t in scenario.frames for track in scenario.tracks
                 if _visible(track, index) for horizon in PREDICTION_HORIZONS_S]
    times = [t for _, t in scenario.frames]
    return {
        "sequence_id": scenario.sequence_id, "width": scenario.width, "height": scenario.height,
        "frame_times": [{"frame_index": index, "timestamp_s": t} for index, t in scenario.frames],
        "delta_t_s": [b - a for a, b in zip(times, times[1:])],
        "frame_index_gaps": _frame_gaps(scenario),
        "empty_frame_indices": [index for index, _ in scenario.frames
                                if not any(_visible(track, index) for track in scenario.tracks)],
        "tracks": tracks,
        "pair_events": _pair_events(scenario),
        "detections": detection_rows,
        "missed_detections": [{"frame_index": index, "track_id": track_id}
                              for index, track_id in sorted(scenario.missed)],
        "forecast_targets": forecasts,
    }


def build_image_space(seed: int = DEFAULT_SEED,
                      scenarios: tuple[Scenario, ...] = IMAGE_SCENARIOS) -> dict[str, bytes]:
    """Return {relative path: bytes} for the image-space family, images included."""
    files, frames, detections, annotations, expected = {}, [], [], [], []
    for scenario in scenarios:
        sid, rows = scenario.sequence_id, []
        for index, t in scenario.frames:
            image = _render(scenario, index, t)
            name = f"{sid}_{index:04d}.pgm"
            files[f"image_space/inputs/images/{name}"] = image
            frame = FrameRecord(1, sid, index, t, "provided", scenario.width, scenario.height, name,
                                sha256(image), IMAGE_CAMERA_ID, "complete", {})
            frames.append(frame.to_json())
            for number, (box, score, source, kind) in enumerate(_detections(scenario, seed, index, t)):
                detections.append(DetectionRecord(
                    1, sid, index, t, scenario.width, scenario.height, box, 0,
                    CLASS_MAPPING_VERSION, score, DETECTOR, {}).to_json())
                rows.append({"frame_index": index, "detection_index": number, "kind": kind,
                             "source_track_id": source})
            objects = tuple(GroundTruthObject(track.track_id, 0, track.bbox_at(t), track.track_id,
                                              False, False, False, {})
                            for track in scenario.tracks if _visible(track, index))
            annotations.append(AnnotationRecord(1, frame, CLASS_MAPPING_VERSION, objects).to_json())
        expected.append(_expected_sequence(scenario, rows))
    files["image_space/inputs/frames.jsonl"] = _lines(frames)
    files["image_space/inputs/detections.jsonl"] = _lines(detections)
    files["image_space/truth/annotations.jsonl"] = _lines(annotations)
    files["image_space/truth/dataset.json"] = _json({
        "schema_version": 1, "dataset_id": "aeris-sim-image-space-v1",
        "class_mapping_version": CLASS_MAPPING_VERSION, "image_root": "../inputs/images",
        "group_basis": f"one generated sequence per group; {SCOPE}",
        "unannotated_attributes": [], "transforms": {},
        "sequences": [{"sequence_id": s.sequence_id, "group_id": s.sequence_id,
                       "split": "synthetic"} for s in scenarios]})
    files["image_space/truth/expected.json"] = _json({
        "schema_version": 1, "scope": SCOPE, "evaluator_only": True,
        "units": {"position": "px, original-frame continuous coordinates", "velocity": "px/s",
                  "time": "s", "heading": "degrees counterclockwise from image right, [0, 360)"},
        "pixel_convention": PIXEL_CONVENTION,
        "detection_noise": {"model": "Irwin-Hall sum of four uniforms per box coordinate",
                            "sigma_px": NOISE_SIGMA_PX,
                            "bound_px": 2 * math.sqrt(3.0) * NOISE_SIGMA_PX},
        "sequences": expected})
    return files


def _distort(x: float, y: float) -> tuple[float, float]:
    """OpenCV Brown-Conrady model on normalized coordinates (x = X/Z, y = Y/Z)."""
    c = CAMERA
    r2 = x * x + y * y
    radial = 1 + c["k1"] * r2 + c["k2"] * r2 * r2 + c["k3"] * r2 * r2 * r2
    return (x * radial + 2 * c["p1"] * x * y + c["p2"] * (r2 + 2 * x * x),
            y * radial + c["p1"] * (r2 + 2 * y * y) + 2 * c["p2"] * x * y)


def project(point: tuple[float, float, float]) -> tuple[float, float]:
    """Camera-frame point (Z > 0) to distorted continuous original-frame pixels."""
    if point[2] <= 0:
        raise ValueError("point must lie in front of the camera")
    x, y = _distort(point[0] / point[2], point[1] / point[2])
    return CAMERA["fx"] * x + CAMERA["cx"], CAMERA["fy"] * y + CAMERA["cy"]


def _norm(v) -> float:
    return math.sqrt(sum(value * value for value in v))


def _unit_vector(v) -> list[float]:
    length = _norm(v)
    return [value / length for value in v]


def _cross(a, b) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _angle(a, b) -> float:
    """Angle between two directions, atan2(|a x b|, a . b), rounded to ANGLE_DECIMALS."""
    return round(math.atan2(_norm(_cross(a, b)), sum(p * q for p, q in zip(a, b))), ANGLE_DECIMALS)


def _markers(t: float) -> dict[str, list[float]]:
    center = [TARGET_CENTER[k] + TARGET_VELOCITY[k] * t for k in range(3)]
    offsets = {"center": (0, 0, 0), "left": (-TARGET_HALF_SPAN, 0, 0),
               "right": (TARGET_HALF_SPAN, 0, 0), "top": (0, -TARGET_HALF_HEIGHT, 0),
               "bottom": (0, TARGET_HALF_HEIGHT, 0)}
    return {name: [center[k] + offset[k] for k in range(3)] for name, offset in offsets.items()}


def _to_model_input(u: float, v: float) -> list[float]:
    (x0, y0, w, h), (width, height) = MODEL_INPUT_CROP_XYWH, MODEL_INPUT_RESIZE_WH
    return [(u - x0) * width / w, (v - y0) * height / h]


def build_projection(times: tuple[float, ...] = PROJECTION_TIMES) -> dict[str, bytes]:
    """Return {relative path: bytes} for the calibrated projection family."""
    c = CAMERA
    observations, frames, first_pixels = [], [], None
    for index, t in enumerate(times):
        markers = _markers(t)
        pixels = {name: list(project(point)) for name, point in markers.items()}
        first_pixels = first_pixels or pixels
        us, vs = [p[0] for p in pixels.values()], [p[1] for p in pixels.values()]
        box = BBox(min(us), min(vs), max(us), max(vs))
        box.validate_bounds(c["width"], c["height"])
        observations.append(json.dumps({
            "schema_version": 1, "sequence_id": "sim-projection", "camera_id": c["camera_id"],
            "frame_index": index, "timestamp_s": t, "bbox": box.to_dict(), "points_px": pixels},
            allow_nan=False, sort_keys=True, separators=(",", ":")))
        center = markers["center"]
        rays = {name: _unit_vector(point) for name, point in markers.items()}
        frames.append({
            "frame_index": index, "timestamp_s": t, "center_scene": center,
            "velocity_scene_per_s": list(TARGET_VELOCITY), "markers_scene": markers,
            "range_scene": _norm(center),
            "range_rate_scene_per_s": sum(p * q for p, q in zip(center, TARGET_VELOCITY)) / _norm(center),
            "rays": rays,
            "angular_extent_rad": {"horizontal": _angle(rays["left"], rays["right"]),
                                   "vertical": _angle(rays["top"], rays["bottom"])},
            "angular_rate_rad_s": _norm(_cross(center, TARGET_VELOCITY)) / _norm(center) ** 2,
        })
    reference = REFERENCE_ON_AXIS
    left = (-reference["half_span"], 0.0, reference["depth"])
    right = (reference["half_span"], 0.0, reference["depth"])
    x0, y0, w, h = MODEL_INPUT_CROP_XYWH
    checks = [(x0, y0), (x0 + w, y0 + h), (float(c["cx"]), float(c["cy"]))]
    checks += [tuple(point) for point in first_pixels.values()]
    camera = {
        "schema_version": 1, "scope": SCOPE, "camera_id": c["camera_id"],
        "calibration_id": c["calibration_id"], "original_resolution": [c["width"], c["height"]],
        "intrinsics": {"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"], "skew": 0.0},
        "distortion": {"model": "opencv_brown_conrady", "order": ["k1", "k2", "p1", "p2", "k3"],
                       "coefficients": {k: c[k] for k in ("k1", "k2", "p1", "p2", "k3")}},
        "pixel_convention": PIXEL_CONVENTION,
        "camera_frame": "x right, y down, z forward along the optical axis (OpenCV convention)",
        "lens": "synthetic pinhole with Brown-Conrady distortion", "zoom": "fixed",
        "quality": {"source": "exact synthetic parameters; no calibration was estimated",
                    "reprojection_error_px": None,
                    "availability_reason": {"reprojection_error_px":
                                            "not estimated: parameters are exact by construction"}},
        "model_input_transform": {"crop_xywh": list(MODEL_INPUT_CROP_XYWH),
                                  "resize_wh": list(MODEL_INPUT_RESIZE_WH),
                                  "mapping": "model = (original - crop_origin) * resize / crop_size"},
    }
    truth = {
        "schema_version": 1, "scope": SCOPE, "evaluator_only": True,
        "units": {"scene": "arbitrary length unit; only ratios matter", "time": "s",
                  "angle": "rad", "angular_rate": "rad/s (camera-relative)"},
        "sequence_id": "sim-projection", "frames": frames,
        "round_trip_points": [
            {"normalized": [x, y], "ray": _unit_vector((x, y, 1.0)),
             "pixel": list(project((x, y, 1.0)))} for x, y in ROUND_TRIP_NORMALIZED],
        "reference_on_axis": {**reference, "angular_extent_rad":
                              _angle(_unit_vector(left), _unit_vector(right))},
        "transform_checks": [{"original_px": list(point), "model_input_px": _to_model_input(*point)}
                             for point in checks],
    }
    return {"projection/inputs/camera.json": _json(camera),
            "projection/inputs/observations.jsonl": _lines(observations),
            "projection/truth/truth.json": _json(truth)}


ROLES = {"image_space/inputs/": "perception_input", "projection/inputs/": "perception_input",
         "image_space/truth/": "evaluator_truth", "projection/truth/": "evaluator_truth"}


def build_fixture_set(seed: int = DEFAULT_SEED) -> tuple[dict[str, bytes], dict]:
    """Return (artifacts, manifest). Images are listed in the manifest but not committed."""
    files = {**build_image_space(seed), **build_projection()}
    artifacts = []
    for path, data in sorted(files.items()):
        entry = {"path": path, "role": next(role for prefix, role in ROLES.items()
                                            if path.startswith(prefix)),
                 "sha256": sha256(data), "bytes": len(data),
                 "committed": not path.endswith(".pgm")}
        if path.endswith(".jsonl"):
            entry["records"] = data.count(b"\n")
        artifacts.append(entry)
    manifest = {
        "schema_version": 1, "fixture_set_id": f"aeris-sim-fixtures-v{GENERATOR_VERSION}",
        "generator": "python/aeris/simulation/fixtures.py", "generator_version": GENERATOR_VERSION,
        "seed": seed, "scope": SCOPE,
        "regenerate": ("PYTHONPATH=python python -m aeris.simulation.fixtures "
                       "tests/fixtures/simulation  # add --images OUT for rendered frames"),
        "roles": {"perception_input": "may be given to runtime code",
                  "evaluator_truth": "evaluator and tests only; never a runtime input"},
        "artifacts": artifacts,
    }
    return files, manifest


def write_fixture_set(out_dir, seed: int = DEFAULT_SEED, images: bool = False) -> dict:
    """Write committed artifacts (and PGM frames when images=True) plus manifest.json."""
    root = Path(out_dir)
    files, manifest = build_fixture_set(seed)
    for path, data in files.items():
        if images or not path.endswith(".pgm"):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    (root / "manifest.json").write_bytes(_json(manifest))
    return manifest


def _json(value) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, indent=1) + "\n").encode()


def _lines(records: list[str]) -> bytes:
    return "".join(record + "\n" for record in records).encode()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the AERIS synthetic analytic fixtures.")
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--images", action="store_true", help="also write rendered PGM frames")
    args = parser.parse_args(argv)
    try:
        manifest = write_fixture_set(args.out_dir, args.seed, args.images)
    except OSError as error:
        print(f"could not write fixtures: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"artifacts": len(manifest["artifacts"]), "seed": manifest["seed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
