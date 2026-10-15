"""Bounded real-video pilot on QUT S5; labels enter only the evaluator.

The CSV contains approximate target centroids, not exhaustive boxes. This script
reports descriptive, association-conditioned pixel errors, never AP or ENU/range
accuracy. The pinned source media and labels live only in ignored data/raw/qut/.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from aeris.detection import baseline  # noqa: E402
from aeris.estimation.trajectory import MODELS, TrajectoryForecaster  # noqa: E402
from aeris.tracking.tracker import CONFIRMED, KalmanTracker, load_config as tracker_config  # noqa: E402
from aeris.video import pipeline  # noqa: E402

REVISION = "0bb9ece0c70d56a56f0334dfdd867b0d7fce15f9"
BASE_URL = ("https://huggingface.co/datasets/jasminmartin/"
            f"vision_based_aircraft_detection_dataset/resolve/{REVISION}")
SEQUENCE_BOUNDS_URL = f"{BASE_URL}/target-info.csv"
VIDEO_URL = f"{BASE_URL}/Videos/S5.mp4"
LABEL_URL = f"{BASE_URL}/Target%20Labels/target-aircarft-labels/S5_labels.csv"
VIDEO_SHA256 = "f02f21c5fa60eb64e553961b522644e02620303677457e160a2ba41672e2efdd"
LABEL_SHA256 = "e9b1721f13f9208aa0486df1a4d7e655655a86f745f8fc21d8d582217236f9ab"
SOURCE_FIRST_FRAME = 57750
SOURCE_LAST_FRAME = 59426
MATCH_RADIUS_PX = 50.0  # fixed descriptive association gate, never optimized on S5
TRACK_SCORE = 0.3  # existing untuned CLI default; not validation-selected
HORIZONS_S = (0.1, 0.25, 0.5, 1.0, 2.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path: Path, *, frame_count: int, width: int, height: int,
                first_global_frame: int = SOURCE_FIRST_FRAME,
                last_global_frame: int = SOURCE_LAST_FRAME) -> dict[int, tuple[float, float]]:
    """Validate the published global-frame CSV and map it to zero-based video frames."""
    labels = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["", "frame", "u", "v", "range"]:
            raise ValueError(f"unexpected QUT CSV columns: {reader.fieldnames}")
        previous = -1
        for row in reader:
            global_frame = int(row["frame"])
            index = global_frame - first_global_frame
            u, v, source_range = float(row["u"]), float(row["v"]), float(row["range"])
            if not previous < index < frame_count or not (0 <= u < width and 0 <= v < height):
                raise ValueError("QUT label frame order or original-frame coordinates invalid")
            if not all(math.isfinite(x) for x in (u, v, source_range)) or source_range <= 0:
                raise ValueError("QUT label has invalid numeric value")
            labels[index] = (u, v)
            previous = index
    if not labels or first_global_frame + frame_count - 1 != last_global_frame:
        raise ValueError("video frame count disagrees with published global-frame bounds")
    if max(labels) != frame_count - 1:
        raise ValueError("last QUT label does not align with last decoded frame")
    return labels


def unique_near(center: tuple[float, float], candidates: list[tuple[float, float]],
                radius_px: float = MATCH_RADIUS_PX) -> int | None:
    """Return a candidate only when exactly one lies in the fixed pixel gate."""
    matches = [i for i, point in enumerate(candidates) if math.dist(center, point) <= radius_px]
    return matches[0] if len(matches) == 1 else None


def interpolate_label(target_s: float, labeled_times: list[float],
                      labeled_points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Evaluator-only target centroid; interpolate across adjacent labeled 25-fps frames."""
    i = bisect.bisect_left(labeled_times, target_s)
    if i < len(labeled_times) and abs(labeled_times[i] - target_s) < 1e-9:
        return labeled_points[i]
    if i == 0 or i == len(labeled_times):
        return None
    t0, t1 = labeled_times[i - 1], labeled_times[i]
    if t1 - t0 > 0.041:  # no interpolation across missing labels
        return None
    w = (target_s - t0) / (t1 - t0)
    a, b = labeled_points[i - 1], labeled_points[i]
    return a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])


def run_runtime(video: Path) -> tuple[dict, dict]:
    """Run vision, tracking and causal forecasts without opening the label file."""
    detector_cfg = baseline.load_config()
    model_path = baseline.verified_model_path(detector_cfg)
    runner = baseline.OnnxRuntimeRunner(model_path)
    stages = pipeline.BaselineStages(detector_cfg, runner)
    video_run = pipeline.process_video(video, stages, sequence_id="qut-s5", timestamp_policy="pts")
    by_frame = defaultdict(list)
    for detection in video_run["detections"]:
        by_frame[detection.frame_index].append(detection)
    tracker = KalmanTracker("qut-s5", tracker_config())
    forecasters = {name: TrajectoryForecaster("qut-s5", name) for name in MODELS}
    observed = defaultdict(list)
    predictions = defaultdict(list)
    for frame in video_run["frames"]:
        index, t = frame["frame_index"], frame["timestamp_s"]
        kept = [d for d in by_frame[index] if d.score >= TRACK_SCORE]
        states = tracker.step(t, kept)
        for state in states:
            if state.observed and state.measurement_center_px is not None:
                observed[index].append((state.track_id, state.measurement_center_px))
                for forecaster in forecasters.values():
                    forecaster.update(state.track_id, t, state.measurement_center_px)
            if state.lifecycle == CONFIRMED:
                for model, forecaster in forecasters.items():
                    predictions[index].extend(forecaster.forecast(state.track_id, t, HORIZONS_S))
    summary = {
        "decoded_frames": video_run["decoded_frames"],
        "width_px": video_run["width"], "height_px": video_run["height"],
        "nominal_fps": video_run["nominal_fps"],
        "timestamp_source": video_run["timestamp_source"],
        "first_timestamp_s": video_run["first_timestamp_s"],
        "last_timestamp_s": video_run["frames"][-1]["timestamp_s"],
        "detector_outputs_at_floor": len(video_run["detections"]),
        "detections_at_tracking_score": sum(d.score >= TRACK_SCORE for group in by_frame.values()
                                            for d in group),
        "frames_with_detector_output": len({d.frame_index for d in video_run["detections"]}),
        "tracker_counts": tracker.counters,
        "observed_track_measurements": sum(map(len, observed.values())),
        "forecast_records": sum(map(len, predictions.values())),
        "runtime": runner.name,
        "model_sha256": detector_cfg["model"]["sha256"],
    }
    return summary, {"frames": video_run["frames"], "detections": by_frame,
                     "observed": observed, "predictions": predictions}


def describe_errors(errors: list[float]) -> dict:
    return {"support": len(errors), "mean_px": statistics.mean(errors) if errors else None,
            "median_px": statistics.median(errors) if errors else None,
            "max_px": max(errors) if errors else None,
            "availability_reason": None if errors else "no uniquely associated target observations"}


def evaluate(runtime: dict, labels: dict[int, tuple[float, float]]) -> dict:
    """Apply QUT approximate truth only after all runtime outputs are fixed."""
    times = {f["frame_index"]: f["timestamp_s"] for f in runtime["frames"]}
    ordered = sorted(labels)
    labeled_times = [times[i] for i in ordered]
    labeled_points = [labels[i] for i in ordered]
    detection_errors, observation_errors = [], []
    forecast_errors = defaultdict(list)
    associated_issues = 0
    for index, truth in labels.items():
        detections = runtime["detections"].get(index, [])
        centers = [((d.bbox.x_min + d.bbox.x_max) / 2,
                    (d.bbox.y_min + d.bbox.y_max) / 2) for d in detections]
        match = unique_near(truth, centers)
        if match is not None:
            detection_errors.append(math.dist(truth, centers[match]))
        observations = runtime["observed"].get(index, [])
        match = unique_near(truth, [point for _, point in observations])
        if match is None:
            continue
        track_id, point = observations[match]
        observation_errors.append(math.dist(truth, point))
        associated_issues += 1
        for prediction in runtime["predictions"].get(index, []):
            if prediction.track_id != track_id or prediction.center is None:
                continue
            target = interpolate_label(prediction.target_time_s, labeled_times, labeled_points)
            if target is not None:
                forecast_errors[prediction.model_name, prediction.horizon_s].append(
                    math.dist(prediction.center, target))
    return {
        "label_count": len(labels), "label_first_video_frame": min(labels),
        "label_last_video_frame": max(labels),
        "association_gate_px": MATCH_RADIUS_PX,
        "detector_centroid_error": describe_errors(detection_errors),
        "tracked_observation_centroid_error": describe_errors(observation_errors),
        "associated_forecast_issue_frames": associated_issues,
        "forecast_centroid_error": {
            model: {str(h): describe_errors(forecast_errors[model, h]) for h in HORIZONS_S}
            for model in MODELS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/raw/qut/S5.mp4")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/raw/qut/S5_labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/qut_video_pilot.json")
    args = parser.parse_args()
    if sha256(args.video) != VIDEO_SHA256 or sha256(args.labels) != LABEL_SHA256:
        raise ValueError("QUT source hash mismatch; do not run on an unpinned revision")
    if args.video.stat().st_size + args.labels.stat().st_size > 2_000_000:
        raise ValueError("QUT source exceeds reviewed download budget")
    summary, runtime = run_runtime(args.video)
    labels = load_labels(args.labels, frame_count=summary["decoded_frames"],
                         width=summary["width_px"], height=summary["height_px"])
    evaluation = evaluate(runtime, labels)
    report = {
        "schema_version": 1,
        "scope": "one real fixed-camera compressed QUT aircraft clip; descriptive pilot, not a benchmark",
        "source": {"release_revision": REVISION, "video_url": VIDEO_URL, "label_url": LABEL_URL,
                   "sequence_bounds_url": SEQUENCE_BOUNDS_URL,
                   "rights_url": "https://open.qcr.ai/dataset/aircraft-collision-course/",
                   "rights": "BSD-3-Clause per publisher", "video_sha256": VIDEO_SHA256,
                   "label_sha256": LABEL_SHA256, "video_bytes": args.video.stat().st_size,
                   "label_bytes": args.labels.stat().st_size,
                   "published_first_global_frame": SOURCE_FIRST_FRAME,
                   "published_last_global_frame": SOURCE_LAST_FRAME,
                   "frame_offset_evidence": "target-info.csv S5 bounds 57750..59426; 1677 decoded frames and final label 59426 independently align index 0..1676",
                   "label_schema": ["row", "frame", "u", "v", "range"],
                   "label_coordinates": "approximate original-frame target centroid pixels",
                   "label_range": "published source column, unit unverified; not used by runtime or evaluator"},
        "runtime": summary, "evaluation": evaluation,
        "unavailable": {
            "AP": "approximate target centroids; no exhaustive boxes or verified negative frames",
            "HOTA_IDF1": "no verified all-aircraft identity labels or complete per-frame boxes",
            "calibration_ENU_bearing": "source camera intrinsics and surveyed pose not supplied",
            "predicted_range": "monocular runner has no validated range source; source range labels not runtime inputs",
            "live_accuracy": "recorded-video replay only; no live camera validation",
            "confidence_intervals": "one sequence and approximate labels; no independent sampling units"},
        "limitations": ["Association is conditioned on exactly one runtime center within a fixed 50 px radius of the approximate target label; unmatched frames are excluded from error summaries.",
                        "Tracking score 0.3 and model postprocess floor 0.1 are untuned existing settings.",
                        "A label-aligned detected target in this one clip does not establish general aircraft detection or forecast quality."],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "runtime": summary,
                      "evaluation": evaluation}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
