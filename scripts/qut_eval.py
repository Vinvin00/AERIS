"""QUT multi-sequence evaluation harness (AERIS-035, replacing AERIS-032); labels only reach the evaluator.

    .venv/bin/python scripts/qut_eval.py fetch
    .venv/bin/python scripts/qut_eval.py run --split dev --name baseline
    .venv/bin/python scripts/qut_eval.py compare benchmarks/qut_eval/A-dev.json benchmarks/qut_eval/B-dev.json

Runs the detector (profile), tracker (config file, AERIS-027 low-score pass) and causal forecasters
on each pinned QUT video, fixes every runtime output, and only then opens the approximate centroid
labels. Measures, split and selection rule are pre-registered in PLAN-011. Frame-pass detections
are cached per profile in Git-ignored data/processed/qut_cache; per-forecast errors used by
``compare`` go to Git-ignored data/processed/qut_eval_raw. Pixel measures are descriptive: no AP,
HOTA/IDF1 or calibrated accuracy is computed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from qut_video_pilot import MATCH_RADIUS_PX, interpolate_label, unique_near  # noqa: E402

MANIFEST = ROOT / "configs/qut/sequences.json"
CANDIDATES = ROOT / "configs/qut/candidates.json"
DATA = ROOT / "data/raw/qut"
CACHE = ROOT / "data/processed/qut_cache"
RAW = ROOT / "data/processed/qut_eval_raw"
TRACK_SCORE = 0.3          # existing untuned CLI default (aeris.cli.DEFAULT_MIN_SCORE)
HORIZONS_S = (0.1, 0.25, 0.5, 1.0, 2.0)
BUDGET_BYTES = 40_000_000
RULE_COUNTS = ("off_target_observations", "target_track_impurity", "non_target_confirmed_tracks")


class EvalError(ValueError):
    """Pinned source, alignment or split rule violated."""


# -- sources -------------------------------------------------------------------------------
def load_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(Path(path).read_text())
    ids = [s["id"] for s in manifest["sequences"]]
    if len(set(ids)) != len(ids):
        raise EvalError("duplicate sequence id in manifest")
    for s in manifest["sequences"]:
        if s["split"] not in ("dev", "holdout"):
            raise EvalError(f"{s['id']}: unknown split {s['split']!r}")
    return manifest


def split_overlaps(manifest: dict) -> list[dict]:
    """Same-date global-frame overlaps between a development and a holdout sequence."""
    dev = [s for s in manifest["sequences"] if s["split"] == "dev"]
    hold = [s for s in manifest["sequences"] if s["split"] == "holdout"]
    out = []
    for a in dev:
        for b in hold:
            lo, hi = max(a["start_frame"], b["start_frame"]), min(a["end_frame"], b["end_frame"])
            if a["date"] == b["date"] and lo <= hi:
                out.append({"dev": a["id"], "holdout": b["id"], "date": a["date"],
                            "global_frames": [lo, hi], "frames": hi - lo + 1})
    return out


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_file(path: Path, entry: dict) -> None:
    """Size plus every published or measured digest the manifest carries for this file."""
    if not path.is_file():
        raise EvalError(f"missing {path}; run the fetch command")
    data = path.read_bytes()
    if len(data) != entry["bytes"]:
        raise EvalError(f"{path.name}: size {len(data)} != pinned {entry['bytes']}")
    if "sha256" in entry and hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise EvalError(f"{path.name}: SHA-256 mismatch with pinned value")
    if "git_blob_sha1" in entry and git_blob_sha1(data) != entry["git_blob_sha1"]:
        raise EvalError(f"{path.name}: Git blob SHA-1 mismatch with published value")


def sequence_files(sequence: dict) -> list[dict]:
    return [sequence["video"], sequence["target_labels"], *sequence["other_labels"]]


def all_files(manifest: dict) -> list[dict]:
    return [manifest["target_info"], *(f for s in manifest["sequences"] for f in sequence_files(s))]


def fetch(manifest: dict, data_dir: Path = DATA) -> None:
    files = all_files(manifest)
    total = sum(f["bytes"] for f in files)
    if total > BUDGET_BYTES:
        raise EvalError(f"pinned sources total {total} bytes, above the {BUDGET_BYTES} budget")
    data_dir.mkdir(parents=True, exist_ok=True)
    for entry in files:
        path = data_dir / entry["local"]
        if not path.is_file():
            url = f"{manifest['base_url']}/{entry['remote']}"
            print(f"fetch {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (pinned https)
                path.write_bytes(response.read())
        check_file(path, entry)


# -- labels --------------------------------------------------------------------------------
def read_centroids(path: Path, *, first_global_frame: int, frame_count: int,
                   width: int, height: int, drop_outside: bool = False) -> tuple[dict, int]:
    """Published global-frame centroid CSV -> ({zero-based video frame: (u, v)}, dropped).

    Target labels must lie inside the video. Other-aircraft labels may start before or end after
    the published video bounds; with ``drop_outside`` those rows are dropped and counted, never
    shifted."""
    out, dropped = {}, 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"frame", "u", "v"} <= set(reader.fieldnames):
            raise EvalError(f"{Path(path).name}: unexpected columns {reader.fieldnames}")
        previous = -1
        for row in reader:
            index = int(row["frame"]) - first_global_frame
            u, v = float(row["u"]), float(row["v"])
            if drop_outside and not 0 <= index < frame_count:
                dropped += 1
                continue
            if not previous < index < frame_count:
                raise EvalError(f"{Path(path).name}: label frame outside video or out of order")
            if not (math.isfinite(u) and math.isfinite(v) and 0 <= u < width and 0 <= v < height):
                raise EvalError(f"{Path(path).name}: label outside original-frame coordinates")
            out[index] = (u, v)
            previous = index
    if not out and not drop_outside:
        raise EvalError(f"{Path(path).name}: no labels")
    return out, dropped


def check_frame_count(sequence: dict, decoded: int) -> None:
    if decoded != sequence["decoded_frames"]:
        raise EvalError(f"{sequence['id']}: decoded {decoded} frames, manifest pins {sequence['decoded_frames']}")


def check_alignment(sequence: dict, target: dict, decoded: int) -> None:
    """The pinned alignment rule: the last target label falls on the last decoded frame. It holds
    with the published start_frame for 12 sequences; S1, S2 and S4 disagree with their published
    bounds by one frame, and the manifest records their first decoded frame as inferred."""
    if max(target) != decoded - 1:
        raise EvalError(f"{sequence['id']}: last target label is video frame {max(target)}, "
                        f"not the last decoded frame {decoded - 1}")


# -- synthetic pan (development-only secondary measure) -------------------------------------------
PAN_FPS = 25.0   # QUT nominal rate; the pan is a function of frame index only


def pan_offset(index: int, amplitude_px: float, seed: int) -> tuple[float, float]:
    """Smooth, seeded camera offset for frame ``index``: three sinusoids per axis, 0.05-0.5 Hz at
    25 frames/s, peak at most ``amplitude_px``. Known exactly, so labels shift identically."""
    rng = np.random.default_rng(seed)
    freqs, phases = rng.uniform(0.05, 0.5, (2, 3)), rng.uniform(0, 2 * math.pi, (2, 3))
    t = index / PAN_FPS
    return tuple(float(amplitude_px / 3 * np.sin(2 * math.pi * freqs[a] * t + phases[a]).sum())
                 for a in range(2))


def shift_labels(labels: dict, offsets: dict, width: int, height: int) -> tuple[dict, int]:
    out, dropped = {}, 0
    for index, (u, v) in labels.items():
        dx, dy = offsets[index]
        if 0 <= u + dx < width and 0 <= v + dy < height:
            out[index] = (u + dx, v + dy)
        else:
            dropped += 1
    return out, dropped


class PannedStages:
    """Wrap detector stages so every frame is translated by its known pan offset (grey border)
    before detection. Boxes are then in the panned frame's pixels, like the shifted labels."""

    def __init__(self, stages, offsets: dict):
        self.stages, self.offsets, self.index = stages, offsets, 0
        self.name = f"{stages.name}+synthetic-pan"

    def preprocess(self, image):
        import cv2
        dx, dy = self.offsets[self.index]
        self.index += 1
        shifted = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                                 (image.shape[1], image.shape[0]), borderValue=(114, 114, 114))
        return self.stages.preprocess(shifted)

    def infer(self, prepared):
        return self.stages.infer(prepared)

    def postprocess(self, raw, prepared, **kwargs):
        return self.stages.postprocess(raw, prepared, **kwargs)


# -- runtime (never opens labels) ------------------------------------------------------------
def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def detect(video: Path, sequence_id: str, *, profile: str, video_sha256: str, frame_count: int,
           pan: dict | None = None, cache_dir: Path | None = CACHE) -> dict:
    """Frame-pass detections for one video, cached by video SHA-256, profile, detector config,
    weights and pan; the cache holds runtime outputs only."""
    from aeris.data.contracts import DetectionRecord
    from aeris.detection import profiles
    from aeris.video import pipeline

    config = profiles.config_path(profile).read_bytes()
    key = hashlib.sha256(b"|".join([video_sha256.encode(), profile.encode(), config,
                                    json.loads(config)["model"]["sha256"].encode(),
                                    json.dumps(pan, sort_keys=True).encode()])).hexdigest()[:16]
    path = cache_dir / f"{sequence_id}-{profile}-{key}.json" if cache_dir else None
    if path and path.is_file():
        cached = json.loads(path.read_text())
        cached["detections"] = [DetectionRecord.from_dict(d) for d in cached["detections"]]
        return {**cached, "cache": "hit"}
    stages = profiles.build(profile)
    offsets = None
    if pan:
        offsets = {i: pan_offset(i, pan["amplitude_px"], pan["seed"]) for i in range(frame_count)}
        stages = PannedStages(stages, offsets)
    run = pipeline.process_video(video, stages, sequence_id=sequence_id, timestamp_policy="pts")
    per_frame_ms = [1000 * f["timings_s"]["total"] for f in run["frames"]]
    out = {"frames": [{"frame_index": f["frame_index"], "timestamp_s": f["timestamp_s"]} for f in run["frames"]],
           "width": run["width"], "height": run["height"], "decoded_frames": run["decoded_frames"],
           "detector": run["detector"],
           "detect_ms": {"p50": percentile(per_frame_ms, 50), "p95": percentile(per_frame_ms, 95)},
           "pan_offsets": {str(k): v for k, v in offsets.items()} if offsets else None, "cache_key": key}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**out, "detections": [d.to_dict() for d in run["detections"]]}))
    return {**out, "detections": list(run["detections"]), "cache": "miss"}


def track(detected: dict, sequence_id: str, tracker_config: dict) -> dict:
    """Tracker (low-score pass) and causal forecasters over fixed detections; no labels."""
    from aeris.estimation.trajectory import MODELS, TrajectoryForecaster
    from aeris.tracking.tracker import CONFIRMED, KalmanTracker

    by_frame = defaultdict(list)
    for d in detected["detections"]:
        by_frame[d.frame_index].append(d)
    tracker = KalmanTracker(sequence_id, tracker_config)
    forecasters = {name: TrajectoryForecaster(sequence_id, name) for name in MODELS}
    observed, confirmed_ids, predictions, step_ms = defaultdict(list), set(), defaultdict(list), []
    for frame in detected["frames"]:
        index, t = frame["frame_index"], frame["timestamp_s"]
        t0 = time.perf_counter()
        for state in tracker.step(t, by_frame[index], high_score=TRACK_SCORE):
            if state.observed:
                observed[index].append((state.track_id, state.lifecycle, state.measurement_center_px))
                for f in forecasters.values():
                    f.update(state.track_id, t, state.measurement_center_px)
            if state.lifecycle == CONFIRMED:
                confirmed_ids.add(state.track_id)
                for f in forecasters.values():
                    predictions[index].extend(f.forecast(state.track_id, t, HORIZONS_S))
        step_ms.append(1000 * (time.perf_counter() - t0))
    return {"frames": detected["frames"], "detections": by_frame, "observed": observed,
            "confirmed_ids": confirmed_ids, "predictions": predictions,
            "counts": {"decoded_frames": detected["decoded_frames"],
                       "detections_at_floor": len(detected["detections"]),
                       "detections_at_track_score": sum(d.score >= TRACK_SCORE for d in detected["detections"]),
                       "tracker": dict(tracker.counters), "confirmed_tracks": len(confirmed_ids)},
            "detect_ms": detected["detect_ms"], "detection_cache": detected["cache"],
            "track_and_forecast_ms": {"p50": percentile(step_ms, 50), "p95": percentile(step_ms, 95)}}


# -- evaluator (labels opened only here) -------------------------------------------------------
def _centre(d) -> tuple[float, float]:
    return (d.bbox.x_min + d.bbox.x_max) / 2, (d.bbox.y_min + d.bbox.y_max) / 2


def fragments_and_switches(ids_in_order: list[str]) -> tuple[int, int]:
    """Distinct track IDs that carried the target, and ID changes between consecutive
    matched frames (a gap between matched frames is not a switch by itself)."""
    return len(set(ids_in_order)), sum(a != b for a, b in zip(ids_in_order, ids_in_order[1:]))


def evaluate(runtime: dict, target: dict, others: list[dict], radius_px: float = MATCH_RADIUS_PX) -> dict:
    """PLAN-011 measures 1-8 for one sequence. Observations are assigned measurement centres of
    observed tracks; coasting states never count."""
    times = {f["frame_index"]: f["timestamp_s"] for f in runtime["frames"]}
    ordered = sorted(target)
    label_times, label_points = [times[i] for i in ordered], [target[i] for i in ordered]
    det_hits = track_hits = off_target = 0
    carried, forecast_errors = [], {}
    for index in ordered:
        truth = target[index]
        dets = [_centre(d) for d in runtime["detections"].get(index, []) if d.score >= TRACK_SCORE]
        det_hits += unique_near(truth, dets, radius_px) is not None
        confirmed = [(tid, c) for tid, life, c in runtime["observed"].get(index, []) if life == "confirmed"]
        aircraft = [truth, *(o[index] for o in others if index in o)]
        off_target += sum(all(math.dist(c, p) > radius_px for p in aircraft) for _, c in confirmed)
        match = unique_near(truth, [c for _, c in confirmed], radius_px)
        if match is None:
            continue
        track_hits += 1
        track_id = confirmed[match][0]
        carried.append(track_id)
        for p in runtime["predictions"].get(index, []):
            if p.track_id != track_id or p.center is None:
                continue
            goal = interpolate_label(p.target_time_s, label_times, label_points)
            if goal is not None:
                forecast_errors[f"{index}|{p.model_name}@{p.horizon_s}"] = math.dist(p.center, goal)
    target_ids = set(carried)
    impurity = 0
    swapped, near_aircraft = set(), set()
    for index, items in runtime["observed"].items():
        labelled = [o[index] for o in others if index in o]
        points = labelled + ([target[index]] if index in target else [])
        for tid, life, c in items:
            if any(math.dist(c, p) <= radius_px for p in points):
                near_aircraft.add(tid)
            if life == "confirmed" and tid in target_ids and index in target \
                    and math.dist(c, target[index]) > radius_px:
                impurity += 1
        confirmed = [(tid, c) for tid, life, c in items if life == "confirmed"]
        truth = target.get(index)
        for p in labelled:
            m = unique_near(p, [c for _, c in confirmed], radius_px)
            if m is None or confirmed[m][0] not in target_ids:
                continue
            if truth is None or math.dist(confirmed[m][1], truth) > radius_px:   # [Q1]
                swapped.add(confirmed[m][0])
    ambiguous = sum(any(index in o and math.dist(point, o[index]) <= 2 * radius_px for o in others)
                    for index, point in target.items())
    fragments, switches = fragments_and_switches(carried)
    return {"labelled_target_frames": len(target), "target_detection_hits": det_hits,
            "target_track_hits": track_hits, "target_fragments": fragments,
            "target_id_switches": switches, "off_target_observations": off_target,
            "target_track_impurity": impurity, "identity_swaps": len(swapped),
            "ambiguous_swap_frames": ambiguous,
            "non_target_confirmed_tracks": len(runtime["confirmed_ids"] - near_aircraft),
            "forecast_errors_px": forecast_errors}


def describe(values: list[float]) -> dict:
    if not values:
        return {"support": 0, "median_px": None, "p90_px": None,
                "availability_reason": "no forecast from a uniquely target-matched confirmed track had a labelled target time"}
    ordered = sorted(values)
    return {"support": len(values), "median_px": statistics.median(ordered),
            "p90_px": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
            "availability_reason": None}


def by_model_horizon(errors: dict) -> dict:
    grouped = defaultdict(list)
    for key, value in errors.items():
        grouped[key.split("|", 1)[1]].append(value)
    return {k: describe(v) for k, v in sorted(grouped.items())}


COUNT_KEYS = ("labelled_target_frames", "target_detection_hits", "target_track_hits", "target_fragments",
              "target_id_switches", "off_target_observations", "target_track_impurity", "identity_swaps",
              "ambiguous_swap_frames", "non_target_confirmed_tracks")


def summarise(result: dict) -> dict:
    n = result["labelled_target_frames"]
    out = {k: result[k] for k in COUNT_KEYS}
    out["target_detection_recall"] = result["target_detection_hits"] / n if n else None
    out["target_track_recall"] = result["target_track_hits"] / n if n else None
    out["forecast_centroid_error"] = by_model_horizon(result["forecast_errors_px"])
    return out


def pool(results: dict[str, dict]) -> dict:
    """Frame-weighted pooling: counts are summed over sequences."""
    pooled = {k: sum(r[k] for r in results.values()) for k in COUNT_KEYS}
    pooled["forecast_errors_px"] = {f"{sid}:{k}": v for sid, r in results.items()
                                    for k, v in r["forecast_errors_px"].items()}
    return summarise(pooled)


# -- selection rule and comparison (PLAN-011) -----------------------------------------------------
def bound(baseline: float) -> float:
    return max(1.25 * baseline, baseline + 5)


def passes_rule(candidate: dict, baseline: dict) -> tuple[bool, list[str]]:
    """PLAN-011 rule on pooled measures; returns (passes, failed conditions)."""
    failed = []
    if candidate["target_track_recall"] < baseline["target_track_recall"]:
        failed.append("target_track_recall below baseline")
    if candidate["target_fragments"] > baseline["target_fragments"]:
        failed.append("target_fragments above baseline")
    if candidate["identity_swaps"] > baseline["identity_swaps"]:
        failed.append("identity_swaps above baseline")
    for key in RULE_COUNTS:
        if candidate[key] > bound(baseline[key]):
            failed.append(f"{key} above max(1.25b, b+5)")
    return not failed, failed


def rank_key(pooled: dict, listed_order: int) -> tuple:
    """Choice among passing candidates: fewest fragments, highest track recall, fewest
    off-target observations, then pre-written list order."""
    return (pooled["target_fragments"], -pooled["target_track_recall"],
            pooled["off_target_observations"], listed_order)


def compare(a: dict, b: dict, raw_a: dict, raw_b: dict) -> dict:
    """Rule verdict of B against baseline A, per-sequence recall regressions and forecast error
    on the common subset of (sequence, issue frame, model, horizon) keys."""
    ok, failed = passes_rule(b["pooled"], a["pooled"])
    common = raw_a.keys() & raw_b.keys()
    regress = sorted(s for s in a["sequences"].keys() & b["sequences"].keys()
                     if b["sequences"][s]["target_track_recall"] < a["sequences"][s]["target_track_recall"])
    return {"baseline": a["name"], "candidate": b["name"], "split": b["split"],
            "passes_plan011_rule": ok, "failed_conditions": failed,
            "sequences_with_lower_track_recall": regress,
            "forecast_common_subset": {
                "common": len(common), "only_baseline": len(raw_a.keys() - common),
                "only_candidate": len(raw_b.keys() - common),
                "baseline": by_model_horizon({k: raw_a[k] for k in common}),
                "candidate": by_model_horizon({k: raw_b[k] for k in common})}}


# -- candidate list (PLAN-011 [Q7]) -------------------------------------------------------------
def load_candidate(name: str, path: Path = CANDIDATES) -> tuple[dict, dict, str]:
    """The named variant from the pre-written list, the list's pan settings, and its SHA-256.
    Profile and tracker config come only from the list, so an unlisted variant cannot run."""
    data = Path(path).read_bytes()
    listed = json.loads(data)
    names = [c["name"] for c in listed["candidates"]]
    if len(set(names)) != len(names):
        raise EvalError("duplicate candidate name in list")
    per_task = defaultdict(int)
    for c in listed["candidates"]:
        per_task[c["task"]] += 1
    if any(n > 12 for n in per_task.values()):
        raise EvalError("more than 12 candidates for one task (PLAN-011)")
    if name not in names:
        raise EvalError(f"variant {name!r} is not on the candidate list {Path(path).name}")
    candidate = listed["candidates"][names.index(name)]
    config = ROOT / candidate["tracker_config"]
    if hashlib.sha256(config.read_bytes()).hexdigest() != candidate["tracker_config_sha256"]:
        raise EvalError(f"{candidate['tracker_config']} changed since it was listed for {name!r}")
    return candidate, listed["synthetic_pan"], hashlib.sha256(data).hexdigest()


# -- split enforcement -----------------------------------------------------------------------
def select_sequences(manifest: dict, split: str, only: list[str] | None, selection_record: Path | None,
                     pan: bool = False):
    if split == "holdout":
        if pan:
            raise EvalError("the synthetic pan is a development-only measure")
        if selection_record is None or not Path(selection_record).is_file():
            raise EvalError("holdout runs need --selection-record pointing to the written candidate "
                            "selection (PLAN-011); the holdout never chooses parameters")
        if Path(selection_record).resolve().parent != (ROOT / "docs/decisions").resolve():
            raise EvalError("--selection-record must be a decision record in docs/decisions/")
    chosen = [s for s in manifest["sequences"] if s["split"] == split]
    if only:
        unknown = set(only) - {s["id"] for s in chosen}
        if unknown:
            raise EvalError(f"sequences not in the {split} split: {sorted(unknown)}")
        chosen = [s for s in chosen if s["id"] in only]
    return chosen


def provenance(profile: str, tracker_config: Path) -> dict:
    from aeris.detection import profiles
    config = profiles.config_path(profile)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "python", "configs", "scripts"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return {"profile": profile, "profile_tiles": profiles.PROFILES[profile].tiles,
            "profile_ownership": ("accepted AERIS-006 whole-frame baseline" if profile == "fast" else
                                  "external, unaccepted profile/tiling code (commits 79a7cb3, c480bf0)"),
            "detector_config": str(config.relative_to(ROOT)),
            "detector_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weights_sha256": json.loads(config.read_text())["model"]["sha256"],
            "tracker_config": str(tracker_config.resolve().relative_to(ROOT)),
            "tracker_config_sha256": hashlib.sha256(tracker_config.read_bytes()).hexdigest(),
            "code_commit": commit, "working_tree_dirty": dirty,
            "track_score": TRACK_SCORE, "low_score_pass": True,
            "association_gate_px": MATCH_RADIUS_PX, "horizons_s": list(HORIZONS_S)}


def run(args, manifest: dict) -> dict:
    candidate, pan_settings, list_sha256 = load_candidate(args.name)
    profile, tracker_path = candidate["profile"], ROOT / candidate["tracker_config"]
    chosen = select_sequences(manifest, args.split, args.sequences, args.selection_record, args.pan)
    tracker_config = json.loads(tracker_path.read_text())
    check_file(DATA / manifest["target_info"]["local"], manifest["target_info"])
    results, per_sequence, raw = {}, {}, {}
    for s in chosen:
        for entry in sequence_files(s):
            check_file(DATA / entry["local"], entry)
        sequence_id = f"qut-{s['id'].lower()}"
        frame_count = s["decoded_frames"]
        pan = dict(pan_settings) if args.pan else None
        detected = detect(DATA / s["video"]["local"], sequence_id, profile=profile,
                          video_sha256=s["video"]["sha256"], frame_count=frame_count, pan=pan,
                          cache_dir=None if args.no_cache else CACHE)
        check_frame_count(s, detected["decoded_frames"])
        runtime = track(detected, sequence_id, tracker_config)    # runtime outputs now fixed
        bounds = dict(first_global_frame=s["first_decoded_global_frame"], frame_count=detected["decoded_frames"],
                      width=detected["width"], height=detected["height"])
        target, _ = read_centroids(DATA / s["target_labels"]["local"], **bounds)
        check_alignment(s, target, detected["decoded_frames"])
        others, outside, pan_dropped = [], 0, 0
        for o in s["other_labels"]:
            labels, n = read_centroids(DATA / o["local"], **bounds, drop_outside=True)
            others.append(labels)
            outside += n
        if pan:
            offsets = {int(k): tuple(v) for k, v in detected["pan_offsets"].items()}
            target, pan_dropped = shift_labels(target, offsets, detected["width"], detected["height"])
            others = [shift_labels(o, offsets, detected["width"], detected["height"])[0] for o in others]
        result = evaluate(runtime, target, others)
        results[s["id"]] = result
        raw.update({f"{s['id']}:{k}": v for k, v in result["forecast_errors_px"].items()})
        per_sequence[s["id"]] = {**summarise(result), "runtime": runtime["counts"],
                                 "resolution_px": [detected["width"], detected["height"]],
                                 "alignment": s["alignment"],
                                 "other_aircraft_labels_outside_video": outside,
                                 "target_labels_panned_out_of_frame": pan_dropped,
                                 "detection_cache": runtime["detection_cache"],
                                 "detect_ms": runtime["detect_ms"],
                                 "track_and_forecast_ms": runtime["track_and_forecast_ms"]}
        print(s["id"], json.dumps({k: per_sequence[s["id"]][k] for k in
              ("target_detection_recall", "target_track_recall", "target_fragments",
               "off_target_observations", "non_target_confirmed_tracks")}), flush=True)
    return {
        "schema_version": 1, "name": args.name, "split": args.split,
        "scope": "QUT fixed ground-camera clips with approximate centroid labels; descriptive, not a benchmark",
        "plan": "docs/decisions/PLAN-011-qut-eval-tracker-guided-detection.md",
        "variant": {**provenance(profile, tracker_path), "candidate": candidate,
                    "candidate_list_sha256": list_sha256, "stride": 1, "timestamp_policy": "pts"},
        "synthetic_pan": ({**pan_settings, "note": "development-only secondary measure; translation only"}
                          if args.pan else None),
        "selection_record": ({"path": str(Path(args.selection_record).resolve().relative_to(ROOT)),
                              "sha256": hashlib.sha256(Path(args.selection_record).read_bytes()).hexdigest()}
                             if args.selection_record else None),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "split_overlaps": split_overlaps(manifest),
        "sequences": per_sequence, "pooled": pool(results),
        "unavailable": {
            "AP": "approximate target centroids; no exhaustive boxes or verified negative frames",
            "HOTA_IDF1": "identity labels cover the target and some other aircraft only; no complete per-frame boxes",
            "false_alarm_rate": "off-target and non-target counts are proxies; unlabelled aircraft, birds and label gaps are not separable",
            "confidence_intervals": "few sequences from one camera on two days; no independent sampling design"},
    }, raw


def output_path(args) -> Path:
    """Result path. A sequence subset gets its own name so it never replaces a full-split result,
    and a holdout result is never overwritten (PLAN-011 records every holdout run)."""
    stem = f"{args.name}-{args.split}" + ("-pan" if args.pan else "")
    if args.sequences:
        stem += "-" + "-".join(sorted(args.sequences))
    out = args.output_dir / f"{stem}.json"
    if args.split == "holdout" and out.exists():
        raise EvalError(f"{out} exists; every holdout run is kept, so use a new candidate name")
    return out


def load_raw(report: dict) -> dict:
    """Per-forecast errors belonging to ``report``; refuse a file another run has replaced."""
    meta = report["raw_forecast_errors"]
    data = (ROOT / meta["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise EvalError(f"{meta['path']} does not belong to report {report['name']}; re-run it")
    return json.loads(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download and verify the pinned QUT sources")
    r = sub.add_parser("run", help="evaluate one variant on one split")
    r.add_argument("--split", choices=("dev", "holdout"), required=True)
    r.add_argument("--name", required=True, help="candidate name from configs/qut/candidates.json")
    r.add_argument("--sequences", nargs="+", default=None)
    r.add_argument("--selection-record", type=Path, default=None)
    r.add_argument("--pan", action="store_true", help="dev-only synthetic pan with the list's settings")
    r.add_argument("--no-cache", action="store_true", help="ignore and do not write the detection cache")
    r.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/qut_eval")
    c = sub.add_parser("compare", help="PLAN-011 rule of CANDIDATE against BASELINE")
    c.add_argument("baseline", type=Path)
    c.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest()
    if args.command == "fetch":
        fetch(manifest)
        return 0
    if args.command == "compare":
        a, b = (json.loads(p.read_text()) for p in (args.baseline, args.candidate))
        if a["split"] != b["split"] or set(a["sequences"]) != set(b["sequences"]) \
            or bool(a["synthetic_pan"]) != bool(b["synthetic_pan"]):
            raise EvalError("compare needs reports on the same split and sequences")
        raw_a, raw_b = (load_raw(r) for r in (a, b))
        print(json.dumps(compare(a, b, raw_a, raw_b), indent=1, sort_keys=True))
        return 0
    out = output_path(args)
    report, raw = run(args, manifest)
    raw_bytes = json.dumps(raw, sort_keys=True).encode()
    report["raw_forecast_errors"] = {"path": str((RAW / out.name).relative_to(ROOT)),
                                     "sha256": hashlib.sha256(raw_bytes).hexdigest(), "keys": len(raw)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / out.name).write_bytes(raw_bytes)
    print(json.dumps({"output": str(out), "pooled": report["pooled"]}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
